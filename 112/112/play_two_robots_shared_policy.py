#!/usr/bin/env python3
"""
Two-robot shared-policy rollout for your PyBullet badminton model.

Purpose:
- Load TWO humanoid robots: left + right.
- Load ONE trained PPO policy.
- Reuse the same policy on both sides by mirroring the right robot's observation.
- Alternate hitting according to shuttle side.
- Use racket-state-based shuttle rebound:
      vout = vin - 2(vin·n)n + 2(vracket·n)n

Save as:
    play_two_robots_shared_policy.py

Run:
    python play_two_robots_shared_policy.py --model runs\badminton_final.zip --vecnorm runs\vecnormalize_final.pkl

Notes:
- This is a rollout/demo script, not a new trainer.
- It uses your existing trained single-side PPO policy.
- Because your simplified humanoid is not a stable dynamic biped, robots are loaded with fixed bases by default.
"""

import argparse
import os
import time
from collections import deque
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from physics_shuttle import ShuttlePhysics


# Must match train_rl_badminton.py
JOINT_NAMES = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_shoulder_pitch", "left_elbow", "left_wrist",
    "right_shoulder_pitch", "right_elbow", "right_wrist",
]

JOINT_LIMITS = {
    "left_hip": (-0.9, 0.9),
    "left_knee": (0.0, 1.4),
    "left_ankle": (-0.5, 0.5),
    "right_hip": (-0.9, 0.9),
    "right_knee": (0.0, 1.4),
    "right_ankle": (-0.5, 0.5),
    "left_shoulder_pitch": (-1.8, 1.8),
    "left_elbow": (0.0, 1.8),
    "left_wrist": (-0.9, 0.9),
    "right_shoulder_pitch": (-1.8, 1.8),
    "right_elbow": (0.0, 1.8),
    "right_wrist": (-0.9, 0.9),
}

OBS_DIM = 18 + 2 + 6 + 24 + 1
ACTION_DIM = 14


class NormDummyEnv(gym.Env):
    """Tiny dummy env used only to load VecNormalize statistics."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


class TwoRobotSharedPolicyRally:
    def __init__(
        self,
        model_path: str,
        vecnorm_path: str,
        left_urdf: str = "models/humanoid_left.urdf",
        right_urdf: str = "models/humanoid_right.urdf",
        fixed_base: bool = True,
        control_dt: float = 0.02,
        sim_dt: float = 1.0 / 240.0,
    ):
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.left_urdf = left_urdf
        self.right_urdf = right_urdf
        self.fixed_base = fixed_base
        self.control_dt = control_dt
        self.sim_dt = sim_dt

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=6.0,
            cameraYaw=35,
            cameraPitch=-25,
            cameraTargetPosition=[0.0, 0.0, 1.0],
        )

        self.model = PPO.load(self.model_path)
        self.vecnorm = self._load_vecnormalize(self.vecnorm_path)

        self.left_robot = -1
        self.right_robot = -1
        self.left_joint_map: Dict[str, int] = {}
        self.right_joint_map: Dict[str, int] = {}
        self.left_racket_link = -1
        self.right_racket_link = -1

        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_vel = np.zeros(3, dtype=float)
        self.right_racket_vel = np.zeros(3, dtype=float)

        self.shuttle_id = -1
        self.shuttle_pos = np.zeros(3, dtype=float)
        self.shuttle_vel = np.zeros(3, dtype=float)
        self.shuttle_hist = deque(maxlen=6)
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.012)

        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -100
        self.last_hitter = None

        self._build_world()
        self.reset_rally()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_vecnormalize(self, path: str):
        if not path or not os.path.exists(path):
            print(f"Warning: VecNormalize not found: {path}")
            print("Policy playback will run without saved observation normalization.")
            return None

        dummy_vec = DummyVecEnv([lambda: NormDummyEnv()])
        vecnorm = VecNormalize.load(path, dummy_vec)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print(f"Loaded VecNormalize: {path}")
        return vecnorm

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32).reshape(1, -1)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        return obs

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_world(self):
        for path in [self.left_urdf, self.right_urdf]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Cannot find {path}. Run 2.py once first so it generates humanoid_left/right.urdf."
                )

        p.loadURDF("plane.urdf")
        self._add_court()
        self._add_net()

        self.left_robot = p.loadURDF(
            self.left_urdf,
            [-2.5, 0.0, 1.30],
            [0, 0, 0, 1],
            useFixedBase=self.fixed_base,
        )
        self.right_robot = p.loadURDF(
            self.right_urdf,
            [2.5, 0.0, 1.30],
            [0, 0, 1, 0],
            useFixedBase=self.fixed_base,
        )

        self.left_joint_map = self._make_joint_map(self.left_robot)
        self.right_joint_map = self._make_joint_map(self.right_robot)

        self.left_racket_link = self._find_first_link(self.left_robot, ["left_racket", "racket", "left_hand"])
        self.right_racket_link = self._find_first_link(self.right_robot, ["right_racket", "racket", "right_hand"])

        self.shuttle_id = self._create_shuttle()
        self._set_neutral_pose(self.left_robot, self.left_joint_map)
        self._set_neutral_pose(self.right_robot, self.right_joint_map)

        print("Left racket link:", self.left_racket_link)
        print("Right racket link:", self.right_racket_link)

    def _add_court(self):
        scale = 0.60
        line_w = 0.035
        line_h = 0.004
        z = 0.006
        white = [1.0, 1.0, 1.0, 1.0]
        green = [0.08, 0.42, 0.18, 1.0]
        service_green = [0.10, 0.50, 0.22, 1.0]

        half_len = 6.70 * scale
        half_doubles = 3.05 * scale
        half_singles = 2.59 * scale
        short_service = 1.98 * scale
        long_service_doubles = 5.94 * scale

        def add_box(center, half_extents, rgba):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=center)

        def add_line_x(x1, x2, y):
            add_box([(x1 + x2) / 2, y, z + 0.002], [abs(x2 - x1) / 2, line_w / 2, line_h], white)

        def add_line_y(x, y1, y2):
            add_box([x, (y1 + y2) / 2, z + 0.002], [line_w / 2, abs(y2 - y1) / 2, line_h], white)

        add_box([0, 0, z / 2], [half_len + 0.15, half_doubles + 0.15, z / 2], green)
        add_box([-(short_service + half_len) / 2, 0, z + 0.001], [(half_len - short_service) / 2, half_doubles, 0.001], service_green)
        add_box([(short_service + half_len) / 2, 0, z + 0.001], [(half_len - short_service) / 2, half_doubles, 0.001], service_green)

        add_line_x(-half_len, half_len, -half_doubles)
        add_line_x(-half_len, half_len, half_doubles)
        add_line_y(-half_len, -half_doubles, half_doubles)
        add_line_y(half_len, -half_doubles, half_doubles)
        add_line_x(-half_len, half_len, -half_singles)
        add_line_x(-half_len, half_len, half_singles)
        add_line_y(-short_service, -half_doubles, half_doubles)
        add_line_y(short_service, -half_doubles, half_doubles)
        add_line_y(-long_service_doubles, -half_doubles, half_doubles)
        add_line_y(long_service_doubles, -half_doubles, half_doubles)
        add_line_x(-half_len, -short_service, 0.0)
        add_line_x(short_service, half_len, 0.0)

    def _add_net(self):
        net_height = 1.55
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, net_height / 2],
            rgbaColor=[0.2, 0.8, 0.2, 0.45],
        )
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, net_height / 2])
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, net_height / 2],
        )

    def _create_shuttle(self):
        feather_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09)
        feather_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.038,
            length=0.09,
            rgbaColor=[0.95, 0.95, 1.0, 1.0],
        )
        return p.createMultiBody(0.005, feather_col, feather_vis, [0, 0, 1.4])

    def _make_joint_map(self, robot_id: int) -> Dict[str, int]:
        mapping = {}
        for i in range(p.getNumJoints(robot_id)):
            name = p.getJointInfo(robot_id, i)[1].decode("utf-8")
            mapping[name] = i
        missing = [name for name in JOINT_NAMES if name not in mapping]
        if missing:
            raise RuntimeError(f"URDF missing required joints: {missing}")
        return mapping

    def _get_link_index(self, robot_id: int, link_name: str) -> int:
        for i in range(p.getNumJoints(robot_id)):
            link = p.getJointInfo(robot_id, i)[12].decode("utf-8")
            if link == link_name:
                return i
        return -1

    def _find_first_link(self, robot_id: int, names: List[str]) -> int:
        for name in names:
            idx = self._get_link_index(robot_id, name)
            if idx >= 0:
                return idx
        raise RuntimeError(f"Could not find any link from {names}")

    # ------------------------------------------------------------------
    # Reset and neutral pose
    # ------------------------------------------------------------------

    def reset_rally(self):
        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -100
        self.last_hitter = None
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_vel[:] = 0
        self.right_racket_vel[:] = 0

        p.resetBasePositionAndOrientation(self.left_robot, [-2.5, 0.0, 1.30], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.right_robot, [2.5, 0.0, 1.30], [0, 0, 1, 0])
        self._set_neutral_pose(self.left_robot, self.left_joint_map)
        self._set_neutral_pose(self.right_robot, self.right_joint_map)

        self._serve_random()
        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())

    def _set_neutral_pose(self, robot_id: int, joint_map: Dict[str, int]):
        neutral = {
            "left_hip": 0.0,
            "left_knee": 0.15,
            "left_ankle": -0.05,
            "right_hip": 0.0,
            "right_knee": 0.15,
            "right_ankle": -0.05,
            "left_shoulder_pitch": 0.25,
            "left_elbow": 0.35,
            "left_wrist": 0.0,
            "right_shoulder_pitch": 0.25,
            "right_elbow": 0.35,
            "right_wrist": 0.0,
        }
        for name, target in neutral.items():
            jid = joint_map[name]
            p.resetJointState(robot_id, jid, target, 0.0)
            p.setJointMotorControl2(
                robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=160,
                maxVelocity=5,
            )

    def _serve_random(self):
        # Serve from one side toward the opposite robot.
        side = np.random.choice([-1, 1])  # -1: from left to right, +1: from right to left
        start_x = 3.2 if side > 0 else -3.2
        vx = np.random.uniform(-5.0, -4.0) if side > 0 else np.random.uniform(4.0, 5.0)
        self.shuttle_pos = np.array([start_x, np.random.uniform(-0.8, 0.8), np.random.uniform(1.35, 1.65)], dtype=float)
        self.shuttle_vel = np.array([vx, np.random.uniform(-0.5, 0.5), np.random.uniform(2.5, 3.8)], dtype=float)
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])
        print(f"SERVE: pos={self.shuttle_pos.round(2)}, vel={self.shuttle_vel.round(2)}")

    # ------------------------------------------------------------------
    # Observation and policy inference
    # ------------------------------------------------------------------

    def _mirror_vec_for_right_policy(self, vec: np.ndarray) -> np.ndarray:
        out = np.array(vec, dtype=float).copy()
        out[0] *= -1.0
        return out

    def _get_racket_state(self, robot_id: int, link_idx: int, prev_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(robot_id, link_idx, computeLinkVelocity=1)
        pos = np.array(state[0], dtype=float)
        if prev_pos is None:
            vel = np.zeros(3, dtype=float)
        else:
            vel = (pos - prev_pos) / self.control_dt
        return pos, vel

    def _normalized_joint_state(self, robot_id: int, joint_map: Dict[str, int], side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return joint state in the policy's expected left-side convention.

        For right robot, map actual right arm into policy's left-arm slots, so the same
        policy can use its learned hitting arm behavior on the opposite side.
        """
        if side == "left":
            order = JOINT_NAMES
        else:
            order = [
                "left_hip", "left_knee", "left_ankle",
                "right_hip", "right_knee", "right_ankle",
                "right_shoulder_pitch", "right_elbow", "right_wrist",
                "left_shoulder_pitch", "left_elbow", "left_wrist",
            ]

        q_list = []
        qd_list = []
        for actual_name, policy_name in zip(order, JOINT_NAMES):
            jid = joint_map[actual_name]
            js = p.getJointState(robot_id, jid)
            low, high = JOINT_LIMITS[policy_name]
            q_norm = 2.0 * (js[0] - low) / (high - low) - 1.0
            q_list.append(q_norm)
            qd_list.append(js[1] * 0.1)
        return np.array(q_list, dtype=float), np.array(qd_list, dtype=float)

    def _make_policy_obs(self, side: str) -> np.ndarray:
        if side == "left":
            robot_id = self.left_robot
            joint_map = self.left_joint_map
            racket_pos, racket_vel = self._get_racket_state(robot_id, self.left_racket_link, self.prev_left_racket)
            base_pos, _ = p.getBasePositionAndOrientation(robot_id)
            base_xy = np.array(base_pos[:2], dtype=float)
            shuttle_hist = np.array(list(self.shuttle_hist), dtype=float)
        else:
            robot_id = self.right_robot
            joint_map = self.right_joint_map
            racket_pos, racket_vel = self._get_racket_state(robot_id, self.right_racket_link, self.prev_right_racket)
            base_pos, _ = p.getBasePositionAndOrientation(robot_id)
            base_xy = np.array(base_pos[:2], dtype=float)
            shuttle_hist = np.array(list(self.shuttle_hist), dtype=float)

            # Mirror world into left-side policy convention.
            shuttle_hist[:, 0] *= -1.0
            base_xy[0] *= -1.0
            racket_pos = self._mirror_vec_for_right_policy(racket_pos)
            racket_vel = self._mirror_vec_for_right_policy(racket_vel)

        q, qd = self._normalized_joint_state(robot_id, joint_map, side)
        time_frac = np.array([0.0], dtype=float)

        obs = np.concatenate(
            [
                shuttle_hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
                base_xy * np.array([0.25, 0.5]),
                racket_pos * np.array([0.25, 0.5, 0.5]),
                racket_vel * 0.1,
                q,
                qd,
                time_frac,
            ]
        ).astype(np.float32)
        return obs

    def _policy_action(self, side: str) -> np.ndarray:
        obs = self._make_policy_obs(side)
        norm_obs = self._normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return np.asarray(action).reshape(-1).astype(float)

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _action_to_joint_targets(self, action: np.ndarray) -> Dict[str, float]:
        targets = {}
        for value, name in zip(action[2:], JOINT_NAMES):
            low, high = JOINT_LIMITS[name]
            targets[name] = low + (float(value) + 1.0) * 0.5 * (high - low)
        return targets

    def _apply_action_left(self, action: np.ndarray):
        self._apply_base_action(self.left_robot, action[:2], side="left")
        targets = self._action_to_joint_targets(action)
        self._apply_joint_targets(self.left_robot, self.left_joint_map, targets)

    def _apply_action_right(self, action: np.ndarray):
        # Invert x base command because policy obs was mirrored.
        base_action = np.array([-action[0], action[1]], dtype=float)
        self._apply_base_action(self.right_robot, base_action, side="right")

        policy_targets = self._action_to_joint_targets(action)

        # Map policy's learned left racket arm to actual right racket arm.
        targets = {
            "left_hip": policy_targets["left_hip"],
            "left_knee": policy_targets["left_knee"],
            "left_ankle": policy_targets["left_ankle"],
            "right_hip": policy_targets["right_hip"],
            "right_knee": policy_targets["right_knee"],
            "right_ankle": policy_targets["right_ankle"],
            "right_shoulder_pitch": policy_targets["left_shoulder_pitch"],
            "right_elbow": policy_targets["left_elbow"],
            "right_wrist": policy_targets["left_wrist"],
            "left_shoulder_pitch": policy_targets["right_shoulder_pitch"],
            "left_elbow": policy_targets["right_elbow"],
            "left_wrist": policy_targets["right_wrist"],
        }
        self._apply_joint_targets(self.right_robot, self.right_joint_map, targets)

    def _apply_base_action(self, robot_id: int, base_action: np.ndarray, side: str):
        vx = float(np.clip(base_action[0], -1, 1)) * 1.7
        vy = float(np.clip(base_action[1], -1, 1)) * 1.4
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        pos = np.array(pos, dtype=float)
        pos[0] += vx * self.control_dt
        pos[1] += vy * self.control_dt

        if side == "left":
            pos[0] = np.clip(pos[0], -3.8, -0.45)
            orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], 0.45, 3.8)
            orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -1.85, 1.85)
        p.resetBasePositionAndOrientation(robot_id, pos, orn)

    def _apply_joint_targets(self, robot_id: int, joint_map: Dict[str, int], targets: Dict[str, float]):
        for name, target in targets.items():
            if name not in joint_map:
                continue
            p.setJointMotorControl2(
                robot_id,
                joint_map[name],
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=180,
                maxVelocity=6.0,
            )

    # ------------------------------------------------------------------
    # Shuttle contact and rally logic
    # ------------------------------------------------------------------

    def _maybe_hit(self, side: str) -> bool:
        if (self.step_count - self.last_hit_step) < 25:
            return False

        if side == "left":
            racket_pos, racket_vel = self._get_racket_state(self.left_robot, self.left_racket_link, self.prev_left_racket)
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
            forward_speed = racket_vel[0]
        else:
            racket_pos, racket_vel = self._get_racket_state(self.right_robot, self.right_racket_link, self.prev_right_racket)
            normal = np.array([-1.0, 0.0, 0.0], dtype=float)
            forward_speed = -racket_vel[0]

        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        good_height = 0.35 < self.shuttle_pos[2] < 1.95
        swinging_forward = forward_speed > 0.20

        if dist > 0.75 or not good_height or not swinging_forward:
            return False

        vin = self.shuttle_vel.copy()
        v_racket_n = np.dot(racket_vel, normal) * normal
        v_shuttle_n = np.dot(vin, normal) * normal
        vout = vin - 2.0 * v_shuttle_n + 2.0 * v_racket_n

        # Make the demo robust: enforce the outgoing direction and net clearance.
        if side == "left":
            vout[0] = max(vout[0], 4.8)
        else:
            vout[0] = min(vout[0], -4.8)
        vout[2] = max(vout[2], 4.8)

        speed = np.linalg.norm(vout)
        if speed > 14.0:
            vout = vout / speed * 14.0

        self.shuttle_vel = vout
        self.last_hit_step = self.step_count
        self.last_hitter = side
        self.hit_count += 1
        print(f"{side.upper()} HIT #{self.hit_count}: dist={dist:.2f}, racket_vel={racket_vel.round(2)}, vout={vout.round(2)}")
        return True

    def _update_shuttle(self):
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(
            self.shuttle_pos,
            self.shuttle_vel,
            self.control_dt,
            force=None,
        )
        self.shuttle_hist.append(self.shuttle_pos.copy())
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])

    def _is_rally_done(self) -> bool:
        if self.shuttle_pos[2] < 0.06:
            print("RALLY END: shuttle landed. hits=", self.hit_count)
            return True
        if abs(self.shuttle_pos[0]) > 4.4 or abs(self.shuttle_pos[1]) > 2.5:
            print("RALLY END: shuttle out. hits=", self.hit_count)
            return True
        # If it crosses the net too low, treat as failed.
        if abs(self.shuttle_pos[0]) < 0.06 and self.shuttle_pos[2] < 1.55:
            print("RALLY END: net fail. hits=", self.hit_count)
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        try:
            while True:
                active_side = "left" if self.shuttle_pos[0] < 0 else "right"

                # Both robots run the same policy every step.
                left_action = self._policy_action("left")
                right_action = self._policy_action("right")
                self._apply_action_left(left_action)
                self._apply_action_right(right_action)

                # Step PyBullet motors.
                sim_steps = max(1, int(round(self.control_dt / self.sim_dt)))
                for _ in range(sim_steps):
                    p.stepSimulation()

                # Update racket velocities after motor movement.
                left_racket, self.left_racket_vel = self._get_racket_state(self.left_robot, self.left_racket_link, self.prev_left_racket)
                right_racket, self.right_racket_vel = self._get_racket_state(self.right_robot, self.right_racket_link, self.prev_right_racket)
                self.prev_left_racket = left_racket.copy()
                self.prev_right_racket = right_racket.copy()

                # Only the robot on the current shuttle side can hit.
                self._maybe_hit(active_side)
                self._update_shuttle()

                self.step_count += 1
                if self._is_rally_done():
                    time.sleep(0.4)
                    self.reset_rally()

                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("Exit")
        finally:
            p.disconnect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/badminton_final.zip")
    parser.add_argument("--vecnorm", type=str, default="runs/vecnormalize_final.pkl")
    parser.add_argument("--left_urdf", type=str, default="models/humanoid_left.urdf")
    parser.add_argument("--right_urdf", type=str, default="models/humanoid_right.urdf")
    parser.add_argument("--free_base", action="store_true", help="Use free floating bases. Not recommended for this simplified robot.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = TwoRobotSharedPolicyRally(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        left_urdf=args.left_urdf,
        right_urdf=args.right_urdf,
        fixed_base=not args.free_base,
    )
    demo.run()
