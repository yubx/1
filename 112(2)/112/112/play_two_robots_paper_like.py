#!/usr/bin/env python3
"""
Paper-like two-robot shared-policy badminton rally demo for the simplified PyBullet model.

Save as:
    play_two_robots_paper_like.py

Recommended command:
    python play_two_robots_paper_like.py --model runs_rally_easy\badminton_rally_final.zip --vecnorm runs_rally_easy\vecnormalize_rally_final.pkl

If the rally is still difficult, use a more permissive contact region:
    python play_two_robots_paper_like.py --model runs_rally_easy\badminton_rally_final.zip --vecnorm runs_rally_easy\vecnormalize_rally_final.pkl --hit_radius 1.35

Purpose:
- Use TWO robots in one PyBullet scene.
- Use ONE shared PPO policy for both sides.
- Mirror the right robot into the policy's canonical left-side frame.
- Use an online shuttle interception target and time-to-hit.
- Use prediction-based footwork, because the trained policy's high-level action is mainly useful for swing phase and power.
- Use a racket-triggered assisted return model to maintain rally stability.

This is intended for the high-level policy trained by:
    train_rl_badminton_easy.py
or:
    train_rl_badminton_rally_easy.py

The uploaded training code uses a 4D high-level action:
    action[0] = base x movement
    action[1] = base y movement
    action[2] = swing phase
    action[3] = swing power

This rollout script keeps that design but makes two-robot rally more stable.
"""

import argparse
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from physics_shuttle import ShuttlePhysics


# Must match the high-level training observation and action dimensions.
OBS_DIM = 18 + 2 + 6 + 3 + 1 + 1
ACTION_DIM = 4

JOINT_NAMES = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_shoulder_pitch", "left_elbow", "left_wrist",
    "right_shoulder_pitch", "right_elbow", "right_wrist",
]


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


class PaperLikeTwoRobotRally:
    def __init__(
        self,
        model_path: str,
        vecnorm_path: Optional[str],
        left_urdf: str = "models/humanoid_left.urdf",
        right_urdf: str = "models/humanoid_right.urdf",
        hit_radius: float = 1.20,
        fixed_base: bool = True,
        seed: int = 0,
        use_policy_footwork: bool = False,
    ):
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.left_urdf = left_urdf
        self.right_urdf = right_urdf
        self.hit_radius = float(hit_radius)
        self.fixed_base = bool(fixed_base)
        self.use_policy_footwork = bool(use_policy_footwork)
        self.rng = np.random.default_rng(seed)

        self.control_dt = 0.02
        self.sim_dt = 1.0 / 240.0
        self.net_height = 1.55
        self.max_x = 3.85
        self.max_y = 1.80

        self.client = p.connect(p.GUI)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet GUI.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(6.4, 35, -25, [0.0, 0.0, 1.0])

        self.model = PPO.load(self.model_path)
        self.vecnorm = self._load_vecnormalize(self.vecnorm_path)

        self.left_robot = -1
        self.right_robot = -1
        self.left_map: Dict[str, int] = {}
        self.right_map: Dict[str, int] = {}
        self.left_racket_link = -1
        self.right_racket_link = -1

        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_vel = np.zeros(3, dtype=float)
        self.right_racket_vel = np.zeros(3, dtype=float)
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0

        self.shuttle_id = -1
        self.shuttle_pos = np.zeros(3, dtype=float)
        self.shuttle_vel = np.zeros(3, dtype=float)
        self.shuttle_hist = deque(maxlen=6)
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.010)

        self.step_count = 0
        self.hit_count = 0
        self.rally_count = 0
        self.last_hit_step = -100
        self.last_hitter = None

        self._build_world()
        self.reset_rally()

    # ------------------------------------------------------------------
    # Loading and normalization
    # ------------------------------------------------------------------

    def _load_vecnormalize(self, path: Optional[str]):
        if path is None:
            path = os.path.join(os.path.dirname(self.model_path), "vecnormalize_rally_final.pkl")
            if not os.path.exists(path):
                path = os.path.join(os.path.dirname(self.model_path), "vecnormalize_easy_final.pkl")

        if not path or not os.path.exists(path):
            print("Warning: VecNormalize file not found. Policy playback may be worse.")
            return None

        dummy = DummyVecEnv([lambda: NormDummyEnv()])
        vecnorm = VecNormalize.load(path, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print("Loaded VecNormalize:", path)
        return vecnorm

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
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
                    f"Missing {path}. Run 2.py first so it generates models/humanoid_left.urdf and models/humanoid_right.urdf."
                )

        p.loadURDF("plane.urdf")
        self._add_court()
        self._add_net()

        self.left_robot = p.loadURDF(
            self.left_urdf,
            [-2.45, 0.0, 1.30],
            [0, 0, 0, 1],
            useFixedBase=self.fixed_base,
        )
        self.right_robot = p.loadURDF(
            self.right_urdf,
            [2.45, 0.0, 1.30],
            [0, 0, 1, 0],
            useFixedBase=self.fixed_base,
        )

        self.left_map = self._make_joint_map(self.left_robot)
        self.right_map = self._make_joint_map(self.right_robot)
        self.left_racket_link = self._find_link(self.left_robot, ["left_racket", "racket", "left_hand"])
        self.right_racket_link = self._find_link(self.right_robot, ["right_racket", "racket", "right_hand"])

        self.shuttle_id = self._create_shuttle()
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

        print("Left racket link:", self.left_racket_link)
        print("Right racket link:", self.right_racket_link)

    def _add_court(self):
        green = [0.08, 0.42, 0.18, 1.0]
        green2 = [0.10, 0.50, 0.22, 1.0]
        white = [1.0, 1.0, 1.0, 1.0]
        z = 0.006

        def add_box(center, half_extents, rgba):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=center)

        add_box([0, 0, z / 2], [4.15, 1.95, z / 2], green)
        add_box([-2.55, 0, z + 0.001], [1.35, 1.85, 0.001], green2)
        add_box([2.55, 0, z + 0.001], [1.35, 1.85, 0.001], green2)

        def line(pos, size):
            add_box(pos, size, white)

        # Outer boundary.
        line([0, -1.85, 0.016], [4.0, 0.018, 0.006])
        line([0, 1.85, 0.016], [4.0, 0.018, 0.006])
        line([-4.0, 0, 0.016], [0.018, 1.85, 0.006])
        line([4.0, 0, 0.016], [0.018, 1.85, 0.006])

        # Net line and service lines.
        line([0, 0, 0.018], [0.018, 1.85, 0.006])
        line([-1.20, 0, 0.018], [0.018, 1.85, 0.006])
        line([1.20, 0, 0.018], [0.018, 1.85, 0.006])
        line([-2.70, 0, 0.018], [0.018, 1.85, 0.006])
        line([2.70, 0, 0.018], [0.018, 1.85, 0.006])

        # Center service line segments.
        line([-2.60, 0, 0.018], [1.40, 0.018, 0.006])
        line([2.60, 0, 0.018], [1.40, 0.018, 0.006])

    def _add_net(self):
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, self.net_height / 2],
            rgbaColor=[0.2, 0.8, 0.2, 0.45],
        )
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, self.net_height / 2])
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, self.net_height / 2],
        )

        # Net posts.
        post_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=1.70, rgbaColor=[0.1, 0.1, 0.1, 1])
        p.createMultiBody(0, baseVisualShapeIndex=post_vis, basePosition=[0, -2.05, 0.85])
        p.createMultiBody(0, baseVisualShapeIndex=post_vis, basePosition=[0, 2.05, 0.85])

    def _create_shuttle(self):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.038,
            length=0.09,
            rgbaColor=[0.95, 0.95, 1.0, 1.0],
        )
        return p.createMultiBody(0.005, col, vis, [0, 0, 1.4])

    def _make_joint_map(self, robot_id: int) -> Dict[str, int]:
        mapping = {}
        for i in range(p.getNumJoints(robot_id)):
            name = p.getJointInfo(robot_id, i)[1].decode("utf-8")
            mapping[name] = i
        missing = [name for name in JOINT_NAMES if name not in mapping]
        if missing:
            raise RuntimeError(f"URDF missing required joints: {missing}")
        return mapping

    def _find_link(self, robot_id: int, candidates: List[str]) -> int:
        for name in candidates:
            for i in range(p.getNumJoints(robot_id)):
                link = p.getJointInfo(robot_id, i)[12].decode("utf-8")
                if link == name:
                    return i
        raise RuntimeError(f"Cannot find link from {candidates}")

    # ------------------------------------------------------------------
    # Reset and serve
    # ------------------------------------------------------------------

    def reset_rally(self):
        self.rally_count += 1
        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -100
        self.last_hitter = None
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_vel[:] = 0
        self.right_racket_vel[:] = 0
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0

        p.resetBasePositionAndOrientation(self.left_robot, [-2.45, 0.0, 1.30], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.right_robot, [2.45, 0.0, 1.30], [0, 0, 1, 0])
        p.resetBaseVelocity(self.left_robot, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.right_robot, [0, 0, 0], [0, 0, 0])
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

        receiver = "left" if self.rng.random() < 0.5 else "right"
        self._serve_to(receiver)

        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())
        print(f"NEW RALLY {self.rally_count}, receiver={receiver}")

    def _serve_to(self, receiver: str):
        if receiver == "left":
            start = np.array([3.20, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.75)], dtype=float)
            target = np.array([-1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)], dtype=float)
        else:
            start = np.array([-3.20, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.75)], dtype=float)
            target = np.array([1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)], dtype=float)

        t_flight = self.rng.uniform(0.95, 1.15)
        self.shuttle_pos = start
        self.shuttle_vel = self._ballistic_velocity(start, target, t_flight)
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])
        print("SERVE:", start.round(2), "->", target.round(2), "vel", self.shuttle_vel.round(2))

    def _ballistic_velocity(self, start: np.ndarray, target: np.ndarray, t_flight: float) -> np.ndarray:
        g = -9.8
        return np.array(
            [
                (target[0] - start[0]) / t_flight,
                (target[1] - start[1]) / t_flight,
                (target[2] - start[2] - 0.5 * g * t_flight * t_flight) / t_flight,
            ],
            dtype=float,
        )

    def _reset_neutral(self, robot_id: int, joint_map: Dict[str, int]):
        neutral = {
            "left_hip": 0.0,
            "left_knee": 0.20,
            "left_ankle": -0.06,
            "right_hip": 0.0,
            "right_knee": 0.20,
            "right_ankle": -0.06,
            "left_shoulder_pitch": 0.25,
            "left_elbow": 0.35,
            "left_wrist": 0.0,
            "right_shoulder_pitch": 0.25,
            "right_elbow": 0.35,
            "right_wrist": 0.0,
        }
        for name, value in neutral.items():
            jid = joint_map[name]
            p.resetJointState(robot_id, jid, value, 0.0)
            p.setJointMotorControl2(
                robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=value,
                force=180,
                maxVelocity=6.0,
            )

    # ------------------------------------------------------------------
    # Prediction and observation
    # ------------------------------------------------------------------

    def _receiver_side(self) -> str:
        return "left" if self.shuttle_pos[0] < 0 else "right"

    def _other_side(self, side: str) -> str:
        return "right" if side == "left" else "left"

    def _estimate_intercept(self, side: str) -> Tuple[np.ndarray, float]:
        """Estimate a target-known interception point and time-to-hit.

        This is the key paper-like part: the policy is given a reachable interception
        target and estimated contact time. The training code also uses target and time.
        """
        target_x = -1.65 if side == "left" else 1.65
        vx = float(self.shuttle_vel[0])

        if abs(vx) < 0.2:
            t_hit = 0.75
        else:
            t_hit = (target_x - float(self.shuttle_pos[0])) / vx

        # If the mathematical intersection is behind or too late, clamp to a useful window.
        t_hit = float(np.clip(t_hit, 0.08, 1.35))
        g = -9.8
        target = self.shuttle_pos + self.shuttle_vel * t_hit + np.array([0.0, 0.0, 0.5 * g * t_hit * t_hit])
        target[0] = target_x
        target[1] = float(np.clip(target[1], -0.75, 0.75))
        target[2] = float(np.clip(target[2], 1.20, 1.70))
        return target, t_hit

    def _racket_state(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        if side == "left":
            robot, link, prev = self.left_robot, self.left_racket_link, self.prev_left_racket
        else:
            robot, link, prev = self.right_robot, self.right_racket_link, self.prev_right_racket

        state = p.getLinkState(robot, link, computeLinkVelocity=1)
        pos = np.array(state[0], dtype=float)
        vel = np.zeros(3, dtype=float) if prev is None else (pos - prev) / self.control_dt
        return pos, vel

    def _make_policy_obs(self, side: str) -> np.ndarray:
        if side == "left":
            robot = self.left_robot
            base_pos, _ = p.getBasePositionAndOrientation(robot)
            base_xy = np.array(base_pos[:2], dtype=float)
            racket_pos, racket_vel = self._racket_state("left")
            hist = np.array(list(self.shuttle_hist), dtype=float)
            target, t_hit = self._estimate_intercept("left")
            phase = self.last_phase_left
        else:
            robot = self.right_robot
            base_pos, _ = p.getBasePositionAndOrientation(robot)
            base_xy = np.array(base_pos[:2], dtype=float)
            racket_pos, racket_vel = self._racket_state("right")
            hist = np.array(list(self.shuttle_hist), dtype=float)
            target, t_hit = self._estimate_intercept("right")

            # Mirror the right-side real world into the canonical left-side policy frame.
            hist = hist.copy()
            hist[:, 0] *= -1.0
            base_xy = base_xy.copy()
            base_xy[0] *= -1.0
            racket_pos = racket_pos.copy()
            racket_vel = racket_vel.copy()
            target = target.copy()
            racket_pos[0] *= -1.0
            racket_vel[0] *= -1.0
            target[0] *= -1.0
            phase = self.last_phase_right

        obs = np.concatenate(
            [
                hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
                base_xy * np.array([0.25, 0.5]),
                racket_pos * np.array([0.25, 0.5, 0.5]),
                racket_vel * 0.1,
                target * np.array([0.25, 0.5, 0.5]),
                np.array([t_hit, phase], dtype=float),
            ]
        ).astype(np.float32)
        return obs

    def _policy_action(self, side: str) -> np.ndarray:
        obs = self._make_policy_obs(side)
        norm_obs = self._normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        return action

    # ------------------------------------------------------------------
    # Footwork and action application
    # ------------------------------------------------------------------

    def _make_chase_action(self, side: str, policy_action: np.ndarray) -> np.ndarray:
        """Use prediction-based footwork and policy-based swing.

        The policy was trained as a high-level policy, but in two-robot rally the
        incoming ball is more variable. Prediction-based footwork makes the two-side
        rollout much more robust and closer to target-known control.
        """
        action = policy_action.copy()
        target, t_hit = self._estimate_intercept(side)

        if side == "left":
            robot_id = self.left_robot
            desired_x = target[0] - 0.55
        else:
            robot_id = self.right_robot
            desired_x = target[0] + 0.55

        desired_y = target[1]
        base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        base_pos = np.array(base_pos, dtype=float)
        dx = desired_x - base_pos[0]
        dy = desired_y - base_pos[1]

        # Convert desired velocity into normalized command.
        # Faster than the training environment because rally trajectories are harder.
        t_safe = max(t_hit, 0.14)
        vx_cmd = np.clip(dx / t_safe / 4.5, -1.0, 1.0)
        vy_cmd = np.clip(dy / t_safe / 3.6, -1.0, 1.0)

        action[0] = vx_cmd
        action[1] = vy_cmd
        return action

    def _apply_active_policy(self, side: str, raw_action: np.ndarray):
        action = raw_action.copy()
        if not self.use_policy_footwork:
            action = self._make_chase_action(side, action)

        if side == "left":
            self._apply_action_to_robot(self.left_robot, self.left_map, action, "left", racket_side="left")
            self.last_phase_left = float((action[2] + 1.0) * 0.5)
        else:
            # action[0] was produced in mirrored canonical coordinates; map it back.
            action = action.copy()
            action[0] *= -1.0
            self._apply_action_to_robot(self.right_robot, self.right_map, action, "right", racket_side="right")
            self.last_phase_right = float((raw_action[2] + 1.0) * 0.5)

    def _apply_passive_ready(self, side: str):
        if side == "left":
            self._move_base_toward(self.left_robot, [-2.45, 0.0], "left", max_step=0.060)
            targets = self._swing_primitive(phase=0.10, power=0.20, racket_side="left")
            self._apply_leg_stance(targets, phase=0.10)
            self._apply_joint_targets(self.left_robot, self.left_map, targets)
        else:
            self._move_base_toward(self.right_robot, [2.45, 0.0], "right", max_step=0.060)
            targets = self._swing_primitive(phase=0.10, power=0.20, racket_side="right")
            self._apply_leg_stance(targets, phase=0.10)
            self._apply_joint_targets(self.right_robot, self.right_map, targets)

    def _move_base_toward(self, robot_id: int, target_xy: List[float], side: str, max_step: float = 0.060):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        pos = np.array(pos, dtype=float)
        target = np.array(target_xy, dtype=float)
        delta = np.clip(target - pos[:2], -max_step, max_step)
        pos[:2] += delta

        if side == "left":
            pos[0] = np.clip(pos[0], -self.max_x, -0.50)
            orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], 0.50, self.max_x)
            orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -self.max_y, self.max_y)
        p.resetBasePositionAndOrientation(robot_id, pos, orn)

    def _apply_action_to_robot(
        self,
        robot_id: int,
        joint_map: Dict[str, int],
        action: np.ndarray,
        side: str,
        racket_side: str,
    ):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        pos = np.array(pos, dtype=float)

        # Faster footwork for rally. This is the main fix for "the other robot moves too slowly".
        pos[0] += float(action[0]) * 4.2 * self.control_dt
        pos[1] += float(action[1]) * 3.4 * self.control_dt

        if side == "left":
            pos[0] = np.clip(pos[0], -self.max_x, -0.50)
            orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], 0.50, self.max_x)
            orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -self.max_y, self.max_y)
        p.resetBasePositionAndOrientation(robot_id, pos, orn)

        phase = float((action[2] + 1.0) * 0.5)
        power = float((action[3] + 1.0) * 0.5)
        targets = self._swing_primitive(phase, power, racket_side=racket_side)
        self._apply_leg_stance(targets, phase)
        self._apply_joint_targets(robot_id, joint_map, targets)

    def _apply_leg_stance(self, targets: Dict[str, float], phase: float):
        targets.update(
            {
                "left_hip": 0.05 * np.sin(phase * 2 * np.pi),
                "right_hip": -0.05 * np.sin(phase * 2 * np.pi),
                "left_knee": 0.22,
                "right_knee": 0.22,
                "left_ankle": -0.08,
                "right_ankle": -0.08,
            }
        )

    def _apply_joint_targets(self, robot_id: int, joint_map: Dict[str, int], targets: Dict[str, float]):
        for name, value in targets.items():
            if name not in joint_map:
                continue
            p.setJointMotorControl2(
                robot_id,
                joint_map[name],
                p.POSITION_CONTROL,
                targetPosition=float(value),
                force=280,
                maxVelocity=11.0,
            )

    def _swing_primitive(self, phase: float, power: float, racket_side: str) -> Dict[str, float]:
        amp = 0.80 + 0.75 * power
        if phase < 0.35:
            t = phase / 0.35
            shoulder = 0.20 + 1.30 * amp * t
            elbow = 0.30 + 0.80 * amp * t
            wrist = 0.00 + 0.50 * amp * t
        elif phase < 0.65:
            t = (phase - 0.35) / 0.30
            shoulder = 1.50 * amp - 2.30 * amp * t
            elbow = 1.10 * amp - 0.85 * amp * t
            wrist = 0.55 * amp - 1.05 * amp * t
        else:
            t = (phase - 0.65) / 0.35
            shoulder = -0.75 * amp + 1.00 * amp * t
            elbow = 0.25 + 0.15 * t
            wrist = -0.50 * amp + 0.50 * amp * t

        if racket_side == "left":
            return {
                "left_shoulder_pitch": float(np.clip(shoulder, -1.8, 1.8)),
                "left_elbow": float(np.clip(elbow, 0.0, 1.8)),
                "left_wrist": float(np.clip(wrist, -0.9, 0.9)),
                "right_shoulder_pitch": float(np.clip(-0.35 * shoulder, -1.8, 1.8)),
                "right_elbow": 0.35,
                "right_wrist": float(np.clip(-0.3 * wrist, -0.9, 0.9)),
            }

        # Right robot is rotated 180 deg. Use mirrored shoulder and wrist signs.
        return {
            "right_shoulder_pitch": float(np.clip(-shoulder, -1.8, 1.8)),
            "right_elbow": float(np.clip(elbow, 0.0, 1.8)),
            "right_wrist": float(np.clip(-wrist, -0.9, 0.9)),
            "left_shoulder_pitch": float(np.clip(0.35 * shoulder, -1.8, 1.8)),
            "left_elbow": 0.35,
            "left_wrist": float(np.clip(0.3 * wrist, -0.9, 0.9)),
        }

    # ------------------------------------------------------------------
    # Shuttle contact and rally logic
    # ------------------------------------------------------------------

    def _maybe_hit(self, side: str):
        if self.step_count - self.last_hit_step < 22:
            return

        racket_pos, racket_vel = self._racket_state(side)
        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        if side == "left":
            forward = float(racket_vel[0])
            valid_half = self.shuttle_pos[0] < 0.15
            outgoing_target = np.array(
                [1.65, self.rng.uniform(-0.60, 0.60), self.rng.uniform(1.35, 1.65)],
                dtype=float,
            )
        else:
            forward = float(-racket_vel[0])
            valid_half = self.shuttle_pos[0] > -0.15
            outgoing_target = np.array(
                [-1.65, self.rng.uniform(-0.60, 0.60), self.rng.uniform(1.35, 1.65)],
                dtype=float,
            )

        good_height = 0.30 < self.shuttle_pos[2] < 2.05
        if not (valid_half and good_height and dist < self.hit_radius and forward > 0.04):
            return

        # Racket-triggered assisted trajectory.
        # The hit only triggers when the racket is near the shuttle; then the return is stabilized
        # to a reachable opponent-side interception point to mimic controlled rally training.
        t_flight = self.rng.uniform(0.90, 1.12)
        vout = self._ballistic_velocity(self.shuttle_pos, outgoing_target, t_flight)
        vout[1] += 0.12 * racket_vel[1]
        vout[2] = max(vout[2], 4.70)

        speed = np.linalg.norm(vout)
        if speed > 13.5:
            vout = vout / speed * 13.5

        self.shuttle_vel = vout
        self.hit_count += 1
        self.last_hit_step = self.step_count
        self.last_hitter = side
        print(
            f"{side.upper()} HIT #{self.hit_count}: "
            f"dist={dist:.2f}, forward={forward:.2f}, target={outgoing_target.round(2)}, vout={vout.round(2)}"
        )

    def _update_racket_memory(self):
        left_pos, self.left_racket_vel = self._racket_state("left")
        right_pos, self.right_racket_vel = self._racket_state("right")
        self.prev_left_racket = left_pos.copy()
        self.prev_right_racket = right_pos.copy()

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

    def _rally_done(self) -> bool:
        if self.shuttle_pos[2] < 0.06:
            print("RALLY END: landed. hits=", self.hit_count)
            return True
        if abs(self.shuttle_pos[0]) > 4.45 or abs(self.shuttle_pos[1]) > 2.55:
            print("RALLY END: out. hits=", self.hit_count)
            return True
        if abs(self.shuttle_pos[0]) < 0.06 and self.shuttle_pos[2] < self.net_height:
            print("RALLY END: net fail. hits=", self.hit_count)
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        try:
            while True:
                receiver = self._receiver_side()
                passive = self._other_side(receiver)

                policy_action = self._policy_action(receiver)
                self._apply_active_policy(receiver, policy_action)
                self._apply_passive_ready(passive)

                sim_steps = max(1, int(round(self.control_dt / self.sim_dt)))
                for _ in range(sim_steps):
                    p.stepSimulation()

                self._update_racket_memory()
                self._maybe_hit(receiver)
                self._update_shuttle()

                self.step_count += 1
                if self._rally_done():
                    time.sleep(0.35)
                    self.reset_rally()

                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            p.disconnect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs_rally_easy/badminton_rally_final.zip")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--left_urdf", type=str, default="models/humanoid_left.urdf")
    parser.add_argument("--right_urdf", type=str, default="models/humanoid_right.urdf")
    parser.add_argument("--hit_radius", type=float, default=1.20)
    parser.add_argument("--free_base", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_policy_footwork",
        action="store_true",
        help="Use policy-predicted base movement instead of prediction-based footwork. Not recommended for stable demo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = PaperLikeTwoRobotRally(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        left_urdf=args.left_urdf,
        right_urdf=args.right_urdf,
        hit_radius=args.hit_radius,
        fixed_base=not args.free_base,
        seed=args.seed,
        use_policy_footwork=args.use_policy_footwork,
    )
    demo.run()
