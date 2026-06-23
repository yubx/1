#!/usr/bin/env python3
"""
Improved two-robot shared-policy rally demo for the EASY badminton PPO model.

This script is meant for the model trained by:
    train_rl_badminton_easy.py

Why it improves the two-robot result:
1. Uses the same trained policy for both robots.
2. Mirrors the right robot's observation into the left-side policy convention.
3. Computes a target-known interception point online for each side.
4. Lets only the receiving side actively chase/swing; the other side recenters.
5. Uses racket-triggered but assisted return velocity so rallies are easier to sustain.

Save as:
    play_two_robots_easy_improved.py

Run:
    python play_two_robots_easy_improved.py --model runs_easy\badminton_easy_final.zip --vecnorm runs_easy\vecnormalize_easy_final.pkl

If rallies are still short, increase --hit_radius:
    python play_two_robots_easy_improved.py --model runs_easy\badminton_easy_final.zip --hit_radius 1.05
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


OBS_DIM = 18 + 2 + 6 + 3 + 1 + 1
ACTION_DIM = 4

JOINT_NAMES = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_shoulder_pitch", "left_elbow", "left_wrist",
    "right_shoulder_pitch", "right_elbow", "right_wrist",
]


class NormDummyEnv(gym.Env):
    """Dummy env only for loading VecNormalize stats."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


class ImprovedTwoRobotEasyRally:
    def __init__(
        self,
        model_path: str,
        vecnorm_path: Optional[str],
        left_urdf: str = "models/humanoid_left.urdf",
        right_urdf: str = "models/humanoid_right.urdf",
        hit_radius: float = 0.95,
        fixed_base: bool = True,
        seed: int = 0,
    ):
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.left_urdf = left_urdf
        self.right_urdf = right_urdf
        self.hit_radius = hit_radius
        self.fixed_base = fixed_base
        self.rng = np.random.default_rng(seed)

        self.control_dt = 0.02
        self.sim_dt = 1.0 / 240.0
        self.net_height = 1.55

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(6.2, 35, -25, [0.0, 0.0, 1.0])

        self.model = PPO.load(model_path)
        self.vecnorm = self._load_vecnormalize(vecnorm_path)

        self.left_robot = -1
        self.right_robot = -1
        self.left_map: Dict[str, int] = {}
        self.right_map: Dict[str, int] = {}
        self.left_racket_link = -1
        self.right_racket_link = -1
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_vel = np.zeros(3)
        self.right_racket_vel = np.zeros(3)
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0

        self.shuttle_id = -1
        self.shuttle_pos = np.zeros(3)
        self.shuttle_vel = np.zeros(3)
        self.shuttle_hist = deque(maxlen=6)
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.010)

        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -100
        self.last_hitter = None

        self._build_world()
        self.reset_rally()

    # ------------------------------------------------------------------
    # Model and normalization
    # ------------------------------------------------------------------

    def _load_vecnormalize(self, path: Optional[str]):
        if path is None:
            path = os.path.join(os.path.dirname(self.model_path), "vecnormalize_easy_final.pkl")
        if not os.path.exists(path):
            print("Warning: VecNormalize not found:", path)
            return None
        dummy = DummyVecEnv([lambda: NormDummyEnv()])
        vecnorm = VecNormalize.load(path, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print("Loaded VecNormalize:", path)
        return vecnorm

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32).reshape(1, -1)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        return obs

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _build_world(self):
        for path in [self.left_urdf, self.right_urdf]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}. Run 2.py once first to generate the URDF files.")

        p.loadURDF("plane.urdf")
        self._add_court()
        self._add_net()

        self.left_robot = p.loadURDF(self.left_urdf, [-2.45, 0.0, 1.30], [0, 0, 0, 1], useFixedBase=self.fixed_base)
        self.right_robot = p.loadURDF(self.right_urdf, [2.45, 0.0, 1.30], [0, 0, 1, 0], useFixedBase=self.fixed_base)

        self.left_map = self._make_joint_map(self.left_robot)
        self.right_map = self._make_joint_map(self.right_robot)
        self.left_racket_link = self._find_link(self.left_robot, ["left_racket", "left_hand"])
        self.right_racket_link = self._find_link(self.right_robot, ["right_racket", "right_hand"])

        self.shuttle_id = self._create_shuttle()
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

        print("Left racket link:", self.left_racket_link)
        print("Right racket link:", self.right_racket_link)

    def _add_court(self):
        # Full enough visual court for demo.
        green = [0.08, 0.42, 0.18, 1.0]
        white = [1.0, 1.0, 1.0, 1.0]
        z = 0.006
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.15, 1.95, z / 2], rgbaColor=green)
        p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[0, 0, z / 2])

        def line(pos, size):
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=white)
            p.createMultiBody(0, baseVisualShapeIndex=v, basePosition=pos)

        line([0, -1.85, 0.015], [4.0, 0.018, 0.006])
        line([0, 1.85, 0.015], [4.0, 0.018, 0.006])
        line([-4.0, 0, 0.015], [0.018, 1.85, 0.006])
        line([4.0, 0, 0.015], [0.018, 1.85, 0.006])
        line([0, 0, 0.018], [0.018, 1.85, 0.006])
        line([-1.2, 0, 0.018], [0.018, 1.85, 0.006])
        line([1.2, 0, 0.018], [0.018, 1.85, 0.006])
        line([-2.7, 0, 0.018], [0.018, 1.85, 0.006])
        line([2.7, 0, 0.018], [0.018, 1.85, 0.006])
        line([-2.6, 0, 0.018], [1.4, 0.018, 0.006])
        line([2.6, 0, 0.018], [1.4, 0.018, 0.006])

    def _add_net(self):
        h = self.net_height
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, h / 2],
            rgbaColor=[0.2, 0.8, 0.2, 0.45],
        )
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2])
        p.createMultiBody(0, collision, visual, [0, 0, h / 2])

    def _create_shuttle(self):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.038, length=0.09, rgbaColor=[0.95, 0.95, 1.0, 1.0])
        return p.createMultiBody(0.005, col, vis, [0, 0, 1.4])

    def _make_joint_map(self, robot_id: int) -> Dict[str, int]:
        mapping = {}
        for i in range(p.getNumJoints(robot_id)):
            mapping[p.getJointInfo(robot_id, i)[1].decode("utf-8")] = i
        missing = [name for name in JOINT_NAMES if name not in mapping]
        if missing:
            raise RuntimeError(f"URDF missing required joints: {missing}")
        return mapping

    def _find_link(self, robot_id: int, candidates: List[str]) -> int:
        for name in candidates:
            for i in range(p.getNumJoints(robot_id)):
                if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
                    return i
        raise RuntimeError(f"Cannot find link from {candidates}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_rally(self):
        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -100
        self.last_hitter = None
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0

        p.resetBasePositionAndOrientation(self.left_robot, [-2.45, 0.0, 1.30], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.right_robot, [2.45, 0.0, 1.30], [0, 0, 1, 0])
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

        receiver = "left" if self.rng.random() < 0.5 else "right"
        self._serve_to(receiver)
        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())
        print("NEW RALLY, receiver:", receiver)

    def _reset_neutral(self, robot_id: int, joint_map: Dict[str, int]):
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
        for name, value in neutral.items():
            jid = joint_map[name]
            p.resetJointState(robot_id, jid, value, 0.0)
            p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=value, force=180, maxVelocity=6.0)

    def _serve_to(self, receiver: str):
        if receiver == "left":
            start = np.array([3.15, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.70)], dtype=float)
            target = np.array([-1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)], dtype=float)
        else:
            start = np.array([-3.15, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.70)], dtype=float)
            target = np.array([1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)], dtype=float)
        T = self.rng.uniform(0.95, 1.15)
        self.shuttle_pos = start
        self.shuttle_vel = self._ballistic_velocity(start, target, T)
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])
        print("SERVE:", self.shuttle_pos.round(2), "->", target.round(2), "vel", self.shuttle_vel.round(2))

    def _ballistic_velocity(self, start: np.ndarray, target: np.ndarray, T: float) -> np.ndarray:
        g = -9.8
        return np.array(
            [
                (target[0] - start[0]) / T,
                (target[1] - start[1]) / T,
                (target[2] - start[2] - 0.5 * g * T * T) / T,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Policy observation
    # ------------------------------------------------------------------

    def _racket_state(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        if side == "left":
            robot, link, prev = self.left_robot, self.left_racket_link, self.prev_left_racket
        else:
            robot, link, prev = self.right_robot, self.right_racket_link, self.prev_right_racket
        state = p.getLinkState(robot, link, computeLinkVelocity=1)
        pos = np.array(state[0], dtype=float)
        vel = np.zeros(3) if prev is None else (pos - prev) / self.control_dt
        return pos, vel

    def _estimate_intercept(self, side: str) -> Tuple[np.ndarray, float]:
        """Estimate target-known hit point for the current receiver.

        The easy policy was trained with target position and time-to-hit in the observation.
        Two-robot rollout works better if we reconstruct those fields online.
        """
        target_x = -1.65 if side == "left" else 1.65
        vx = self.shuttle_vel[0]
        if abs(vx) < 0.2:
            t_hit = 0.7
        else:
            t_hit = (target_x - self.shuttle_pos[0]) / vx
        t_hit = float(np.clip(t_hit, 0.05, 1.4))
        g = -9.8
        target = self.shuttle_pos + self.shuttle_vel * t_hit + np.array([0, 0, 0.5 * g * t_hit * t_hit])
        target[0] = target_x
        target[1] = float(np.clip(target[1], -0.75, 0.75))
        target[2] = float(np.clip(target[2], 1.20, 1.70))
        return target, t_hit

    def _make_obs(self, side: str) -> np.ndarray:
        if side == "left":
            robot = self.left_robot
            racket_pos, racket_vel = self._racket_state("left")
            base_pos, _ = p.getBasePositionAndOrientation(robot)
            hist = np.array(list(self.shuttle_hist), dtype=float)
            target, t_hit = self._estimate_intercept("left")
            phase = self.last_phase_left
        else:
            robot = self.right_robot
            racket_pos, racket_vel = self._racket_state("right")
            base_pos, _ = p.getBasePositionAndOrientation(robot)
            hist = np.array(list(self.shuttle_hist), dtype=float)
            target, t_hit = self._estimate_intercept("right")

            # Mirror actual right side into the left-side policy convention.
            hist[:, 0] *= -1.0
            base_pos = list(base_pos)
            base_pos[0] *= -1.0
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
                np.array(base_pos[:2], dtype=float) * np.array([0.25, 0.5]),
                racket_pos * np.array([0.25, 0.5, 0.5]),
                racket_vel * 0.1,
                target * np.array([0.25, 0.5, 0.5]),
                np.array([t_hit, phase], dtype=float),
            ]
        ).astype(np.float32)
        return obs

    def _predict_action(self, side: str) -> np.ndarray:
        obs = self._make_obs(side)
        norm_obs = self._normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return np.asarray(action).reshape(-1).astype(float)

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------

    def _apply_active_policy(self, side: str, action: np.ndarray):
        if side == "left":
            self._apply_action_to_robot(self.left_robot, self.left_map, action, "left", active=True)
            self.last_phase_left = float((action[2] + 1.0) * 0.5)
        else:
            action = action.copy()
            # x movement was predicted in mirrored coordinates.
            action[0] *= -1.0
            self._apply_action_to_robot(self.right_robot, self.right_map, action, "right", active=True)
            self.last_phase_right = float((action[2] + 1.0) * 0.5)

    def _apply_passive_ready(self, side: str):
        if side == "left":
            self._move_base_toward(self.left_robot, [-2.45, 0.0], "left")
            self._apply_ready_pose(self.left_robot, self.left_map, "left")
        else:
            self._move_base_toward(self.right_robot, [2.45, 0.0], "right")
            self._apply_ready_pose(self.right_robot, self.right_map, "right")

    def _move_base_toward(self, robot_id: int, target_xy: List[float], side: str):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        pos = np.array(pos, dtype=float)
        target = np.array(target_xy, dtype=float)
        delta = np.clip(target - pos[:2], -0.025, 0.025)
        pos[:2] += delta
        if side == "left":
            orn = [0, 0, 0, 1]
            pos[0] = np.clip(pos[0], -3.7, -0.55)
        else:
            orn = [0, 0, 1, 0]
            pos[0] = np.clip(pos[0], 0.55, 3.7)
        pos[1] = np.clip(pos[1], -1.75, 1.75)
        p.resetBasePositionAndOrientation(robot_id, pos, orn)

    def _apply_action_to_robot(self, robot_id: int, joint_map: Dict[str, int], action: np.ndarray, side: str, active: bool):
        # Base movement.
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        pos = np.array(pos, dtype=float)
        pos[0] += float(action[0]) * 1.8 * self.control_dt
        pos[1] += float(action[1]) * 1.5 * self.control_dt
        if side == "left":
            orn = [0, 0, 0, 1]
            pos[0] = np.clip(pos[0], -3.7, -0.55)
            racket_side = "left"
        else:
            orn = [0, 0, 1, 0]
            pos[0] = np.clip(pos[0], 0.55, 3.7)
            racket_side = "right"
        pos[1] = np.clip(pos[1], -1.75, 1.75)
        p.resetBasePositionAndOrientation(robot_id, pos, orn)

        phase = float((action[2] + 1.0) * 0.5)
        power = float((action[3] + 1.0) * 0.5)
        targets = self._swing_primitive(phase, power, racket_side)
        targets.update({
            "left_hip": 0.05,
            "right_hip": -0.05,
            "left_knee": 0.22,
            "right_knee": 0.22,
            "left_ankle": -0.08,
            "right_ankle": -0.08,
        })
        self._apply_joint_targets(robot_id, joint_map, targets)

    def _apply_ready_pose(self, robot_id: int, joint_map: Dict[str, int], side: str):
        racket_side = "left" if side == "left" else "right"
        targets = self._swing_primitive(0.12, 0.2, racket_side)
        targets.update({
            "left_hip": 0.0,
            "right_hip": 0.0,
            "left_knee": 0.18,
            "right_knee": 0.18,
            "left_ankle": -0.05,
            "right_ankle": -0.05,
        })
        self._apply_joint_targets(robot_id, joint_map, targets)

    def _apply_joint_targets(self, robot_id: int, joint_map: Dict[str, int], targets: Dict[str, float]):
        for name, value in targets.items():
            if name not in joint_map:
                continue
            p.setJointMotorControl2(
                robot_id,
                joint_map[name],
                p.POSITION_CONTROL,
                targetPosition=float(value),
                force=260,
                maxVelocity=10.0,
            )

    def _swing_primitive(self, phase: float, power: float, racket_side: str) -> Dict[str, float]:
        amp = 0.75 + 0.65 * power
        if phase < 0.35:
            t = phase / 0.35
            shoulder = 0.20 + amp * 1.25 * t
            elbow = 0.30 + amp * 0.70 * t
            wrist = 0.00 + amp * 0.45 * t
        elif phase < 0.65:
            t = (phase - 0.35) / 0.30
            shoulder = 1.45 * amp - 2.15 * amp * t
            elbow = 1.00 * amp - 0.75 * amp * t
            wrist = 0.50 * amp - 0.95 * amp * t
        else:
            t = (phase - 0.65) / 0.35
            shoulder = -0.65 * amp + 0.90 * amp * t
            elbow = 0.25 + 0.15 * t
            wrist = -0.45 * amp + 0.45 * amp * t

        if racket_side == "left":
            return {
                "left_shoulder_pitch": float(np.clip(shoulder, -1.8, 1.8)),
                "left_elbow": float(np.clip(elbow, 0.0, 1.8)),
                "left_wrist": float(np.clip(wrist, -0.9, 0.9)),
                "right_shoulder_pitch": float(np.clip(-0.35 * shoulder, -1.8, 1.8)),
                "right_elbow": 0.35,
                "right_wrist": float(np.clip(-0.3 * wrist, -0.9, 0.9)),
            }
        return {
            "right_shoulder_pitch": float(np.clip(-shoulder, -1.8, 1.8)),
            "right_elbow": float(np.clip(elbow, 0.0, 1.8)),
            "right_wrist": float(np.clip(-wrist, -0.9, 0.9)),
            "left_shoulder_pitch": float(np.clip(0.35 * shoulder, -1.8, 1.8)),
            "left_elbow": 0.35,
            "left_wrist": float(np.clip(0.3 * wrist, -0.9, 0.9)),
        }

    # ------------------------------------------------------------------
    # Hit and shuttle
    # ------------------------------------------------------------------

    def _maybe_hit(self, side: str):
        if self.step_count - self.last_hit_step < 25:
            return
        racket_pos, racket_vel = self._racket_state(side)
        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        if side == "left":
            forward = racket_vel[0]
            outgoing_target = np.array([1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)])
        else:
            forward = -racket_vel[0]
            outgoing_target = np.array([-1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)])

        good_height = 0.35 < self.shuttle_pos[2] < 1.95
        in_correct_half = self.shuttle_pos[0] < 0 if side == "left" else self.shuttle_pos[0] > 0
        if not (in_correct_half and good_height and dist < self.hit_radius and forward > 0.08):
            return

        # Racket-triggered assisted return. The hit is only allowed when the racket is near the shuttle,
        # but the outgoing trajectory is stabilized toward a reachable region on the other side.
        T = self.rng.uniform(0.92, 1.12)
        vout = self._ballistic_velocity(self.shuttle_pos, outgoing_target, T)
        vout[1] += 0.15 * racket_vel[1]
        vout[2] = max(vout[2], 4.7)
        speed = np.linalg.norm(vout)
        if speed > 13.5:
            vout = vout / speed * 13.5
        self.shuttle_vel = vout
        self.hit_count += 1
        self.last_hit_step = self.step_count
        self.last_hitter = side
        print(f"{side.upper()} HIT #{self.hit_count}: dist={dist:.2f}, forward={forward:.2f}, target={outgoing_target.round(2)}, vout={vout.round(2)}")

    def _update_shuttle(self):
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(self.shuttle_pos, self.shuttle_vel, self.control_dt, None)
        self.shuttle_hist.append(self.shuttle_pos.copy())
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])

    def _rally_done(self) -> bool:
        if self.shuttle_pos[2] < 0.06:
            print("RALLY END: landed. hits=", self.hit_count)
            return True
        if abs(self.shuttle_pos[0]) > 4.4 or abs(self.shuttle_pos[1]) > 2.5:
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
                receiver = "left" if self.shuttle_pos[0] < 0 else "right"
                passive = "right" if receiver == "left" else "left"

                action = self._predict_action(receiver)
                self._apply_active_policy(receiver, action)
                self._apply_passive_ready(passive)

                sim_steps = max(1, int(round(self.control_dt / self.sim_dt)))
                for _ in range(sim_steps):
                    p.stepSimulation()

                left_pos, self.left_racket_vel = self._racket_state("left")
                right_pos, self.right_racket_vel = self._racket_state("right")
                self.prev_left_racket = left_pos.copy()
                self.prev_right_racket = right_pos.copy()

                self._maybe_hit(receiver)
                self._update_shuttle()
                self.step_count += 1

                if self._rally_done():
                    time.sleep(0.4)
                    self.reset_rally()

                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("Exit")
        finally:
            p.disconnect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs_easy/badminton_easy_final.zip")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--left_urdf", type=str, default="models/humanoid_left.urdf")
    parser.add_argument("--right_urdf", type=str, default="models/humanoid_right.urdf")
    parser.add_argument("--hit_radius", type=float, default=0.95)
    parser.add_argument("--free_base", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = ImprovedTwoRobotEasyRally(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        left_urdf=args.left_urdf,
        right_urdf=args.right_urdf,
        hit_radius=args.hit_radius,
        fixed_base=not args.free_base,
        seed=args.seed,
    )
    demo.run()
