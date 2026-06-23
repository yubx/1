#!/usr/bin/env python3
"""
Two-side PPO training + two-robot shared-policy rally demo for the simplified
PyBullet humanoid badminton model.

Save as:
    train_rl_badminton_two_side.py

Why this version is different from train_rl_badminton_rally_easy.py:
- The old rally_easy environment was still a canonical LEFT-side receiver task.
- This file trains LEFT and RIGHT receiving episodes in the same environment.
- The right side is mirrored into the same canonical policy coordinate frame.
- The actual right robot is still executed with right-hand racket motion.
- Base speed is trained faster, so the second robot no longer moves too slowly.

Install:
    pip install gymnasium stable-baselines3 tensorboard

Before training:
    python 2.py
    # This should generate:
    # models/humanoid_left.urdf
    # models/humanoid_right.urdf

Train:
    python train_rl_badminton_two_side.py --train --timesteps 2000000 --n_envs 4

Test single side:
    python train_rl_badminton_two_side.py --play_left  --model runs_two_side\badminton_two_side_final.zip
    python train_rl_badminton_two_side.py --play_right --model runs_two_side\badminton_two_side_final.zip

Two-robot shared-policy rally:
    python train_rl_badminton_two_side.py --play_two --model runs_two_side\badminton_two_side_final.zip

If two-robot rally is still strict:
    python train_rl_badminton_two_side.py --play_two --model runs_two_side\badminton_two_side_final.zip --hit_radius 1.25

Notes:
- This is still a simplified PyBullet prototype, not a full Isaac Gym 4096-env
  whole-body humanoid implementation.
- The action remains high-level and learnable:
      action[0] = base x movement
      action[1] = base y movement
      action[2] = swing phase
      action[3] = swing power
- The observation remains close to your previous code:
      6-frame shuttle history + base xy + racket pos/vel + hit target + time-to-hit + phase
"""

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from physics_shuttle import ShuttlePhysics


JOINT_NAMES = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_shoulder_pitch", "left_elbow", "left_wrist",
    "right_shoulder_pitch", "right_elbow", "right_wrist",
]

OBS_DIM = 18 + 2 + 6 + 3 + 1 + 1
ACTION_DIM = 4


@dataclass
class TwoSideConfig:
    left_urdf: str = "models/humanoid_left.urdf"
    right_urdf: str = "models/humanoid_right.urdf"
    gui: bool = False
    fixed_base: bool = True
    control_dt: float = 0.02
    sim_dt: float = 1.0 / 240.0
    max_episode_time: float = 2.4
    stage: int = 1
    seed: int = 0
    hit_radius: float = 0.95
    forced_side: Optional[str] = None  # None, "left", or "right"


class TwoSideBadmintonEnv(gym.Env):
    """Random-left/right single-hit environment.

    Each episode chooses a receiving side. The policy always sees a canonical
    left-side observation. If the active receiver is the right robot, the world is
    mirrored into that canonical frame. The action is then mapped back to the
    actual right robot.

    This gives one shared policy that can control both robots in a later rally.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[TwoSideConfig] = None):
        super().__init__()
        self.cfg = cfg or TwoSideConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.client = p.connect(p.GUI if self.cfg.gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.cfg.sim_dt, physicsClientId=self.client)
        if self.cfg.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(6.2, 35, -25, [0.0, 0.0, 1.0], physicsClientId=self.client)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.left_robot = -1
        self.right_robot = -1
        self.shuttle_id = -1
        self.left_map: Dict[str, int] = {}
        self.right_map: Dict[str, int] = {}
        self.left_racket_link = -1
        self.right_racket_link = -1

        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_pos = np.zeros(3, dtype=float)
        self.right_racket_pos = np.zeros(3, dtype=float)
        self.left_racket_vel = np.zeros(3, dtype=float)
        self.right_racket_vel = np.zeros(3, dtype=float)

        self.shuttle_hist = deque(maxlen=6)
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.010)
        self.shuttle_pos = np.zeros(3, dtype=float)
        self.shuttle_vel = np.zeros(3, dtype=float)
        self.hit_target = np.zeros(3, dtype=float)
        self.time_to_hit = 1.0

        self.active_side = "left"
        self.has_hit = False
        self.cleared_net = False
        self.landed = False
        self.elapsed = 0.0
        self.step_count = 0
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0
        self.last_action = np.zeros(ACTION_DIM, dtype=float)

        self._build_world()

    def close(self):
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1

    def set_stage(self, stage: int):
        self.cfg.stage = int(stage)

    # ------------------------------------------------------------------
    # World
    # ------------------------------------------------------------------

    def _build_world(self):
        for path in [self.cfg.left_urdf, self.cfg.right_urdf]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run python 2.py once first to generate humanoid URDF files."
                )

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.cfg.sim_dt, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self._add_court()
        self._add_net()

        self.left_robot = p.loadURDF(
            self.cfg.left_urdf,
            [-2.45, 0.0, 1.30],
            [0, 0, 0, 1],
            useFixedBase=self.cfg.fixed_base,
            physicsClientId=self.client,
        )
        self.right_robot = p.loadURDF(
            self.cfg.right_urdf,
            [2.45, 0.0, 1.30],
            [0, 0, 1, 0],
            useFixedBase=self.cfg.fixed_base,
            physicsClientId=self.client,
        )

        self.left_map = self._make_joint_map(self.left_robot)
        self.right_map = self._make_joint_map(self.right_robot)
        self.left_racket_link = self._find_link(self.left_robot, ["left_racket", "racket", "left_hand"])
        self.right_racket_link = self._find_link(self.right_robot, ["right_racket", "racket", "right_hand"])
        self.shuttle_id = self._create_shuttle()

    def _add_court(self):
        green = [0.08, 0.42, 0.18, 1.0]
        green2 = [0.10, 0.50, 0.22, 1.0]
        white = [1.0, 1.0, 1.0, 1.0]
        z = 0.006

        def add_box(center, half_extents, rgba):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=self.client)
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=center, physicsClientId=self.client)

        add_box([0, 0, z / 2], [4.15, 1.95, z / 2], green)
        add_box([-2.55, 0, z + 0.001], [1.35, 1.85, 0.001], green2)
        add_box([2.55, 0, z + 0.001], [1.35, 1.85, 0.001], green2)

        def line(pos, size):
            add_box(pos, size, white)

        line([0, -1.85, 0.016], [4.0, 0.018, 0.006])
        line([0, 1.85, 0.016], [4.0, 0.018, 0.006])
        line([-4.0, 0, 0.016], [0.018, 1.85, 0.006])
        line([4.0, 0, 0.016], [0.018, 1.85, 0.006])
        line([0, 0, 0.018], [0.018, 1.85, 0.006])
        line([-1.20, 0, 0.018], [0.018, 1.85, 0.006])
        line([1.20, 0, 0.018], [0.018, 1.85, 0.006])
        line([-2.70, 0, 0.018], [0.018, 1.85, 0.006])
        line([2.70, 0, 0.018], [0.018, 1.85, 0.006])
        line([-2.60, 0, 0.018], [1.40, 0.018, 0.006])
        line([2.60, 0, 0.018], [1.40, 0.018, 0.006])

    def _add_net(self):
        h = 1.55
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, h / 2],
            rgbaColor=[0.2, 0.8, 0.2, 0.45],
            physicsClientId=self.client,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, h / 2],
            physicsClientId=self.client,
        )
        p.createMultiBody(0, collision, visual, [0, 0, h / 2], physicsClientId=self.client)

    def _create_shuttle(self):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09, physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.038,
            length=0.09,
            rgbaColor=[0.95, 0.95, 1.0, 1.0],
            physicsClientId=self.client,
        )
        return p.createMultiBody(0.005, col, vis, [0, 0, 1.4], physicsClientId=self.client)

    def _make_joint_map(self, robot_id: int) -> Dict[str, int]:
        mapping = {}
        for i in range(p.getNumJoints(robot_id, physicsClientId=self.client)):
            name = p.getJointInfo(robot_id, i, physicsClientId=self.client)[1].decode("utf-8")
            mapping[name] = i
        missing = [name for name in JOINT_NAMES if name not in mapping]
        if missing:
            raise RuntimeError(f"URDF missing required joints: {missing}")
        return mapping

    def _find_link(self, robot_id: int, candidates: List[str]) -> int:
        for name in candidates:
            for i in range(p.getNumJoints(robot_id, physicsClientId=self.client)):
                link = p.getJointInfo(robot_id, i, physicsClientId=self.client)[12].decode("utf-8")
                if link == name:
                    return i
        raise RuntimeError(f"Cannot find any racket/hand link from {candidates}")

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.cfg.forced_side in {"left", "right"}:
            self.active_side = self.cfg.forced_side
        else:
            self.active_side = "left" if self.rng.random() < 0.5 else "right"

        self.has_hit = False
        self.cleared_net = False
        self.landed = False
        self.elapsed = 0.0
        self.step_count = 0
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0
        self.last_action[:] = 0.0
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_vel[:] = 0
        self.right_racket_vel[:] = 0

        self._reset_robot_base_and_pose()
        self._sample_incoming_shuttle(self.active_side)

        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())

        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0], physicsClientId=self.client)
        self._update_racket_memory()
        return self._get_obs(self.active_side), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)
        prev_action = self.last_action.copy()
        self.last_action = action.copy()

        self._apply_active_action(self.active_side, action)
        self._apply_passive_ready(self._other_side(self.active_side))

        for _ in range(max(1, int(round(self.cfg.control_dt / self.cfg.sim_dt)))):
            p.stepSimulation(physicsClientId=self.client)

        self._update_racket_memory()
        racket_pos, racket_vel = self._active_racket_state(self.active_side)
        force = self._hit_force(self.active_side, racket_pos, racket_vel)

        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(
            self.shuttle_pos,
            self.shuttle_vel,
            self.cfg.control_dt,
            force,
        )
        self.shuttle_hist.append(self.shuttle_pos.copy())
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0], physicsClientId=self.client)

        self.elapsed += self.cfg.control_dt
        self.time_to_hit -= self.cfg.control_dt
        self.step_count += 1

        reward, info = self._reward(self.active_side, racket_pos, racket_vel, action, prev_action)
        terminated = bool(info.get("landed_in", False))
        truncated = self.elapsed > self.cfg.max_episode_time or self._failure()
        return self._get_obs(self.active_side), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reset / sampling
    # ------------------------------------------------------------------

    def _reset_robot_base_and_pose(self):
        p.resetBasePositionAndOrientation(self.left_robot, [-2.45, 0.0, 1.30], [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.right_robot, [2.45, 0.0, 1.30], [0, 0, 1, 0], physicsClientId=self.client)
        p.resetBaseVelocity(self.left_robot, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        p.resetBaseVelocity(self.right_robot, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

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
            p.resetJointState(robot_id, jid, value, 0.0, physicsClientId=self.client)
            p.setJointMotorControl2(
                robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=value,
                force=180,
                maxVelocity=6.0,
                physicsClientId=self.client,
            )

    def _sample_incoming_shuttle(self, receiver: str):
        """Sample diverse incoming trajectories for left or right side."""
        if receiver == "left":
            target_x_range = (-1.95, -1.25)
            start_x_ranges = {
                "serve": (2.7, 3.6),
                "flat_return": (1.1, 2.4),
                "loft_return": (1.3, 3.0),
                "short_return": (0.7, 1.8),
            }
        else:
            target_x_range = (1.25, 1.95)
            start_x_ranges = {
                "serve": (-3.6, -2.7),
                "flat_return": (-2.4, -1.1),
                "loft_return": (-3.0, -1.3),
                "short_return": (-1.8, -0.7),
            }

        self.hit_target = np.array(
            [
                self.rng.uniform(*target_x_range),
                self.rng.uniform(-0.80, 0.80),
                self.rng.uniform(1.20, 1.72),
            ],
            dtype=float,
        )

        mode = self.rng.choice(
            ["serve", "flat_return", "loft_return", "short_return"],
            p=[0.30, 0.35, 0.25, 0.10],
        )
        x_lo, x_hi = start_x_ranges[mode]
        if mode == "serve":
            start = np.array([self.rng.uniform(x_lo, x_hi), self.rng.uniform(-0.95, 0.95), self.rng.uniform(1.30, 1.80)], dtype=float)
            t_flight = self.rng.uniform(0.85, 1.18)
        elif mode == "flat_return":
            start = np.array([self.rng.uniform(x_lo, x_hi), self.rng.uniform(-0.95, 0.95), self.rng.uniform(1.20, 1.85)], dtype=float)
            t_flight = self.rng.uniform(0.55, 0.90)
        elif mode == "loft_return":
            start = np.array([self.rng.uniform(x_lo, x_hi), self.rng.uniform(-1.10, 1.10), self.rng.uniform(1.55, 2.40)], dtype=float)
            t_flight = self.rng.uniform(0.80, 1.35)
        else:
            start = np.array([self.rng.uniform(x_lo, x_hi), self.rng.uniform(-1.00, 1.00), self.rng.uniform(1.10, 1.65)], dtype=float)
            t_flight = self.rng.uniform(0.45, 0.75)

        self.time_to_hit = float(t_flight)
        self.shuttle_pos = start
        self.shuttle_vel = self._ballistic_velocity(start, self.hit_target, t_flight)
        self.shuttle_vel += np.array(
            [
                self.rng.uniform(-0.30, 0.30),
                self.rng.uniform(-0.30, 0.30),
                self.rng.uniform(-0.18, 0.22),
            ],
            dtype=float,
        )

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

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, side: str) -> np.ndarray:
        if side == "left":
            robot_id = self.left_robot
            racket_pos = self.left_racket_pos.copy()
            racket_vel = self.left_racket_vel.copy()
            base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client)
            base_xy = np.array(base_pos[:2], dtype=float)
            hist = np.array(list(self.shuttle_hist), dtype=float)
            target = self.hit_target.copy()
            phase = self.last_phase_left
        else:
            robot_id = self.right_robot
            racket_pos = self.right_racket_pos.copy()
            racket_vel = self.right_racket_vel.copy()
            base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client)
            base_xy = np.array(base_pos[:2], dtype=float)
            hist = np.array(list(self.shuttle_hist), dtype=float)
            target = self.hit_target.copy()

            # Mirror actual right side to canonical left-side policy frame.
            hist[:, 0] *= -1.0
            base_xy[0] *= -1.0
            racket_pos[0] *= -1.0
            racket_vel[0] *= -1.0
            target[0] *= -1.0
            phase = self.last_phase_right

        if hist.shape[0] < 6:
            hist = np.tile(self.shuttle_pos, (6, 1))

        obs = np.concatenate(
            [
                hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
                base_xy * np.array([0.25, 0.5]),
                racket_pos * np.array([0.25, 0.5, 0.5]),
                racket_vel * 0.1,
                target * np.array([0.25, 0.5, 0.5]),
                np.array([self.time_to_hit, phase], dtype=float),
            ]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Action application and swing
    # ------------------------------------------------------------------

    def _apply_active_action(self, side: str, canonical_action: np.ndarray):
        if side == "left":
            self._apply_action_to_robot(self.left_robot, self.left_map, canonical_action, side="left", racket_side="left")
            self.last_phase_left = float((canonical_action[2] + 1.0) * 0.5)
        else:
            actual_action = canonical_action.copy()
            # Canonical +x becomes actual -x for right-side player.
            actual_action[0] *= -1.0
            self._apply_action_to_robot(self.right_robot, self.right_map, actual_action, side="right", racket_side="right")
            self.last_phase_right = float((canonical_action[2] + 1.0) * 0.5)

    def _apply_action_to_robot(self, robot_id: int, joint_map: Dict[str, int], action: np.ndarray, side: str, racket_side: str):
        pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client)
        pos = np.array(pos, dtype=float)

        # Faster than the old 1.9/1.6. This is trained, not only added at playback.
        pos[0] += float(action[0]) * 3.8 * self.cfg.control_dt
        pos[1] += float(action[1]) * 3.0 * self.cfg.control_dt

        if side == "left":
            pos[0] = np.clip(pos[0], -3.85, -0.50)
            orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], 0.50, 3.85)
            orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -1.80, 1.80)
        p.resetBasePositionAndOrientation(robot_id, pos, orn, physicsClientId=self.client)

        phase = float((action[2] + 1.0) * 0.5)
        power = float((action[3] + 1.0) * 0.5)
        targets = self._swing_primitive(phase, power, racket_side)
        self._apply_leg_stance(targets, phase)
        self._apply_joint_targets(robot_id, joint_map, targets)

    def _apply_passive_ready(self, side: str):
        if side == "left":
            self._move_base_toward(self.left_robot, [-2.45, 0.0], "left", max_step=0.060)
            targets = self._swing_primitive(0.10, 0.20, "left")
            self._apply_leg_stance(targets, 0.10)
            self._apply_joint_targets(self.left_robot, self.left_map, targets)
        else:
            self._move_base_toward(self.right_robot, [2.45, 0.0], "right", max_step=0.060)
            targets = self._swing_primitive(0.10, 0.20, "right")
            self._apply_leg_stance(targets, 0.10)
            self._apply_joint_targets(self.right_robot, self.right_map, targets)

    def _move_base_toward(self, robot_id: int, target_xy: List[float], side: str, max_step: float = 0.060):
        pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.client)
        pos = np.array(pos, dtype=float)
        target = np.array(target_xy, dtype=float)
        pos[:2] += np.clip(target - pos[:2], -max_step, max_step)
        if side == "left":
            pos[0] = np.clip(pos[0], -3.85, -0.50)
            orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], 0.50, 3.85)
            orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -1.80, 1.80)
        p.resetBasePositionAndOrientation(robot_id, pos, orn, physicsClientId=self.client)

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
                physicsClientId=self.client,
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
        return {
            "right_shoulder_pitch": float(np.clip(-shoulder, -1.8, 1.8)),
            "right_elbow": float(np.clip(elbow, 0.0, 1.8)),
            "right_wrist": float(np.clip(-wrist, -0.9, 0.9)),
            "left_shoulder_pitch": float(np.clip(0.35 * shoulder, -1.8, 1.8)),
            "left_elbow": 0.35,
            "left_wrist": float(np.clip(0.3 * wrist, -0.9, 0.9)),
        }

    # ------------------------------------------------------------------
    # Racket, hit, reward
    # ------------------------------------------------------------------

    def _update_racket_memory(self):
        self.left_racket_pos, self.left_racket_vel = self._racket_state("left")
        self.right_racket_pos, self.right_racket_vel = self._racket_state("right")
        self.prev_left_racket = self.left_racket_pos.copy()
        self.prev_right_racket = self.right_racket_pos.copy()

    def _racket_state(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        if side == "left":
            robot_id, link_idx, prev = self.left_robot, self.left_racket_link, self.prev_left_racket
        else:
            robot_id, link_idx, prev = self.right_robot, self.right_racket_link, self.prev_right_racket
        state = p.getLinkState(robot_id, link_idx, computeLinkVelocity=1, physicsClientId=self.client)
        pos = np.array(state[0], dtype=float)
        vel = np.zeros(3, dtype=float) if prev is None else (pos - prev) / self.cfg.control_dt
        return pos, vel

    def _active_racket_state(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        if side == "left":
            return self.left_racket_pos.copy(), self.left_racket_vel.copy()
        return self.right_racket_pos.copy(), self.right_racket_vel.copy()

    def _hit_force(self, side: str, racket_pos: np.ndarray, racket_vel: np.ndarray):
        if self.has_hit:
            return None
        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        forward = float(racket_vel[0]) if side == "left" else float(-racket_vel[0])
        good_height = 0.30 < self.shuttle_pos[2] < 2.05
        near_time = -0.40 < self.time_to_hit < 0.48
        in_half = self.shuttle_pos[0] < 0.20 if side == "left" else self.shuttle_pos[0] > -0.20

        if dist < self.cfg.hit_radius and forward > 0.06 and good_height and near_time and in_half:
            self.has_hit = True
            if side == "left":
                target = np.array(
                    [self.rng.uniform(1.25, 2.35), self.rng.uniform(-0.80, 0.80), self.rng.uniform(1.35, 1.68)],
                    dtype=float,
                )
            else:
                target = np.array(
                    [self.rng.uniform(-2.35, -1.25), self.rng.uniform(-0.80, 0.80), self.rng.uniform(1.35, 1.68)],
                    dtype=float,
                )
            t_flight = self.rng.uniform(0.88, 1.12)
            vout = self._ballistic_velocity(self.shuttle_pos, target, t_flight)
            vout[1] += 0.12 * racket_vel[1]
            vout[2] = max(vout[2], 4.70)
            speed = np.linalg.norm(vout)
            if speed > 14.0:
                vout = vout / speed * 14.0
            mass = 0.005
            return mass * (vout - self.shuttle_vel) / self.cfg.control_dt
        return None

    def _reward(self, side: str, racket_pos: np.ndarray, racket_vel: np.ndarray, action: np.ndarray, prev_action: np.ndarray):
        stage = self.cfg.stage
        base_pos, _ = p.getBasePositionAndOrientation(self.left_robot if side == "left" else self.right_robot, physicsClientId=self.client)
        base_pos = np.array(base_pos, dtype=float)
        if side == "left":
            base_target = np.array([self.hit_target[0] - 0.55, self.hit_target[1]], dtype=float)
            forward = max(0.0, racket_vel[0])
            crossed = self.shuttle_pos[0] > 0.05
            landed_in = 0.20 < self.shuttle_pos[0] < 4.0 and abs(self.shuttle_pos[1]) < 1.85
        else:
            base_target = np.array([self.hit_target[0] + 0.55, self.hit_target[1]], dtype=float)
            forward = max(0.0, -racket_vel[0])
            crossed = self.shuttle_pos[0] < -0.05
            landed_in = -4.0 < self.shuttle_pos[0] < -0.20 and abs(self.shuttle_pos[1]) < 1.85

        base_dist = np.linalg.norm(base_pos[:2] - base_target)
        target_dist = np.linalg.norm(racket_pos - self.hit_target)
        shuttle_dist = np.linalg.norm(racket_pos - self.shuttle_pos)
        timing = np.exp(-abs(self.time_to_hit) / 0.22)

        r_foot = np.exp(-1.8 * base_dist)
        r_target = np.exp(-3.8 * target_dist)
        r_shuttle = np.exp(-4.5 * shuttle_dist)
        r_swing = timing * r_shuttle * forward
        phase = self.last_phase_left if side == "left" else self.last_phase_right
        r_phase = timing * np.exp(-8.0 * abs(phase - 0.50))

        reward = 0.0
        if stage == 1:
            reward += 3.5 * r_foot + 1.3 * r_target
        elif stage == 2:
            reward += 1.8 * r_foot + 3.2 * r_target + 1.4 * r_phase + 0.8 * r_swing
        else:
            reward += 1.0 * r_foot + 2.0 * r_target + 2.2 * r_shuttle + 1.5 * r_phase + 1.0 * r_swing

        reward -= 0.01 * float(np.mean(action ** 2))
        reward -= 0.02 * float(np.mean((action - prev_action) ** 2))

        info = {
            "hit": self.has_hit,
            "side": side,
            "stage": stage,
            "target_dist": target_dist,
            "shuttle_dist": shuttle_dist,
        }
        if self.has_hit:
            reward += 95.0
            info["hit_success"] = True

        if self.has_hit and not self.cleared_net and crossed:
            if self.shuttle_pos[2] > 1.62:
                self.cleared_net = True
                reward += 85.0
                info["cleared_net"] = True
            else:
                reward -= 22.0
                info["net_fail"] = True

        if self.has_hit and self.shuttle_pos[2] < 0.08:
            self.landed = True
            if landed_in:
                reward += 95.0
                info["landed_in"] = True
            else:
                reward -= 12.0
                info["landed_out"] = True

        if not self.has_hit and self.time_to_hit < -0.48:
            reward -= 14.0
            info["missed_window"] = True
        return float(reward), info

    def _failure(self) -> bool:
        if not self.has_hit and self.time_to_hit < -0.65:
            return True
        if self.shuttle_pos[2] < 0.05:
            return True
        if abs(self.shuttle_pos[0]) > 4.6 or abs(self.shuttle_pos[1]) > 2.6:
            return True
        return False

    def _other_side(self, side: str) -> str:
        return "right" if side == "left" else "left"


# ----------------------------------------------------------------------
# Vector env and training
# ----------------------------------------------------------------------


def make_env(rank: int, stage: int, gui: bool, seed: int, forced_side: Optional[str] = None):
    def _init():
        cfg = TwoSideConfig(gui=gui, stage=stage, seed=seed + rank, forced_side=forced_side)
        return TwoSideBadmintonEnv(cfg)
    return _init


def train(args):
    os.makedirs(args.run_dir, exist_ok=True)
    vec_cls = DummyVecEnv if args.n_envs == 1 else SubprocVecEnv
    env = vec_cls([make_env(i, 1, False, args.seed) for i in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    import torch.nn as nn
    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=6,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]), activation_fn=nn.ELU),
        tensorboard_log=os.path.join(args.run_dir, "tb"),
        verbose=1,
        seed=args.seed,
    )

    ckpt = CheckpointCallback(
        save_freq=max(args.n_steps * args.n_envs, 10000),
        save_path=args.run_dir,
        name_prefix="two_side_ckpt",
    )

    stage_steps = args.timesteps // 3
    for stage in [1, 2, 3]:
        print(f"\n========== TWO-SIDE TRAINING STAGE {stage} ==========")
        env.env_method("set_stage", stage)
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=(stage == 1), callback=ckpt, progress_bar=True)
        model.save(os.path.join(args.run_dir, f"badminton_two_side_s{stage}"))
        env.save(os.path.join(args.run_dir, f"vecnormalize_two_side_s{stage}.pkl"))

    model.save(os.path.join(args.run_dir, "badminton_two_side_final"))
    env.save(os.path.join(args.run_dir, "vecnormalize_two_side_final.pkl"))
    env.close()
    print("Two-side training complete.")


# ----------------------------------------------------------------------
# Single-side playback
# ----------------------------------------------------------------------


def _load_vec_env_for_play(model_path: str, vecnorm_path: Optional[str], seed: int, side: str):
    env = DummyVecEnv([make_env(0, 3, True, seed, forced_side=side)])
    if vecnorm_path is None:
        vecnorm_path = os.path.join(os.path.dirname(model_path), "vecnormalize_two_side_final.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize:", vecnorm_path)
    else:
        print("Warning: vecnormalize file not found:", vecnorm_path)
    return env


def play_single_side(args, side: str):
    env = _load_vec_env_for_play(args.model, args.vecnorm, args.seed, side)
    model = PPO.load(args.model, env=env)
    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            time.sleep(0.02)
            if bool(done[0]):
                print("episode end", info[0])
                # DummyVecEnv resets automatically on done.
    except KeyboardInterrupt:
        print("Stopped by user.")
    except p.error as exc:
        print("PyBullet GUI disconnected. Stop playback with Ctrl+C instead of closing the window.")
        print("Original error:", exc)
    finally:
        env.close()


# ----------------------------------------------------------------------
# Two-robot continuous rally playback
# ----------------------------------------------------------------------


class TwoRobotRallyPlayer:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.control_dt = 0.02
        self.sim_dt = 1.0 / 240.0
        self.net_height = 1.55
        self.hit_radius = float(args.hit_radius)
        self.assist_footwork = not args.policy_footwork

        self.client = p.connect(p.GUI)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet GUI.")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(6.4, 35, -25, [0.0, 0.0, 1.0])

        self.model = PPO.load(args.model)
        self.vecnorm = self._load_vecnormalize(args.vecnorm)
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.010)

        self.left_robot = -1
        self.right_robot = -1
        self.left_map = {}
        self.right_map = {}
        self.left_racket_link = -1
        self.right_racket_link = -1
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.left_racket_pos = np.zeros(3)
        self.right_racket_pos = np.zeros(3)
        self.left_racket_vel = np.zeros(3)
        self.right_racket_vel = np.zeros(3)
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0

        self.shuttle_id = -1
        self.shuttle_pos = np.zeros(3)
        self.shuttle_vel = np.zeros(3)
        self.shuttle_hist = deque(maxlen=6)
        self.step_count = 0
        self.hit_count = 0
        self.rally_count = 0
        self.last_hit_step = -100

        self._build_world()
        self.reset_rally()

    def _load_vecnormalize(self, path: Optional[str]):
        if path is None:
            path = os.path.join(os.path.dirname(self.args.model), "vecnormalize_two_side_final.pkl")
        if not path or not os.path.exists(path):
            print("Warning: VecNormalize file not found. Playback may be worse.")
            return None
        dummy = DummyVecEnv([lambda: NormDummyEnv()])
        vec = VecNormalize.load(path, dummy)
        vec.training = False
        vec.norm_reward = False
        print("Loaded VecNormalize:", path)
        return vec

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32).reshape(1, -1)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        return obs

    def _build_world(self):
        for path in [self.args.left_urdf, self.args.right_urdf]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}. Run python 2.py first.")
        p.loadURDF("plane.urdf")
        # Use the same visual court as the training env.
        temp_cfg = TwoSideConfig(gui=False)
        # Inline minimal court construction.
        self._add_court()
        self._add_net()
        self.left_robot = p.loadURDF(self.args.left_urdf, [-2.45, 0, 1.30], [0, 0, 0, 1], useFixedBase=not self.args.free_base)
        self.right_robot = p.loadURDF(self.args.right_urdf, [2.45, 0, 1.30], [0, 0, 1, 0], useFixedBase=not self.args.free_base)
        self.left_map = self._make_joint_map(self.left_robot)
        self.right_map = self._make_joint_map(self.right_robot)
        self.left_racket_link = self._find_link(self.left_robot, ["left_racket", "racket", "left_hand"])
        self.right_racket_link = self._find_link(self.right_robot, ["right_racket", "racket", "right_hand"])
        self.shuttle_id = self._create_shuttle()
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

    def _add_court(self):
        green = [0.08, 0.42, 0.18, 1.0]
        white = [1, 1, 1, 1]
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.15, 1.95, 0.003], rgbaColor=green)
        p.createMultiBody(0, baseVisualShapeIndex=v, basePosition=[0, 0, 0.002])

        def line(pos, size):
            lv = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=white)
            p.createMultiBody(0, baseVisualShapeIndex=lv, basePosition=pos)

        line([0, -1.85, 0.016], [4.0, 0.018, 0.006])
        line([0, 1.85, 0.016], [4.0, 0.018, 0.006])
        line([-4.0, 0, 0.016], [0.018, 1.85, 0.006])
        line([4.0, 0, 0.016], [0.018, 1.85, 0.006])
        line([0, 0, 0.018], [0.018, 1.85, 0.006])
        line([-1.2, 0, 0.018], [0.018, 1.85, 0.006])
        line([1.2, 0, 0.018], [0.018, 1.85, 0.006])
        line([-2.7, 0, 0.018], [0.018, 1.85, 0.006])
        line([2.7, 0, 0.018], [0.018, 1.85, 0.006])

    def _add_net(self):
        h = 1.55
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2], rgbaColor=[0.2, 0.8, 0.2, 0.45])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2])
        p.createMultiBody(0, collision, visual, [0, 0, h / 2])

    def _create_shuttle(self):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.038, length=0.09, rgbaColor=[0.95, 0.95, 1, 1])
        return p.createMultiBody(0.005, col, vis, [0, 0, 1.4])

    def _make_joint_map(self, robot_id):
        return {p.getJointInfo(robot_id, i)[1].decode("utf-8"): i for i in range(p.getNumJoints(robot_id))}

    def _find_link(self, robot_id, names):
        for name in names:
            for i in range(p.getNumJoints(robot_id)):
                if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
                    return i
        raise RuntimeError(f"Link not found: {names}")

    def _reset_neutral(self, robot_id, joint_map):
        neutral = {
            "left_hip": 0.0, "left_knee": 0.20, "left_ankle": -0.06,
            "right_hip": 0.0, "right_knee": 0.20, "right_ankle": -0.06,
            "left_shoulder_pitch": 0.25, "left_elbow": 0.35, "left_wrist": 0.0,
            "right_shoulder_pitch": 0.25, "right_elbow": 0.35, "right_wrist": 0.0,
        }
        for name, value in neutral.items():
            if name not in joint_map:
                continue
            p.resetJointState(robot_id, joint_map[name], value, 0)
            p.setJointMotorControl2(robot_id, joint_map[name], p.POSITION_CONTROL, targetPosition=value, force=180, maxVelocity=6)

    def reset_rally(self):
        self.rally_count += 1
        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -100
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0
        p.resetBasePositionAndOrientation(self.left_robot, [-2.45, 0, 1.30], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.right_robot, [2.45, 0, 1.30], [0, 0, 1, 0])
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)
        receiver = "left" if self.rng.random() < 0.5 else "right"
        self._serve_to(receiver)
        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())
        self._update_rackets()
        print(f"NEW RALLY {self.rally_count}, receiver={receiver}")

    def _serve_to(self, receiver):
        if receiver == "left":
            start = np.array([3.20, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.75)])
            target = np.array([-1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)])
        else:
            start = np.array([-3.20, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.75)])
            target = np.array([1.65, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.35, 1.60)])
        t_flight = self.rng.uniform(0.95, 1.15)
        self.shuttle_pos = start
        self.shuttle_vel = self._ballistic_velocity(start, target, t_flight)
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])
        print("SERVE:", start.round(2), "->", target.round(2))

    def _ballistic_velocity(self, start, target, t_flight):
        g = -9.8
        return np.array([
            (target[0] - start[0]) / t_flight,
            (target[1] - start[1]) / t_flight,
            (target[2] - start[2] - 0.5 * g * t_flight * t_flight) / t_flight,
        ], dtype=float)

    def _receiver_side(self):
        return "left" if self.shuttle_pos[0] < 0 else "right"

    def _estimate_intercept(self, side):
        target_x = -1.65 if side == "left" else 1.65
        vx = float(self.shuttle_vel[0])
        if abs(vx) < 0.20:
            t_hit = 0.75
        else:
            t_hit = (target_x - float(self.shuttle_pos[0])) / vx
        t_hit = float(np.clip(t_hit, 0.06, 1.35))
        g = -9.8
        target = self.shuttle_pos + self.shuttle_vel * t_hit + np.array([0, 0, 0.5 * g * t_hit * t_hit])
        target[0] = target_x
        target[1] = float(np.clip(target[1], -0.80, 0.80))
        target[2] = float(np.clip(target[2], 1.20, 1.72))
        return target, t_hit

    def _make_obs(self, side):
        target, t_hit = self._estimate_intercept(side)
        hist = np.array(list(self.shuttle_hist), dtype=float)
        if side == "left":
            base, _ = p.getBasePositionAndOrientation(self.left_robot)
            base_xy = np.array(base[:2], dtype=float)
            racket_pos = self.left_racket_pos.copy()
            racket_vel = self.left_racket_vel.copy()
            phase = self.last_phase_left
        else:
            base, _ = p.getBasePositionAndOrientation(self.right_robot)
            base_xy = np.array(base[:2], dtype=float)
            racket_pos = self.right_racket_pos.copy()
            racket_vel = self.right_racket_vel.copy()
            phase = self.last_phase_right
            hist = hist.copy(); hist[:, 0] *= -1
            base_xy[0] *= -1
            racket_pos[0] *= -1
            racket_vel[0] *= -1
            target = target.copy(); target[0] *= -1
        return np.concatenate([
            hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
            base_xy * np.array([0.25, 0.5]),
            racket_pos * np.array([0.25, 0.5, 0.5]),
            racket_vel * 0.1,
            target * np.array([0.25, 0.5, 0.5]),
            np.array([t_hit, phase], dtype=float),
        ]).astype(np.float32)

    def _predict_action(self, side):
        obs = self._make_obs(side).reshape(1, -1)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action).reshape(-1).astype(float)

    def _chase_action(self, side, action):
        target, t_hit = self._estimate_intercept(side)
        if side == "left":
            robot = self.left_robot
            desired_x = target[0] - 0.55
        else:
            robot = self.right_robot
            desired_x = target[0] + 0.55
        desired_y = target[1]
        pos, _ = p.getBasePositionAndOrientation(robot)
        pos = np.array(pos, dtype=float)
        t_safe = max(t_hit, 0.12)
        out = action.copy()
        out[0] = np.clip((desired_x - pos[0]) / t_safe / 4.5, -1.0, 1.0)
        out[1] = np.clip((desired_y - pos[1]) / t_safe / 3.6, -1.0, 1.0)
        return out

    def _apply_policy(self, side, action):
        if self.assist_footwork:
            action = self._chase_action(side, action)
        if side == "left":
            self._apply_to_robot(self.left_robot, self.left_map, action, "left", "left")
            self.last_phase_left = float((action[2] + 1) * 0.5)
        else:
            actual = action.copy()
            actual[0] *= -1
            self._apply_to_robot(self.right_robot, self.right_map, actual, "right", "right")
            self.last_phase_right = float((action[2] + 1) * 0.5)

    def _apply_ready(self, side):
        if side == "left":
            self._move_base_toward(self.left_robot, [-2.45, 0], "left")
            targets = self._swing_primitive(0.10, 0.20, "left")
            self._apply_leg_stance(targets, 0.10)
            self._apply_targets(self.left_robot, self.left_map, targets)
        else:
            self._move_base_toward(self.right_robot, [2.45, 0], "right")
            targets = self._swing_primitive(0.10, 0.20, "right")
            self._apply_leg_stance(targets, 0.10)
            self._apply_targets(self.right_robot, self.right_map, targets)

    def _move_base_toward(self, robot, target_xy, side):
        pos, _ = p.getBasePositionAndOrientation(robot)
        pos = np.array(pos, dtype=float)
        pos[:2] += np.clip(np.array(target_xy) - pos[:2], -0.060, 0.060)
        if side == "left":
            pos[0] = np.clip(pos[0], -3.85, -0.50); orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], 0.50, 3.85); orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -1.80, 1.80)
        p.resetBasePositionAndOrientation(robot, pos, orn)

    def _apply_to_robot(self, robot, joint_map, action, side, racket_side):
        pos, _ = p.getBasePositionAndOrientation(robot)
        pos = np.array(pos, dtype=float)
        pos[0] += float(action[0]) * 4.2 * self.control_dt
        pos[1] += float(action[1]) * 3.4 * self.control_dt
        if side == "left":
            pos[0] = np.clip(pos[0], -3.85, -0.50); orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], 0.50, 3.85); orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -1.80, 1.80)
        p.resetBasePositionAndOrientation(robot, pos, orn)
        phase = float((action[2] + 1) * 0.5)
        power = float((action[3] + 1) * 0.5)
        targets = self._swing_primitive(phase, power, racket_side)
        self._apply_leg_stance(targets, phase)
        self._apply_targets(robot, joint_map, targets)

    def _apply_leg_stance(self, targets, phase):
        targets.update({
            "left_hip": 0.05 * np.sin(phase * 2 * np.pi),
            "right_hip": -0.05 * np.sin(phase * 2 * np.pi),
            "left_knee": 0.22,
            "right_knee": 0.22,
            "left_ankle": -0.08,
            "right_ankle": -0.08,
        })

    def _apply_targets(self, robot, joint_map, targets):
        for name, value in targets.items():
            if name in joint_map:
                p.setJointMotorControl2(robot, joint_map[name], p.POSITION_CONTROL, targetPosition=float(value), force=280, maxVelocity=11)

    def _swing_primitive(self, phase, power, racket_side):
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
                "left_elbow": float(np.clip(elbow, 0, 1.8)),
                "left_wrist": float(np.clip(wrist, -0.9, 0.9)),
                "right_shoulder_pitch": float(np.clip(-0.35 * shoulder, -1.8, 1.8)),
                "right_elbow": 0.35,
                "right_wrist": float(np.clip(-0.3 * wrist, -0.9, 0.9)),
            }
        return {
            "right_shoulder_pitch": float(np.clip(-shoulder, -1.8, 1.8)),
            "right_elbow": float(np.clip(elbow, 0, 1.8)),
            "right_wrist": float(np.clip(-wrist, -0.9, 0.9)),
            "left_shoulder_pitch": float(np.clip(0.35 * shoulder, -1.8, 1.8)),
            "left_elbow": 0.35,
            "left_wrist": float(np.clip(0.3 * wrist, -0.9, 0.9)),
        }

    def _update_rackets(self):
        self.left_racket_pos, self.left_racket_vel = self._racket_state("left")
        self.right_racket_pos, self.right_racket_vel = self._racket_state("right")
        self.prev_left_racket = self.left_racket_pos.copy()
        self.prev_right_racket = self.right_racket_pos.copy()

    def _racket_state(self, side):
        if side == "left":
            robot, link, prev = self.left_robot, self.left_racket_link, self.prev_left_racket
        else:
            robot, link, prev = self.right_robot, self.right_racket_link, self.prev_right_racket
        state = p.getLinkState(robot, link, computeLinkVelocity=1)
        pos = np.array(state[0], dtype=float)
        vel = np.zeros(3) if prev is None else (pos - prev) / self.control_dt
        return pos, vel

    def _maybe_hit(self, side):
        if self.step_count - self.last_hit_step < 22:
            return
        if side == "left":
            racket_pos, racket_vel = self.left_racket_pos, self.left_racket_vel
            forward = float(racket_vel[0])
            valid_half = self.shuttle_pos[0] < 0.20
            target = np.array([1.65, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.65)])
        else:
            racket_pos, racket_vel = self.right_racket_pos, self.right_racket_vel
            forward = float(-racket_vel[0])
            valid_half = self.shuttle_pos[0] > -0.20
            target = np.array([-1.65, self.rng.uniform(-0.65, 0.65), self.rng.uniform(1.35, 1.65)])
        dist = np.linalg.norm(racket_pos - self.shuttle_pos)
        if not (valid_half and 0.30 < self.shuttle_pos[2] < 2.05 and dist < self.hit_radius and forward > 0.04):
            return
        t_flight = self.rng.uniform(0.88, 1.12)
        vout = self._ballistic_velocity(self.shuttle_pos, target, t_flight)
        vout[1] += 0.12 * racket_vel[1]
        vout[2] = max(vout[2], 4.70)
        speed = np.linalg.norm(vout)
        if speed > 13.5:
            vout = vout / speed * 13.5
        self.shuttle_vel = vout
        self.hit_count += 1
        self.last_hit_step = self.step_count
        print(f"{side.upper()} HIT #{self.hit_count}: dist={dist:.2f}, forward={forward:.2f}, vout={vout.round(2)}")

    def _update_shuttle(self):
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(self.shuttle_pos, self.shuttle_vel, self.control_dt, None)
        self.shuttle_hist.append(self.shuttle_pos.copy())
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])

    def _done(self):
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

    def run(self):
        try:
            while True:
                receiver = self._receiver_side()
                passive = "right" if receiver == "left" else "left"
                action = self._predict_action(receiver)
                self._apply_policy(receiver, action)
                self._apply_ready(passive)

                for _ in range(max(1, int(round(self.control_dt / self.sim_dt)))):
                    p.stepSimulation()

                self._update_rackets()
                self._maybe_hit(receiver)
                self._update_shuttle()
                self.step_count += 1

                if self._done():
                    time.sleep(0.35)
                    self.reset_rally()
                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            p.disconnect()


class NormDummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play_left", action="store_true")
    parser.add_argument("--play_right", action="store_true")
    parser.add_argument("--play_two", action="store_true")
    parser.add_argument("--model", type=str, default="runs_two_side/badminton_two_side_final.zip")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default="runs_two_side")
    parser.add_argument("--left_urdf", type=str, default="models/humanoid_left.urdf")
    parser.add_argument("--right_urdf", type=str, default="models/humanoid_right.urdf")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hit_radius", type=float, default=1.15)
    parser.add_argument("--free_base", action="store_true")
    parser.add_argument("--policy_footwork", action="store_true", help="Use policy base movement in two-robot demo instead of assisted chase footwork.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    elif args.play_left:
        play_single_side(args, "left")
    elif args.play_right:
        play_single_side(args, "right")
    elif args.play_two:
        player = TwoRobotRallyPlayer(args)
        player.run()
    else:
        print("Use one of: --train, --play_left, --play_right, --play_two")


