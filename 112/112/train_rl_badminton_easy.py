#!/usr/bin/env python3
"""
Easier PPO training for the simplified PyBullet humanoid badminton model.

Why this version exists:
- The previous low-level 14D joint policy is too hard for a small PyBullet demo.
- This version trains a higher-level policy:
    action[0:2] = base x/y movement
    action[2]   = swing phase
    action[3]   = swing power
- The arm motion is a built-in badminton swing primitive, so PPO only learns
  when/where/how strongly to swing.

This matches the practical idea from the paper:
- use target-known training first because it is easier and more stable;
- use staged curriculum: footwork -> swing -> refinement;
- use the racket velocity to compute shuttle outgoing velocity.

Save as:
    train_rl_badminton_easy.py

Install:
    pip install gymnasium stable-baselines3 tensorboard

Train:
    python train_rl_badminton_easy.py --train --timesteps 600000 --n_envs 4

Single-robot demo:
    python train_rl_badminton_easy.py --play_single --model runs_easy\badminton_easy_final.zip

Two-robot shared-policy demo:
    python train_rl_badminton_easy.py --play_two --model runs_easy\badminton_easy_final.zip
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

OBS_DIM = 18 + 2 + 6 + 3 + 1 + 1
# shuttle history 18 + base xy 2 + racket pos/vel 6 + target pos 3 + time_to_hit 1 + last_phase 1
ACTION_DIM = 4


@dataclass
class EasyConfig:
    left_urdf: str = "models/humanoid_left.urdf"
    right_urdf: str = "models/humanoid_right.urdf"
    gui: bool = False
    fixed_base: bool = True
    control_dt: float = 0.02
    sim_dt: float = 1.0 / 240.0
    max_episode_time: float = 2.2
    stage: int = 1
    seed: int = 0


class EasyBadmintonHitEnv(gym.Env):
    """Single left-side robot training environment.

    This environment is intentionally easier than the previous low-level version.
    The robot has a fixed base and a built-in swing primitive. PPO learns:
    - base movement;
    - swing phase timing;
    - swing power.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[EasyConfig] = None):
        super().__init__()
        self.cfg = cfg or EasyConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.client = p.connect(p.GUI if self.cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.cfg.sim_dt, physicsClientId=self.client)
        if self.cfg.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(6.0, 35, -25, [0.0, 0.0, 1.0], physicsClientId=self.client)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.robot_id = -1
        self.opponent_id = -1
        self.shuttle_id = -1
        self.joint_map: Dict[str, int] = {}
        self.racket_link = -1
        self.prev_racket_pos = None
        self.racket_vel = np.zeros(3, dtype=float)
        self.shuttle_hist = deque(maxlen=6)
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.010)

        self.shuttle_pos = np.zeros(3, dtype=float)
        self.shuttle_vel = np.zeros(3, dtype=float)
        self.hit_target = np.zeros(3, dtype=float)
        self.time_to_hit = 1.0
        self.elapsed = 0.0
        self.step_count = 0
        self.has_hit = False
        self.cleared_net = False
        self.last_phase = 0.0
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
        if not os.path.exists(self.cfg.left_urdf):
            raise FileNotFoundError(
                f"Missing {self.cfg.left_urdf}. Run your 2.py once first to generate models/humanoid_left.urdf."
            )

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.cfg.sim_dt, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self._add_court()
        self._add_net()

        self.robot_id = p.loadURDF(
            self.cfg.left_urdf,
            [-2.45, 0.0, 1.30],
            [0, 0, 0, 1],
            useFixedBase=self.cfg.fixed_base,
            physicsClientId=self.client,
        )
        self.joint_map = self._make_joint_map(self.robot_id)
        self.racket_link = self._find_first_link(self.robot_id, ["left_racket", "racket", "left_hand"])

        if self.cfg.gui and os.path.exists(self.cfg.right_urdf):
            self.opponent_id = p.loadURDF(
                self.cfg.right_urdf,
                [2.5, 0.0, 1.30],
                [0, 0, 1, 0],
                useFixedBase=True,
                physicsClientId=self.client,
            )

        self.shuttle_id = self._create_shuttle()

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
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=self.client)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=center, physicsClientId=self.client)

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
            physicsClientId=self.client,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, net_height / 2],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            0,
            collision,
            visual,
            [0, 0, net_height / 2],
            physicsClientId=self.client,
        )

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
            raise RuntimeError(f"URDF missing joints: {missing}")
        return mapping

    def _find_first_link(self, robot_id: int, names: List[str]) -> int:
        for name in names:
            for i in range(p.getNumJoints(robot_id, physicsClientId=self.client)):
                link = p.getJointInfo(robot_id, i, physicsClientId=self.client)[12].decode("utf-8")
                if link == name:
                    return i
        raise RuntimeError(f"Cannot find any link from {names}")

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.elapsed = 0.0
        self.step_count = 0
        self.has_hit = False
        self.cleared_net = False
        self.last_phase = 0.0
        self.last_action[:] = 0.0
        self.prev_racket_pos = None
        self.racket_vel[:] = 0.0

        p.resetBasePositionAndOrientation(self.robot_id, [-2.45, 0.0, 1.30], [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._reset_neutral_pose(self.robot_id, self.joint_map)
        if self.opponent_id >= 0:
            p.resetBasePositionAndOrientation(self.opponent_id, [2.5, 0.0, 1.30], [0, 0, 1, 0], physicsClientId=self.client)

        self._sample_target_known_shuttle()
        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())

        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0], physicsClientId=self.client)
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)
        prev_action = self.last_action.copy()
        self.last_action = action.copy()

        self._apply_high_level_action(action)

        sim_steps = max(1, int(round(self.cfg.control_dt / self.cfg.sim_dt)))
        for _ in range(sim_steps):
            p.stepSimulation(physicsClientId=self.client)

        racket_pos, racket_vel = self._get_racket_state()
        self.racket_vel = racket_vel

        force = self._compute_hit_force(racket_pos, racket_vel)
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(
            self.shuttle_pos,
            self.shuttle_vel,
            self.cfg.control_dt,
            force,
        )
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0], physicsClientId=self.client)
        self.shuttle_hist.append(self.shuttle_pos.copy())

        self.elapsed += self.cfg.control_dt
        self.time_to_hit -= self.cfg.control_dt
        self.step_count += 1

        reward, info = self._compute_reward(racket_pos, racket_vel, action, prev_action)
        terminated = bool(info.get("landed_in", False))
        truncated = self.elapsed >= self.cfg.max_episode_time or self._failure_done()
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Actions and robot
    # ------------------------------------------------------------------

    def _reset_neutral_pose(self, robot_id: int, joint_map: Dict[str, int]):
        neutral = {
            "left_hip": 0.0, "left_knee": 0.15, "left_ankle": -0.05,
            "right_hip": 0.0, "right_knee": 0.15, "right_ankle": -0.05,
            "left_shoulder_pitch": 0.25, "left_elbow": 0.35, "left_wrist": 0.0,
            "right_shoulder_pitch": 0.25, "right_elbow": 0.35, "right_wrist": 0.0,
        }
        for name, val in neutral.items():
            jid = joint_map[name]
            p.resetJointState(robot_id, jid, val, 0.0, physicsClientId=self.client)
            p.setJointMotorControl2(
                robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=val,
                force=160,
                maxVelocity=6.0,
                physicsClientId=self.client,
            )

    def _apply_high_level_action(self, action: np.ndarray):
        base_vx = float(action[0]) * 1.8
        base_vy = float(action[1]) * 1.5
        phase = float((action[2] + 1.0) * 0.5)
        power = float((action[3] + 1.0) * 0.5)
        self.last_phase = phase

        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        pos = np.array(pos, dtype=float)
        pos[0] += base_vx * self.cfg.control_dt
        pos[1] += base_vy * self.cfg.control_dt
        pos[0] = np.clip(pos[0], -3.7, -0.55)
        pos[1] = np.clip(pos[1], -1.75, 1.75)
        p.resetBasePositionAndOrientation(self.robot_id, pos, [0, 0, 0, 1], physicsClientId=self.client)

        targets = self._swing_primitive(phase, power)
        for name, target in targets.items():
            jid = self.joint_map[name]
            p.setJointMotorControl2(
                self.robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=260,
                maxVelocity=10.0,
                physicsClientId=self.client,
            )

        # Keep legs in a simple athletic stance. Base translation is the simplified footwork.
        leg_pose = {
            "left_hip": 0.08 * np.sin(phase * np.pi * 2),
            "right_hip": -0.08 * np.sin(phase * np.pi * 2),
            "left_knee": 0.22,
            "right_knee": 0.22,
            "left_ankle": -0.08,
            "right_ankle": -0.08,
        }
        for name, target in leg_pose.items():
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_map[name],
                p.POSITION_CONTROL,
                targetPosition=target,
                force=140,
                maxVelocity=5.0,
                physicsClientId=self.client,
            )

    def _swing_primitive(self, phase: float, power: float) -> Dict[str, float]:
        """Forehand-like left-arm swing primitive."""
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

        # Counter-swing on the non-racket arm to make it visually closer to the paper's description.
        return {
            "left_shoulder_pitch": float(np.clip(shoulder, -1.8, 1.8)),
            "left_elbow": float(np.clip(elbow, 0.0, 1.8)),
            "left_wrist": float(np.clip(wrist, -0.9, 0.9)),
            "right_shoulder_pitch": float(np.clip(-0.35 * shoulder, -1.8, 1.8)),
            "right_elbow": 0.35,
            "right_wrist": float(np.clip(-0.3 * wrist, -0.9, 0.9)),
        }

    # ------------------------------------------------------------------
    # Shuttle and hit
    # ------------------------------------------------------------------

    def _sample_target_known_shuttle(self):
        # Easy target-known distribution. The hit point is near the racket workspace.
        self.hit_target = np.array(
            [
                self.rng.uniform(-1.95, -1.35),
                self.rng.uniform(-0.55, 0.55),
                self.rng.uniform(1.25, 1.65),
            ],
            dtype=float,
        )
        start = np.array(
            [
                self.rng.uniform(2.6, 3.4),
                self.rng.uniform(-0.75, 0.75),
                self.rng.uniform(1.35, 1.75),
            ],
            dtype=float,
        )
        self.time_to_hit = float(self.rng.uniform(0.85, 1.20))
        g = -9.8
        vx = (self.hit_target[0] - start[0]) / self.time_to_hit
        vy = (self.hit_target[1] - start[1]) / self.time_to_hit
        vz = (self.hit_target[2] - start[2] - 0.5 * g * self.time_to_hit**2) / self.time_to_hit
        self.shuttle_pos = start
        self.shuttle_vel = np.array([vx, vy, vz], dtype=float)

    def _get_racket_state(self) -> Tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.robot_id, self.racket_link, computeLinkVelocity=1, physicsClientId=self.client)
        pos = np.array(state[0], dtype=float)
        if self.prev_racket_pos is None:
            vel = np.zeros(3, dtype=float)
        else:
            vel = (pos - self.prev_racket_pos) / self.cfg.control_dt
        self.prev_racket_pos = pos.copy()
        return pos, vel

    def _compute_hit_force(self, racket_pos: np.ndarray, racket_vel: np.ndarray):
        if self.has_hit:
            return None

        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        forward_speed = float(racket_vel[0])
        good_height = 0.45 < self.shuttle_pos[2] < 1.85
        near_hit_time = abs(self.time_to_hit) < 0.35

        # Easier than the previous version: close racket + forward swing is enough.
        if dist < 0.70 and forward_speed > 0.18 and good_height and near_hit_time:
            self.has_hit = True
            n = np.array([1.0, 0.0, 0.0], dtype=float)
            vin = self.shuttle_vel.copy()
            v_racket_n = np.dot(racket_vel, n) * n
            v_shuttle_n = np.dot(vin, n) * n
            vout = vin - 2.0 * v_shuttle_n + 2.0 * v_racket_n
            vout[0] = max(vout[0], 5.0)
            vout[2] = max(vout[2], 5.0)
            speed = np.linalg.norm(vout)
            if speed > 14.0:
                vout = vout / speed * 14.0
            mass = 0.005
            return mass * (vout - self.shuttle_vel) / self.cfg.control_dt
        return None

    # ------------------------------------------------------------------
    # Observation and reward
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        racket_pos, racket_vel = self._get_racket_state()
        hist = np.array(list(self.shuttle_hist), dtype=float)
        if hist.shape[0] < 6:
            hist = np.tile(self.shuttle_pos, (6, 1))
        obs = np.concatenate(
            [
                hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
                np.array(base_pos[:2]) * np.array([0.25, 0.5]),
                racket_pos * np.array([0.25, 0.5, 0.5]),
                racket_vel * 0.1,
                self.hit_target * np.array([0.25, 0.5, 0.5]),
                np.array([self.time_to_hit], dtype=float),
                np.array([self.last_phase], dtype=float),
            ]
        ).astype(np.float32)
        return obs

    def _compute_reward(self, racket_pos, racket_vel, action, prev_action):
        stage = self.cfg.stage
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_pos = np.array(base_pos, dtype=float)

        base_target = np.array([self.hit_target[0] - 0.55, self.hit_target[1]])
        base_dist = np.linalg.norm(base_pos[:2] - base_target)
        racket_dist_target = np.linalg.norm(racket_pos - self.hit_target)
        racket_dist_shuttle = np.linalg.norm(racket_pos - self.shuttle_pos)
        timing = np.exp(-abs(self.time_to_hit) / 0.22)
        near_shuttle = np.exp(-4.0 * racket_dist_shuttle)
        forward = max(0.0, racket_vel[0])

        r_foot = np.exp(-1.7 * base_dist)
        r_target = np.exp(-4.0 * racket_dist_target)
        r_near = np.exp(-5.0 * racket_dist_shuttle)
        r_swing = timing * near_shuttle * forward
        r_phase = timing * np.exp(-8.0 * abs(self.last_phase - 0.50))

        reward = 0.0
        if stage == 1:
            reward += 3.0 * r_foot
            reward += 1.0 * r_target
        elif stage == 2:
            reward += 1.5 * r_foot
            reward += 3.0 * r_target
            reward += 1.0 * r_phase
            reward += 0.5 * r_swing
        else:
            reward += 1.0 * r_foot
            reward += 2.5 * r_target
            reward += 1.5 * r_near
            reward += 1.2 * r_phase
            reward += 0.8 * r_swing

        reward -= 0.01 * float(np.mean(action**2))
        reward -= 0.02 * float(np.mean((action - prev_action) ** 2))

        info = {
            "hit": self.has_hit,
            "stage": stage,
            "racket_dist": racket_dist_shuttle,
            "target_dist": racket_dist_target,
        }

        if self.has_hit:
            reward += 80.0
            info["hit_success"] = True

        if self.has_hit and not self.cleared_net and self.shuttle_pos[0] > 0.05:
            if self.shuttle_pos[2] > 1.65:
                self.cleared_net = True
                reward += 60.0
                info["cleared_net"] = True
            else:
                reward -= 20.0
                info["net_fail"] = True

        if self.has_hit and self.shuttle_pos[2] < 0.08:
            in_bounds = 0.2 < self.shuttle_pos[0] < 4.0 and abs(self.shuttle_pos[1]) < 1.85
            if in_bounds:
                reward += 80.0
                info["landed_in"] = True
            else:
                reward -= 12.0
                info["landed_out"] = True

        if not self.has_hit and self.time_to_hit < -0.35:
            reward -= 10.0
            info["missed_window"] = True

        return float(reward), info

    def _failure_done(self) -> bool:
        if not self.has_hit and self.time_to_hit < -0.55:
            return True
        if self.shuttle_pos[2] < 0.05:
            return True
        if abs(self.shuttle_pos[0]) > 4.5 or abs(self.shuttle_pos[1]) > 2.5:
            return True
        return False


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------


def make_env(rank: int, stage: int, gui: bool, seed: int):
    def _init():
        cfg = EasyConfig(gui=gui, stage=stage, seed=seed + rank)
        return EasyBadmintonHitEnv(cfg)
    return _init


def train(args):
    os.makedirs(args.run_dir, exist_ok=True)
    vec_cls = DummyVecEnv if args.n_envs == 1 else SubprocVecEnv
    env = vec_cls([make_env(i, stage=1, gui=False, seed=args.seed) for i in range(args.n_envs)])
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
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]), activation_fn=nn.ELU),
        tensorboard_log=os.path.join(args.run_dir, "tb"),
        verbose=1,
        seed=args.seed,
    )

    ckpt = CheckpointCallback(
        save_freq=max(args.n_steps * args.n_envs, 10000),
        save_path=args.run_dir,
        name_prefix="easy_ckpt",
    )

    stage_steps = args.timesteps // 3
    for stage in [1, 2, 3]:
        print(f"\n========== EASY TRAINING STAGE {stage} ==========")
        env.env_method("set_stage", stage)
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=(stage == 1), callback=ckpt, progress_bar=True)
        model.save(os.path.join(args.run_dir, f"badminton_easy_s{stage}"))
        env.save(os.path.join(args.run_dir, f"vecnormalize_easy_s{stage}.pkl"))

    model.save(os.path.join(args.run_dir, "badminton_easy_final"))
    env.save(os.path.join(args.run_dir, "vecnormalize_easy_final.pkl"))
    env.close()
    print("Easy training complete.")


def _load_vec_env_for_play(model_path: str, vecnorm_path: Optional[str], seed: int, gui: bool):
    env = DummyVecEnv([make_env(0, stage=3, gui=gui, seed=seed)])
    if vecnorm_path is None:
        vecnorm_path = os.path.join(os.path.dirname(model_path), "vecnormalize_easy_final.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize:", vecnorm_path)
    else:
        print("Warning: vecnormalize file not found:", vecnorm_path)
    return env


def play_single(args):
    env = _load_vec_env_for_play(args.model, args.vecnorm, args.seed, gui=True)
    model = PPO.load(args.model, env=env)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(0.02)
        if bool(done[0]):
            print("episode end", info[0])
            obs = env.reset()


# ----------------------------------------------------------------------
# Two-robot shared-policy demo for the easy model
# ----------------------------------------------------------------------


class TwoRobotEasyDemo:
    def __init__(self, model_path: str, vecnorm_path: Optional[str], seed: int = 0):
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path or os.path.join(os.path.dirname(model_path), "vecnormalize_easy_final.pkl")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.control_dt = 0.02
        self.sim_dt = 1.0 / 240.0

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(6.0, 35, -25, [0.0, 0.0, 1.0])

        # Load model and vecnorm through a dummy training-compatible env.
        self.vec_env = _load_vec_env_for_play(model_path, self.vecnorm_path, seed, gui=False)
        self.model = PPO.load(model_path, env=self.vec_env)

        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.010)
        self.shuttle_hist = deque(maxlen=6)
        self.shuttle_pos = np.zeros(3)
        self.shuttle_vel = np.zeros(3)
        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -80
        self.last_hitter = None

        self.left_robot = -1
        self.right_robot = -1
        self.left_map = {}
        self.right_map = {}
        self.left_racket = -1
        self.right_racket = -1
        self.prev_left_racket = None
        self.prev_right_racket = None
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0
        self.shuttle_id = -1

        self._build_world()
        self.reset_rally()

    def _build_world(self):
        if not os.path.exists("models/humanoid_left.urdf") or not os.path.exists("models/humanoid_right.urdf"):
            raise FileNotFoundError("Run 2.py once first to generate models/humanoid_left.urdf and humanoid_right.urdf")
        p.loadURDF("plane.urdf")
        # Minimal court and net.
        self._add_court()
        self._add_net()
        self.left_robot = p.loadURDF("models/humanoid_left.urdf", [-2.45, 0.0, 1.30], [0, 0, 0, 1], useFixedBase=True)
        self.right_robot = p.loadURDF("models/humanoid_right.urdf", [2.45, 0.0, 1.30], [0, 0, 1, 0], useFixedBase=True)
        self.left_map = self._make_joint_map(self.left_robot)
        self.right_map = self._make_joint_map(self.right_robot)
        self.left_racket = self._find_link(self.left_robot, ["left_racket", "left_hand"])
        self.right_racket = self._find_link(self.right_robot, ["right_racket", "right_hand"])
        self.shuttle_id = self._create_shuttle()
        self._reset_pose(self.left_robot, self.left_map)
        self._reset_pose(self.right_robot, self.right_map)

    def _add_court(self):
        green = [0.08, 0.42, 0.18, 1]
        white = [1, 1, 1, 1]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.1, 1.95, 0.003], rgbaColor=green)
        p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0.002])
        def line(pos, size):
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=white)
            p.createMultiBody(0, baseVisualShapeIndex=v, basePosition=pos)
        line([0, -1.85, 0.01], [4.0, 0.015, 0.004])
        line([0, 1.85, 0.01], [4.0, 0.015, 0.004])
        line([-4.0, 0, 0.01], [0.015, 1.85, 0.004])
        line([4.0, 0, 0.01], [0.015, 1.85, 0.004])
        line([0, 0, 0.012], [0.015, 1.85, 0.004])

    def _add_net(self):
        h = 1.55
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2], rgbaColor=[0.2, 0.8, 0.2, 0.45])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2])
        p.createMultiBody(0, col, vis, [0, 0, h / 2])

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
        raise RuntimeError(f"link not found: {names}")

    def _reset_pose(self, robot_id, joint_map):
        neutral = {
            "left_hip": 0.0, "left_knee": 0.15, "left_ankle": -0.05,
            "right_hip": 0.0, "right_knee": 0.15, "right_ankle": -0.05,
            "left_shoulder_pitch": 0.25, "left_elbow": 0.35, "left_wrist": 0.0,
            "right_shoulder_pitch": 0.25, "right_elbow": 0.35, "right_wrist": 0.0,
        }
        for name, val in neutral.items():
            jid = joint_map[name]
            p.resetJointState(robot_id, jid, val, 0)
            p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=val, force=160, maxVelocity=6)

    def reset_rally(self):
        self.hit_count = 0
        self.step_count = 0
        self.last_hit_step = -80
        self.last_hitter = None
        self.prev_left_racket = None
        self.prev_right_racket = None
        p.resetBasePositionAndOrientation(self.left_robot, [-2.45, 0.0, 1.30], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.right_robot, [2.45, 0.0, 1.30], [0, 0, 1, 0])
        self._reset_pose(self.left_robot, self.left_map)
        self._reset_pose(self.right_robot, self.right_map)
        self._serve()
        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())

    def _serve(self):
        side = self.rng.choice([-1, 1])
        if side > 0:
            start = np.array([3.2, self.rng.uniform(-0.6, 0.6), self.rng.uniform(1.35, 1.65)])
            target = np.array([-1.7, self.rng.uniform(-0.5, 0.5), self.rng.uniform(1.3, 1.6)])
        else:
            start = np.array([-3.2, self.rng.uniform(-0.6, 0.6), self.rng.uniform(1.35, 1.65)])
            target = np.array([1.7, self.rng.uniform(-0.5, 0.5), self.rng.uniform(1.3, 1.6)])
        T = self.rng.uniform(0.9, 1.15)
        g = -9.8
        vel = np.array([(target[0] - start[0]) / T, (target[1] - start[1]) / T, (target[2] - start[2] - 0.5 * g * T**2) / T])
        self.shuttle_pos = start
        self.shuttle_vel = vel
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])
        print("SERVE", self.shuttle_pos.round(2), self.shuttle_vel.round(2))

    def _racket_state(self, robot_id, link, prev):
        st = p.getLinkState(robot_id, link, computeLinkVelocity=1)
        pos = np.array(st[0])
        vel = np.zeros(3) if prev is None else (pos - prev) / self.control_dt
        return pos, vel

    def _make_obs(self, side: str):
        if side == "left":
            robot = self.left_robot
            racket, vel = self._racket_state(self.left_robot, self.left_racket, self.prev_left_racket)
            base, _ = p.getBasePositionAndOrientation(robot)
            hist = np.array(list(self.shuttle_hist))
            target = np.array([-1.7, self.shuttle_pos[1], 1.45])
            phase = self.last_phase_left
        else:
            robot = self.right_robot
            racket, vel = self._racket_state(self.right_robot, self.right_racket, self.prev_right_racket)
            base, _ = p.getBasePositionAndOrientation(robot)
            hist = np.array(list(self.shuttle_hist))
            hist[:, 0] *= -1
            base = list(base)
            base[0] *= -1
            racket[0] *= -1
            vel[0] *= -1
            target = np.array([-1.7, self.shuttle_pos[1], 1.45])
            phase = self.last_phase_right
        time_to_hit = min(1.0, max(-0.5, abs(self.shuttle_pos[0]) / max(0.5, abs(self.shuttle_vel[0]))))
        obs = np.concatenate([
            hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
            np.array(base[:2]) * np.array([0.25, 0.5]),
            racket * np.array([0.25, 0.5, 0.5]),
            vel * 0.1,
            target * np.array([0.25, 0.5, 0.5]),
            np.array([time_to_hit, phase]),
        ]).astype(np.float32)
        obs = obs.reshape(1, -1)
        if self.vec_env is not None:
            obs = self.vec_env.normalize_obs(obs)
        return obs

    def _apply_policy_action(self, side: str):
        obs = self._make_obs(side)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action).reshape(-1)
        if side == "left":
            self._apply_action(self.left_robot, self.left_map, action, side="left")
            self.last_phase_left = (action[2] + 1) * 0.5
        else:
            action = action.copy()
            action[0] *= -1
            self._apply_action(self.right_robot, self.right_map, action, side="right")
            self.last_phase_right = (action[2] + 1) * 0.5

    def _swing_primitive(self, phase, power, racket_side):
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
                "left_shoulder_pitch": shoulder, "left_elbow": elbow, "left_wrist": wrist,
                "right_shoulder_pitch": -0.35 * shoulder, "right_elbow": 0.35, "right_wrist": -0.3 * wrist,
            }
        return {
            "right_shoulder_pitch": -shoulder, "right_elbow": elbow, "right_wrist": -wrist,
            "left_shoulder_pitch": 0.35 * shoulder, "left_elbow": 0.35, "left_wrist": 0.3 * wrist,
        }

    def _apply_action(self, robot, joint_map, action, side: str):
        pos, _ = p.getBasePositionAndOrientation(robot)
        pos = np.array(pos)
        pos[0] += action[0] * 1.8 * self.control_dt
        pos[1] += action[1] * 1.5 * self.control_dt
        if side == "left":
            pos[0] = np.clip(pos[0], -3.7, -0.55)
            orn = [0, 0, 0, 1]
            racket_side = "left"
        else:
            pos[0] = np.clip(pos[0], 0.55, 3.7)
            orn = [0, 0, 1, 0]
            racket_side = "right"
        pos[1] = np.clip(pos[1], -1.75, 1.75)
        p.resetBasePositionAndOrientation(robot, pos, orn)
        phase = (action[2] + 1) * 0.5
        power = (action[3] + 1) * 0.5
        targets = self._swing_primitive(phase, power, racket_side)
        targets.update({"left_knee": 0.22, "right_knee": 0.22, "left_ankle": -0.08, "right_ankle": -0.08})
        for name, val in targets.items():
            if name in joint_map:
                p.setJointMotorControl2(robot, joint_map[name], p.POSITION_CONTROL, targetPosition=float(val), force=260, maxVelocity=10)

    def _maybe_hit(self, side):
        if self.step_count - self.last_hit_step < 25:
            return
        if side == "left":
            racket, vel = self._racket_state(self.left_robot, self.left_racket, self.prev_left_racket)
            normal = np.array([1, 0, 0])
            forward = vel[0]
        else:
            racket, vel = self._racket_state(self.right_robot, self.right_racket, self.prev_right_racket)
            normal = np.array([-1, 0, 0])
            forward = -vel[0]
        dist = np.linalg.norm(racket - self.shuttle_pos)
        if dist < 0.85 and forward > 0.10 and 0.35 < self.shuttle_pos[2] < 1.95:
            vin = self.shuttle_vel.copy()
            vout = vin - 2 * np.dot(vin, normal) * normal + 2 * np.dot(vel, normal) * normal
            if side == "left":
                vout[0] = max(vout[0], 5.0)
            else:
                vout[0] = min(vout[0], -5.0)
            vout[2] = max(vout[2], 4.8)
            sp = np.linalg.norm(vout)
            if sp > 14:
                vout = vout / sp * 14
            self.shuttle_vel = vout
            self.hit_count += 1
            self.last_hit_step = self.step_count
            self.last_hitter = side
            print(f"{side.upper()} HIT {self.hit_count}: dist={dist:.2f}, vel={vel.round(2)}, vout={vout.round(2)}")

    def run(self):
        try:
            while True:
                side = "left" if self.shuttle_pos[0] < 0 else "right"
                self._apply_policy_action("left")
                self._apply_policy_action("right")
                for _ in range(int(round(self.control_dt / self.sim_dt))):
                    p.stepSimulation()
                left_racket, _ = self._racket_state(self.left_robot, self.left_racket, self.prev_left_racket)
                right_racket, _ = self._racket_state(self.right_robot, self.right_racket, self.prev_right_racket)
                self.prev_left_racket = left_racket.copy()
                self.prev_right_racket = right_racket.copy()
                self._maybe_hit(side)
                self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(self.shuttle_pos, self.shuttle_vel, self.control_dt, None)
                self.shuttle_hist.append(self.shuttle_pos.copy())
                p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
                p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])
                self.step_count += 1
                if self.shuttle_pos[2] < 0.06 or abs(self.shuttle_pos[0]) > 4.4 or abs(self.shuttle_pos[1]) > 2.5:
                    print("RALLY END, hits=", self.hit_count)
                    time.sleep(0.4)
                    self.reset_rally()
                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            p.disconnect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play_single", action="store_true")
    parser.add_argument("--play_two", action="store_true")
    parser.add_argument("--model", type=str, default="runs_easy/badminton_easy_final.zip")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default="runs_easy")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    elif args.play_single:
        play_single(args)
    elif args.play_two:
        demo = TwoRobotEasyDemo(args.model, args.vecnorm, args.seed)
        demo.run()
    else:
        print("Use one of: --train, --play_single, --play_two")
