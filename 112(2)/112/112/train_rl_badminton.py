#!/usr/bin/env python3
"""
PyBullet badminton RL training script for the simplified humanoid model.

This script trains a simplified single-humanoid badminton hitting policy with PPO.
It is inspired by the paper's setup:
- PPO with gamma=0.99, GAE lambda=0.95, clip=0.2, entropy=0.01.
- Control dt = 0.02 s, approximately 50 Hz.
- Observation includes a sliding window of 6 shuttle positions.
- Three-stage curriculum: footwork -> swing -> refinement.

Important:
- This is not a full Isaac Gym 4096-env whole-body reproduction.
- It is a practical PyBullet training prototype for your current simplified model.
- It assumes models/humanoid_left.urdf already exists. Run your visual demo once first,
  or place the generated URDF at models/humanoid_left.urdf.

Install:
    pip install gymnasium stable-baselines3 numpy tensorboard

Train:
    python train_rl_badminton.py --train --timesteps 300000

Play:
    python train_rl_badminton.py --play --model runs/badminton_s3.zip
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

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError as exc:
    raise ImportError(
        "Missing RL dependencies. Install them with:\n"
        "    pip install gymnasium stable-baselines3 tensorboard\n"
    ) from exc

from physics_shuttle import ShuttlePhysics


@dataclass
class EnvConfig:
    urdf_path: str = "models/humanoid_left.urdf"
    gui: bool = False
    control_dt: float = 0.02
    sim_dt: float = 1.0 / 240.0
    max_episode_time: float = 2.4
    court_x_min: float = -4.0
    court_x_max: float = 4.0
    court_y_min: float = -1.85
    court_y_max: float = 1.85
    net_height: float = 1.55
    robot_start: Tuple[float, float, float] = (-2.45, 0.0, 1.30)
    robot_orn: Tuple[float, float, float, float] = (0, 0, 0, 1)
    racket_side: str = "left"
    stage: int = 1
    seed: int = 0


class BadmintonHitEnv(gym.Env):
    """Single-robot shuttle hitting environment.

    Action space, 14 dimensions:
        [base_vx, base_vy, 12 joint commands]

    The simplified visual model does not have dynamically stable walking, so the
    base x-y motion is treated as a high-level footwork command, matching the
    current demo's resetBasePositionAndOrientation style. The 12 joint commands
    control:
        left/right hip, knee, ankle, shoulder, elbow, wrist.

    Observation:
        - 6-frame shuttle position history, sampled at control_dt = 0.02 s.
        - robot base xy
        - racket position and velocity
        - joint positions and velocities
        - current time fraction
    """

    metadata = {"render_modes": ["human", "direct"]}

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

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.client = p.connect(p.GUI if self.cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.cfg.sim_dt, physicsClientId=self.client)

        # Action: base vx, base vy, and 12 normalized joint targets.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

        # Observation dimensions:
        # shuttle history: 6*3=18
        # base xy: 2
        # racket pos + vel: 6
        # joint pos + vel: 24
        # time fraction: 1
        obs_dim = 18 + 2 + 6 + 24 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.robot_id = -1
        self.shuttle_id = -1
        self.joint_map: Dict[str, int] = {}
        self.racket_link = -1
        self.prev_racket_pos = None
        self.racket_vel = np.zeros(3, dtype=float)
        self.shuttle_pos_hist = deque(maxlen=6)

        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.012)
        self.shuttle_pos = np.zeros(3, dtype=float)
        self.shuttle_vel = np.zeros(3, dtype=float)

        self.elapsed = 0.0
        self.step_count = 0
        self.has_hit = False
        self.crossed_net_after_hit = False
        self.landed_after_hit = False
        self.last_action = np.zeros(14, dtype=float)

        self._build_world()

    def close(self):
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1

    def set_stage(self, stage: int):
        self.cfg.stage = int(stage)

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _build_world(self):
        if not os.path.exists(self.cfg.urdf_path):
            raise FileNotFoundError(
                f"Cannot find {self.cfg.urdf_path}. Run your visual demo once first so it generates the URDF, "
                "or put humanoid_left.urdf under models/."
            )

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.cfg.sim_dt, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self._add_court()
        self._add_net()

        self.robot_id = p.loadURDF(
            self.cfg.urdf_path,
            self.cfg.robot_start,
            self.cfg.robot_orn,
            useFixedBase=False,
            physicsClientId=self.client,
        )
        self.joint_map = self._make_joint_map(self.robot_id)
        self.racket_link = self._find_first_link(self.robot_id, ["left_racket", "racket", "left_hand"])

        self.shuttle_id = self._create_shuttle()

    def _add_court(self):
        # Same geometry idea as the visual demo: scaled badminton court.
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
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, self.cfg.net_height / 2],
            rgbaColor=[0.2, 0.8, 0.2, 0.45],
            physicsClientId=self.client,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 2.0, self.cfg.net_height / 2],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, self.cfg.net_height / 2],
            physicsClientId=self.client,
        )

    def _create_shuttle(self):
        feather_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09, physicsClientId=self.client)
        feather_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.038,
            length=0.09,
            rgbaColor=[0.95, 0.95, 1, 1],
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0.005,
            baseCollisionShapeIndex=feather_col,
            baseVisualShapeIndex=feather_vis,
            basePosition=[3.5, 0, 1.6],
            physicsClientId=self.client,
        )

    def _make_joint_map(self, robot_id: int) -> Dict[str, int]:
        mapping = {}
        for i in range(p.getNumJoints(robot_id, physicsClientId=self.client)):
            name = p.getJointInfo(robot_id, i, physicsClientId=self.client)[1].decode("utf-8")
            mapping[name] = i
        missing = [name for name in self.JOINT_NAMES if name not in mapping]
        if missing:
            raise RuntimeError(f"URDF missing required joints: {missing}")
        return mapping

    def _get_link_index(self, robot_id, link_name):
        for i in range(p.getNumJoints(robot_id, physicsClientId=self.client)):
            link = p.getJointInfo(robot_id, i, physicsClientId=self.client)[12].decode("utf-8")
            if link == link_name:
                return i
        return -1

    def _find_first_link(self, robot_id, names: List[str]) -> int:
        for name in names:
            idx = self._get_link_index(robot_id, name)
            if idx >= 0:
                return idx
        raise RuntimeError(f"Could not find any link from {names}")

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
        self.crossed_net_after_hit = False
        self.landed_after_hit = False
        self.last_action[:] = 0.0
        self.prev_racket_pos = None
        self.racket_vel[:] = 0.0

        p.resetBasePositionAndOrientation(self.robot_id, self.cfg.robot_start, self.cfg.robot_orn, physicsClientId=self.client)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._reset_joints_neutral()

        self._sample_incoming_shuttle()
        self.shuttle_pos_hist.clear()
        for _ in range(6):
            self.shuttle_pos_hist.append(self.shuttle_pos.copy())

        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0], physicsClientId=self.client)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.copy()

        self._apply_base_action(action[:2])
        self._apply_joint_action(action[2:])

        sim_steps = max(1, int(round(self.cfg.control_dt / self.cfg.sim_dt)))
        for _ in range(sim_steps):
            p.stepSimulation(physicsClientId=self.client)

        # Racket state after action.
        racket_pos, racket_vel = self._get_racket_state()
        self.racket_vel = racket_vel

        # Shuttle dynamics are integrated explicitly, as in the visual demo.
        force = self._compute_hit_force(racket_pos, racket_vel)
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(
            self.shuttle_pos,
            self.shuttle_vel,
            self.cfg.control_dt,
            force,
        )
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0], physicsClientId=self.client)

        self.shuttle_pos_hist.append(self.shuttle_pos.copy())
        self.elapsed += self.cfg.control_dt
        self.step_count += 1

        reward, info = self._compute_reward(racket_pos, racket_vel, action)
        terminated = self._is_success_done(info)
        truncated = self.elapsed >= self.cfg.max_episode_time or self._is_failure_done()
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Dynamics and controls
    # ------------------------------------------------------------------

    def _reset_joints_neutral(self):
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
            jid = self.joint_map[name]
            p.resetJointState(self.robot_id, jid, value, 0.0, physicsClientId=self.client)
            p.setJointMotorControl2(
                self.robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=value,
                force=120,
                maxVelocity=4.0,
                physicsClientId=self.client,
            )

    def _apply_base_action(self, base_action):
        base_vx = float(base_action[0]) * 1.7
        base_vy = float(base_action[1]) * 1.4
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        pos = np.array(pos, dtype=float)
        pos[0] += base_vx * self.cfg.control_dt
        pos[1] += base_vy * self.cfg.control_dt
        pos[0] = np.clip(pos[0], -3.8, -0.45)
        pos[1] = np.clip(pos[1], self.cfg.court_y_min, self.cfg.court_y_max)
        p.resetBasePositionAndOrientation(self.robot_id, pos, self.cfg.robot_orn, physicsClientId=self.client)

    def _apply_joint_action(self, joint_action):
        for value, name in zip(joint_action, self.JOINT_NAMES):
            low, high = self.JOINT_LIMITS[name]
            target = low + (float(value) + 1.0) * 0.5 * (high - low)
            jid = self.joint_map[name]
            p.setJointMotorControl2(
                self.robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=180,
                maxVelocity=6.0,
                physicsClientId=self.client,
            )

    def _sample_incoming_shuttle(self):
        # Scaled version of the paper's model-based shuttle trajectory generation:
        # shuttle starts on opponent side and flies toward the robot side.
        self.shuttle_pos = np.array(
            [
                self.rng.uniform(3.0, 3.8),
                self.rng.uniform(-1.2, 1.2),
                self.rng.uniform(1.25, 1.85),
            ],
            dtype=float,
        )
        self.shuttle_vel = np.array(
            [
                self.rng.uniform(-5.8, -4.0),
                self.rng.uniform(-0.8, 0.8),
                self.rng.uniform(1.8, 4.2),
            ],
            dtype=float,
        )

    def _get_racket_state(self):
        state = p.getLinkState(self.robot_id, self.racket_link, computeLinkVelocity=1, physicsClientId=self.client)
        pos = np.array(state[0], dtype=float)
        if self.prev_racket_pos is None:
            vel = np.zeros(3, dtype=float)
        else:
            vel = (pos - self.prev_racket_pos) / self.cfg.control_dt
        self.prev_racket_pos = pos.copy()
        return pos, vel

    def _compute_hit_force(self, racket_pos, racket_vel):
        if self.has_hit:
            return None

        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        good_height = 0.45 < self.shuttle_pos[2] < 1.85
        forward_speed = float(racket_vel[0])
        swinging_forward = forward_speed > 0.25

        if dist < 0.45 and good_height and swinging_forward:
            self.has_hit = True

            # Paper-inspired nearly elastic racket interaction:
            # vout = vin - 2(vin·n)n + 2(vracket·n)n
            # We approximate the racket normal as +x for the left-side robot.
            n = np.array([1.0, 0.0, 0.0], dtype=float)
            vin = self.shuttle_vel.copy()
            v_racket_n = np.dot(racket_vel, n) * n
            v_shuttle_n = np.dot(vin, n) * n
            vout = vin - 2.0 * v_shuttle_n + 2.0 * v_racket_n

            # Add upward clearance and clamp speed to avoid unstable explosions.
            vout[2] = max(vout[2], 4.5)
            speed = np.linalg.norm(vout)
            if speed > 14.0:
                vout = vout / speed * 14.0

            mass = 0.005
            impulse_dt = self.cfg.control_dt
            return mass * (vout - self.shuttle_vel) / impulse_dt

        return None

    # ------------------------------------------------------------------
    # Reward and observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_xy = np.array(base_pos[:2], dtype=float)
        racket_pos, racket_vel = self._get_racket_state()

        joint_pos = []
        joint_vel = []
        for name in self.JOINT_NAMES:
            jid = self.joint_map[name]
            js = p.getJointState(self.robot_id, jid, physicsClientId=self.client)
            low, high = self.JOINT_LIMITS[name]
            # Normalize joint position to roughly [-1, 1].
            q_norm = 2.0 * (js[0] - low) / (high - low) - 1.0
            joint_pos.append(q_norm)
            joint_vel.append(js[1] * 0.1)

        shuttle_hist = np.array(list(self.shuttle_pos_hist), dtype=float).reshape(-1)
        time_frac = np.array([self.elapsed / self.cfg.max_episode_time], dtype=float)

        obs = np.concatenate(
            [
                shuttle_hist * np.array([0.25, 0.5, 0.5] * 6),
                base_xy * np.array([0.25, 0.5]),
                racket_pos * np.array([0.25, 0.5, 0.5]),
                racket_vel * 0.1,
                np.array(joint_pos, dtype=float),
                np.array(joint_vel, dtype=float),
                time_frac,
            ]
        ).astype(np.float32)
        return obs

    def _predict_intercept_point(self):
        # Privileged target estimate for reward only, not a separate actor input.
        pos = self.shuttle_pos.copy()
        vel = self.shuttle_vel.copy()
        dt = 0.04
        for _ in range(40):
            pos = pos + vel * dt
            vel[2] += -9.8 * dt
            if pos[0] < -0.7 and 1.25 < pos[2] < 1.75:
                return pos
        return np.array([-2.2, self.shuttle_pos[1], 1.45], dtype=float)

    def _compute_reward(self, racket_pos, racket_vel, action):
        stage = self.cfg.stage
        reward = 0.0
        info = {
            "hit": self.has_hit,
            "stage": stage,
        }

        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_pos = np.array(base_pos, dtype=float)
        target = self._predict_intercept_point()

        # Footwork reward: move base near the predicted interception y/x band.
        base_target = np.array([target[0], target[1], base_pos[2]])
        base_dist = np.linalg.norm((base_pos - base_target)[:2])
        r_approach = np.exp(-1.4 * base_dist)

        # Racket-target reward.
        racket_dist = np.linalg.norm(racket_pos - target)
        r_racket = np.exp(-3.0 * racket_dist)

        # Swing reward: encourage +x racket velocity near the shuttle.
        shuttle_dist = np.linalg.norm(racket_pos - self.shuttle_pos)
        near_shuttle = np.exp(-4.0 * shuttle_dist)
        r_swing_speed = near_shuttle * max(0.0, racket_vel[0])

        # Regularization.
        action_rate = np.mean((action - self.last_action) ** 2)
        action_mag = np.mean(action ** 2)

        # Stage curriculum.
        if stage == 1:
            # Footwork acquisition.
            reward += 2.0 * r_approach
            reward += 0.5 * r_racket
            reward -= 0.02 * action_mag
            reward -= 0.01 * action_rate
        elif stage == 2:
            # Swing emergence.
            reward += 1.0 * r_approach
            reward += 2.0 * r_racket
            reward += 0.25 * r_swing_speed
            reward -= 0.02 * action_mag
            reward -= 0.015 * action_rate
        else:
            # Task-focused refinement.
            reward += 0.5 * r_approach
            reward += 1.5 * r_racket
            reward += 0.30 * r_swing_speed
            reward -= 0.03 * action_mag
            reward -= 0.02 * action_rate

        # Sparse task rewards.
        if self.has_hit:
            reward += 25.0
            info["hit_success"] = True

        # After hit: reward clearing net and staying in court.
        if self.has_hit and not self.crossed_net_after_hit and self.shuttle_pos[0] > 0.05:
            if self.shuttle_pos[2] > self.cfg.net_height + 0.10:
                self.crossed_net_after_hit = True
                reward += 20.0
                info["cleared_net"] = True
            else:
                reward -= 10.0
                info["net_fail"] = True

        if self.has_hit and self.shuttle_pos[2] < 0.08:
            self.landed_after_hit = True
            in_bounds = 0.0 < self.shuttle_pos[0] < self.cfg.court_x_max and abs(self.shuttle_pos[1]) < self.cfg.court_y_max
            if in_bounds:
                reward += 30.0
                info["landed_in"] = True
            else:
                reward -= 8.0
                info["landed_out"] = True

        # Keep robot roughly upright through reset-based base orientation.
        reward += 0.05
        return float(reward), info

    def _is_success_done(self, info):
        return bool(info.get("landed_in", False))

    def _is_failure_done(self):
        if self.shuttle_pos[2] < 0.05 and not self.has_hit:
            return True
        if abs(self.shuttle_pos[0]) > 4.5 or abs(self.shuttle_pos[1]) > 2.5:
            return True
        return False


# ----------------------------------------------------------------------
# Training / playing
# ----------------------------------------------------------------------


def make_env(rank: int, stage: int, gui: bool = False, seed: int = 0):
    def _init():
        cfg = EnvConfig(gui=gui, stage=stage, seed=seed + rank)
        return BadmintonHitEnv(cfg)
    return _init


def train(args):
    os.makedirs(args.run_dir, exist_ok=True)

    n_envs = args.n_envs
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = vec_cls([make_env(i, stage=1, gui=False, seed=args.seed) for i in range(n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=None,
    )

    # stable-baselines3 expects an activation class, not a string.
    import torch.nn as nn
    policy_kwargs["activation_fn"] = nn.ELU

    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(args.run_dir, "tb"),
        verbose=1,
        seed=args.seed,
    )

    checkpoint = CheckpointCallback(
        save_freq=max(args.n_steps * n_envs, 10000),
        save_path=args.run_dir,
        name_prefix="badminton_ckpt",
    )

    stage_steps = args.timesteps // 3
    for stage in [1, 2, 3]:
        print(f"\n========== Training stage {stage} ==========")
        env.env_method("set_stage", stage)
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=(stage == 1), callback=checkpoint, progress_bar=True)
        model.save(os.path.join(args.run_dir, f"badminton_s{stage}"))
        env.save(os.path.join(args.run_dir, f"vecnormalize_s{stage}.pkl"))

    model.save(os.path.join(args.run_dir, "badminton_final"))
    env.save(os.path.join(args.run_dir, "vecnormalize_final.pkl"))
    env.close()
    print("Training complete.")


def play(args):
    cfg = EnvConfig(gui=True, stage=3, seed=args.seed)
    env = BadmintonHitEnv(cfg)
    model = PPO.load(args.model)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(env.cfg.control_dt)
        if terminated or truncated:
            print("episode end", info)
            obs, _ = env.reset()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--model", type=str, default="runs/badminton_final.zip")
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args)
    elif args.play:
        play(args)
    else:
        print("Choose one: --train or --play")
