#!/usr/bin/env python3
"""
Rally-oriented retraining script for the simplified PyBullet badminton model.

Use this when:
- train_rl_badminton_easy.py works for one robot;
- but two robots cannot keep rallying.

Core change:
- The policy is no longer trained only on one simple machine-serve distribution.
- It is trained on a broader rally-like incoming-shuttle distribution:
  1) machine serve style;
  2) opponent return style;
  3) faster / higher / lower trajectories;
  4) target-known intercept point and time, matching the easy policy observation.

Action space stays easy and learnable:
    action[0] = base x movement
    action[1] = base y movement
    action[2] = swing phase
    action[3] = swing power

Save as:
    train_rl_badminton_rally_easy.py

Train:
    python train_rl_badminton_rally_easy.py --train --timesteps 1200000 --n_envs 4

Single policy test:
    python train_rl_badminton_rally_easy.py --play_single --model runs_rally_easy\badminton_rally_final.zip

Two robot demo:
    python train_rl_badminton_rally_easy.py --play_two --model runs_rally_easy\badminton_rally_final.zip
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
class RallyConfig:
    left_urdf: str = "models/humanoid_left.urdf"
    right_urdf: str = "models/humanoid_right.urdf"
    gui: bool = False
    fixed_base: bool = True
    control_dt: float = 0.02
    sim_dt: float = 1.0 / 240.0
    max_episode_time: float = 2.2
    stage: int = 1
    seed: int = 0
    hit_radius: float = 0.88


class RallyEasyBadmintonEnv(gym.Env):
    """Left-side canonical receiver environment.

    The policy still learns a canonical left-side task. In two-robot demo, the right
    robot is mirrored into this same canonical frame.
    """

    def __init__(self, cfg: Optional[RallyConfig] = None):
        super().__init__()
        self.cfg = cfg or RallyConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.client = p.connect(p.GUI if self.cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.cfg.sim_dt, physicsClientId=self.client)
        if self.cfg.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(6.0, 35, -25, [0.0, 0.0, 1.0], physicsClientId=self.client)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.robot_id = -1
        self.opponent_id = -1
        self.shuttle_id = -1
        self.joint_map: Dict[str, int] = {}
        self.racket_link = -1

        self.prev_racket_pos = None
        self.racket_vel = np.zeros(3)
        self.shuttle_hist = deque(maxlen=6)
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.010)

        self.shuttle_pos = np.zeros(3)
        self.shuttle_vel = np.zeros(3)
        self.hit_target = np.zeros(3)
        self.time_to_hit = 1.0
        self.elapsed = 0.0
        self.step_count = 0
        self.has_hit = False
        self.cleared_net = False
        self.landed = False
        self.last_phase = 0.0
        self.last_action = np.zeros(ACTION_DIM)

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
            raise FileNotFoundError("Missing models/humanoid_left.urdf. Run 2.py once first.")

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
        self.racket_link = self._find_link(self.robot_id, ["left_racket", "left_hand"])

        if self.cfg.gui and os.path.exists(self.cfg.right_urdf):
            self.opponent_id = p.loadURDF(
                self.cfg.right_urdf,
                [2.45, 0.0, 1.30],
                [0, 0, 1, 0],
                useFixedBase=True,
                physicsClientId=self.client,
            )

        self.shuttle_id = self._create_shuttle()

    def _add_court(self):
        green = [0.08, 0.42, 0.18, 1]
        white = [1, 1, 1, 1]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[4.1, 1.95, 0.003], rgbaColor=green, physicsClientId=self.client)
        p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0.002], physicsClientId=self.client)

        def line(pos, size):
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=white, physicsClientId=self.client)
            p.createMultiBody(0, baseVisualShapeIndex=v, basePosition=pos, physicsClientId=self.client)

        line([0, -1.85, 0.012], [4.0, 0.018, 0.005])
        line([0, 1.85, 0.012], [4.0, 0.018, 0.005])
        line([-4.0, 0, 0.012], [0.018, 1.85, 0.005])
        line([4.0, 0, 0.012], [0.018, 1.85, 0.005])
        line([0, 0, 0.014], [0.018, 1.85, 0.005])
        line([-1.2, 0, 0.014], [0.018, 1.85, 0.005])
        line([1.2, 0, 0.014], [0.018, 1.85, 0.005])

    def _add_net(self):
        h = 1.55
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2], rgbaColor=[0.2, 0.8, 0.2, 0.45], physicsClientId=self.client)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2], physicsClientId=self.client)
        p.createMultiBody(0, collision, visual, [0, 0, h / 2], physicsClientId=self.client)

    def _create_shuttle(self):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.038, length=0.09, rgbaColor=[0.95, 0.95, 1.0, 1.0], physicsClientId=self.client)
        return p.createMultiBody(0.005, col, vis, [0, 0, 1.4], physicsClientId=self.client)

    def _make_joint_map(self, robot_id: int) -> Dict[str, int]:
        mapping = {p.getJointInfo(robot_id, i, physicsClientId=self.client)[1].decode("utf-8"): i for i in range(p.getNumJoints(robot_id, physicsClientId=self.client))}
        missing = [j for j in JOINT_NAMES if j not in mapping]
        if missing:
            raise RuntimeError(f"Missing joints in URDF: {missing}")
        return mapping

    def _find_link(self, robot_id: int, names: List[str]) -> int:
        for name in names:
            for i in range(p.getNumJoints(robot_id, physicsClientId=self.client)):
                link = p.getJointInfo(robot_id, i, physicsClientId=self.client)[12].decode("utf-8")
                if link == name:
                    return i
        raise RuntimeError(f"Cannot find link from {names}")

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
        self.landed = False
        self.last_phase = 0.0
        self.last_action[:] = 0.0
        self.prev_racket_pos = None
        self.racket_vel[:] = 0

        p.resetBasePositionAndOrientation(self.robot_id, [-2.45, 0.0, 1.30], [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self._reset_neutral_pose()
        if self.opponent_id >= 0:
            p.resetBasePositionAndOrientation(self.opponent_id, [2.45, 0.0, 1.30], [0, 0, 1, 0], physicsClientId=self.client)

        self._sample_rally_like_shuttle()
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

        self._apply_action(action)
        for _ in range(max(1, int(round(self.cfg.control_dt / self.cfg.sim_dt)))):
            p.stepSimulation(physicsClientId=self.client)

        racket_pos, racket_vel = self._racket_state()
        force = self._hit_force(racket_pos, racket_vel)
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(self.shuttle_pos, self.shuttle_vel, self.cfg.control_dt, force)
        self.shuttle_hist.append(self.shuttle_pos.copy())
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0], physicsClientId=self.client)

        self.elapsed += self.cfg.control_dt
        self.time_to_hit -= self.cfg.control_dt
        self.step_count += 1

        reward, info = self._reward(racket_pos, racket_vel, action, prev_action)
        terminated = bool(info.get("landed_in", False))
        truncated = self.elapsed > self.cfg.max_episode_time or self._failure()
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_rally_like_shuttle(self):
        """Sample a broad incoming trajectory toward the left robot.

        This intentionally covers the distribution caused by the two-robot demo.
        """
        # Receiver target: where the robot should intercept.
        self.hit_target = np.array(
            [
                self.rng.uniform(-1.95, -1.25),
                self.rng.uniform(-0.75, 0.75),
                self.rng.uniform(1.20, 1.70),
            ],
            dtype=float,
        )

        mode = self.rng.choice(["serve", "flat_return", "loft_return"], p=[0.35, 0.40, 0.25])
        if mode == "serve":
            start = np.array([self.rng.uniform(2.7, 3.6), self.rng.uniform(-0.95, 0.95), self.rng.uniform(1.30, 1.75)], dtype=float)
            T = self.rng.uniform(0.85, 1.18)
        elif mode == "flat_return":
            start = np.array([self.rng.uniform(1.1, 2.4), self.rng.uniform(-0.90, 0.90), self.rng.uniform(1.20, 1.85)], dtype=float)
            T = self.rng.uniform(0.55, 0.90)
        else:
            start = np.array([self.rng.uniform(1.3, 3.0), self.rng.uniform(-1.10, 1.10), self.rng.uniform(1.55, 2.40)], dtype=float)
            T = self.rng.uniform(0.80, 1.35)

        self.time_to_hit = float(T)
        self.shuttle_pos = start
        self.shuttle_vel = self._ballistic_velocity(start, self.hit_target, T)

        # Add modest perturbation so it does not overfit exact ballistic targets.
        self.shuttle_vel += np.array([
            self.rng.uniform(-0.25, 0.25),
            self.rng.uniform(-0.25, 0.25),
            self.rng.uniform(-0.15, 0.20),
        ])

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
    # Robot action
    # ------------------------------------------------------------------

    def _reset_neutral_pose(self):
        neutral = {
            "left_hip": 0.0, "left_knee": 0.20, "left_ankle": -0.06,
            "right_hip": 0.0, "right_knee": 0.20, "right_ankle": -0.06,
            "left_shoulder_pitch": 0.25, "left_elbow": 0.35, "left_wrist": 0.0,
            "right_shoulder_pitch": 0.25, "right_elbow": 0.35, "right_wrist": 0.0,
        }
        for name, value in neutral.items():
            p.resetJointState(self.robot_id, self.joint_map[name], value, 0.0, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot_id, self.joint_map[name], p.POSITION_CONTROL, targetPosition=value, force=180, maxVelocity=6, physicsClientId=self.client)

    def _apply_action(self, action: np.ndarray):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        pos = np.array(pos, dtype=float)
        pos[0] += float(action[0]) * 1.9 * self.cfg.control_dt
        pos[1] += float(action[1]) * 1.6 * self.cfg.control_dt
        pos[0] = np.clip(pos[0], -3.8, -0.50)
        pos[1] = np.clip(pos[1], -1.80, 1.80)
        p.resetBasePositionAndOrientation(self.robot_id, pos, [0, 0, 0, 1], physicsClientId=self.client)

        phase = float((action[2] + 1.0) * 0.5)
        power = float((action[3] + 1.0) * 0.5)
        self.last_phase = phase
        targets = self._swing_primitive(phase, power)
        targets.update({
            "left_hip": 0.05 * np.sin(phase * 2 * np.pi),
            "right_hip": -0.05 * np.sin(phase * 2 * np.pi),
            "left_knee": 0.22,
            "right_knee": 0.22,
            "left_ankle": -0.08,
            "right_ankle": -0.08,
        })
        for name, value in targets.items():
            p.setJointMotorControl2(self.robot_id, self.joint_map[name], p.POSITION_CONTROL, targetPosition=float(value), force=280, maxVelocity=11, physicsClientId=self.client)

    def _swing_primitive(self, phase: float, power: float) -> Dict[str, float]:
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
        return {
            "left_shoulder_pitch": float(np.clip(shoulder, -1.8, 1.8)),
            "left_elbow": float(np.clip(elbow, 0.0, 1.8)),
            "left_wrist": float(np.clip(wrist, -0.9, 0.9)),
            "right_shoulder_pitch": float(np.clip(-0.35 * shoulder, -1.8, 1.8)),
            "right_elbow": 0.35,
            "right_wrist": float(np.clip(-0.3 * wrist, -0.9, 0.9)),
        }

    def _racket_state(self) -> Tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.robot_id, self.racket_link, computeLinkVelocity=1, physicsClientId=self.client)
        pos = np.array(state[0], dtype=float)
        vel = np.zeros(3) if self.prev_racket_pos is None else (pos - self.prev_racket_pos) / self.cfg.control_dt
        self.prev_racket_pos = pos.copy()
        return pos, vel

    # ------------------------------------------------------------------
    # Hit and reward
    # ------------------------------------------------------------------

    def _hit_force(self, racket_pos: np.ndarray, racket_vel: np.ndarray):
        if self.has_hit:
            return None
        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        forward = float(racket_vel[0])
        good_height = 0.35 < self.shuttle_pos[2] < 1.95
        near_time = -0.35 < self.time_to_hit < 0.45
        if dist < self.cfg.hit_radius and forward > 0.10 and good_height and near_time:
            self.has_hit = True
            target = np.array([self.rng.uniform(1.25, 2.35), self.rng.uniform(-0.75, 0.75), self.rng.uniform(1.35, 1.65)])
            T = self.rng.uniform(0.90, 1.12)
            vout = self._ballistic_velocity(self.shuttle_pos, target, T)
            vout[1] += 0.12 * racket_vel[1]
            vout[2] = max(vout[2], 4.7)
            speed = np.linalg.norm(vout)
            if speed > 14.0:
                vout = vout / speed * 14.0
            mass = 0.005
            return mass * (vout - self.shuttle_vel) / self.cfg.control_dt
        return None

    def _reward(self, racket_pos: np.ndarray, racket_vel: np.ndarray, action: np.ndarray, prev_action: np.ndarray):
        stage = self.cfg.stage
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_pos = np.array(base_pos, dtype=float)
        base_target = np.array([self.hit_target[0] - 0.55, self.hit_target[1]])
        base_dist = np.linalg.norm(base_pos[:2] - base_target)
        target_dist = np.linalg.norm(racket_pos - self.hit_target)
        shuttle_dist = np.linalg.norm(racket_pos - self.shuttle_pos)
        timing = np.exp(-abs(self.time_to_hit) / 0.22)
        forward = max(0.0, racket_vel[0])

        r_foot = np.exp(-1.6 * base_dist)
        r_target = np.exp(-3.8 * target_dist)
        r_shuttle = np.exp(-4.5 * shuttle_dist)
        r_swing = timing * r_shuttle * forward
        r_phase = timing * np.exp(-8.0 * abs(self.last_phase - 0.50))

        reward = 0.0
        if stage == 1:
            reward += 3.0 * r_foot + 1.2 * r_target
        elif stage == 2:
            reward += 1.5 * r_foot + 3.0 * r_target + 1.2 * r_phase + 0.7 * r_swing
        else:
            reward += 1.0 * r_foot + 2.0 * r_target + 2.0 * r_shuttle + 1.4 * r_phase + 1.0 * r_swing

        reward -= 0.01 * float(np.mean(action ** 2))
        reward -= 0.02 * float(np.mean((action - prev_action) ** 2))

        info = {"hit": self.has_hit, "stage": stage, "target_dist": target_dist, "shuttle_dist": shuttle_dist}
        if self.has_hit:
            reward += 90.0
            info["hit_success"] = True
        if self.has_hit and not self.cleared_net and self.shuttle_pos[0] > 0.05:
            if self.shuttle_pos[2] > 1.62:
                self.cleared_net = True
                reward += 80.0
                info["cleared_net"] = True
            else:
                reward -= 20.0
                info["net_fail"] = True
        if self.has_hit and self.shuttle_pos[2] < 0.08:
            self.landed = True
            if 0.2 < self.shuttle_pos[0] < 4.0 and abs(self.shuttle_pos[1]) < 1.85:
                reward += 90.0
                info["landed_in"] = True
            else:
                reward -= 10.0
                info["landed_out"] = True
        if not self.has_hit and self.time_to_hit < -0.45:
            reward -= 12.0
            info["missed_window"] = True
        return float(reward), info

    def _get_obs(self) -> np.ndarray:
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        racket_pos, racket_vel = self._racket_state()
        hist = np.array(list(self.shuttle_hist), dtype=float)
        if hist.shape[0] < 6:
            hist = np.tile(self.shuttle_pos, (6, 1))
        return np.concatenate([
            hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
            np.array(base_pos[:2]) * np.array([0.25, 0.5]),
            racket_pos * np.array([0.25, 0.5, 0.5]),
            racket_vel * 0.1,
            self.hit_target * np.array([0.25, 0.5, 0.5]),
            np.array([self.time_to_hit, self.last_phase], dtype=float),
        ]).astype(np.float32)

    def _failure(self) -> bool:
        if not self.has_hit and self.time_to_hit < -0.60:
            return True
        if self.shuttle_pos[2] < 0.05:
            return True
        if abs(self.shuttle_pos[0]) > 4.6 or abs(self.shuttle_pos[1]) > 2.6:
            return True
        return False


# ----------------------------------------------------------------------
# Training / playback
# ----------------------------------------------------------------------


def make_env(rank: int, stage: int, gui: bool, seed: int):
    def _init():
        return RallyEasyBadmintonEnv(RallyConfig(gui=gui, stage=stage, seed=seed + rank))
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
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]), activation_fn=nn.ELU),
        tensorboard_log=os.path.join(args.run_dir, "tb"),
        verbose=1,
        seed=args.seed,
    )

    ckpt = CheckpointCallback(save_freq=max(args.n_steps * args.n_envs, 10000), save_path=args.run_dir, name_prefix="rally_easy_ckpt")
    stage_steps = args.timesteps // 3
    for stage in [1, 2, 3]:
        print(f"\n========== RALLY EASY TRAINING STAGE {stage} ==========")
        env.env_method("set_stage", stage)
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=(stage == 1), callback=ckpt, progress_bar=True)
        model.save(os.path.join(args.run_dir, f"badminton_rally_s{stage}"))
        env.save(os.path.join(args.run_dir, f"vecnormalize_rally_s{stage}.pkl"))

    model.save(os.path.join(args.run_dir, "badminton_rally_final"))
    env.save(os.path.join(args.run_dir, "vecnormalize_rally_final.pkl"))
    env.close()
    print("Rally-oriented training complete.")


def play_single(args):
    env = DummyVecEnv([make_env(0, 3, True, args.seed)])
    vec_path = args.vecnorm or os.path.join(os.path.dirname(args.model), "vecnormalize_rally_final.pkl")
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
    model = PPO.load(args.model, env=env)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(0.02)
        if bool(done[0]):
            print("episode end", info[0])
            obs = env.reset()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play_single", action="store_true")
    parser.add_argument("--model", type=str, default="runs_rally_easy/badminton_rally_final.zip")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default="runs_rally_easy")
    parser.add_argument("--timesteps", type=int, default=1_200_000)
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
    else:
        print("Use --train or --play_single")
