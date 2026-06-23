#!/usr/bin/env python3
"""
双机器人羽毛球对打演示 —— 国际比赛计分规则
- 每球得分制，21分一局，领先2分，29平后先到30分赢
- 三局两胜，得分方继续发球，每局后交换场地和发球权
- 实时预测轨迹、可选PPO挥拍辅助
"""

import argparse
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

try:
    from physics_shuttle import ShuttlePhysics
except Exception:
    class ShuttlePhysics:
        def __init__(self, g=-9.8, drag_coeff=0.004):
            self.g = g
            self.drag_coeff = drag_coeff

        def update(self, pos, vel, dt, force=None):
            pos = np.asarray(pos, dtype=float)
            vel = np.asarray(vel, dtype=float)
            acc = np.array([0.0, 0.0, self.g])
            speed = np.linalg.norm(vel)
            if speed > 1e-8:
                acc += -self.drag_coeff * speed * vel
            if force is not None:
                acc += np.asarray(force) / 0.005
            vel = vel + acc * dt
            pos = pos + vel * dt
            return pos, vel


JOINT_NAMES = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_shoulder_pitch", "left_elbow", "left_wrist",
    "right_shoulder_pitch", "right_elbow", "right_wrist",
]


class StableTwoRobotBadminton:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)

        self.left_urdf = args.left_urdf
        self.right_urdf = args.right_urdf
        self.fixed_base = not args.free_base
        self.control_dt = 0.02
        self.sim_dt = 1.0 / 240.0
        self.net_height = 1.55
        self.hit_radius = float(args.hit_radius)
        self.contact_assist = not args.no_contact_assist
        self.use_policy_swing = bool(args.use_policy_swing)
        self.use_policy_footwork = bool(args.use_policy_footwork)

        self.left_x_min, self.left_x_max = -3.85, -0.45
        self.right_x_min, self.right_x_max = 0.45, 3.85
        self.y_limit = 1.80

        self.footwork_speed_x = float(args.footwork_speed_x)
        self.footwork_speed_y = float(args.footwork_speed_y)
        self.passive_return_step = float(args.passive_return_step)

        self.model = None
        self.vecnorm = None
        if self.use_policy_swing:
            self._load_policy(args.model, args.vecnorm)

        self.client = p.connect(p.GUI)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet GUI.")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(6.4, 35, -25, [0.0, 0.0, 1.0])

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

        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=float(args.drag_coeff))
        self.shuttle_pos = np.zeros(3, dtype=float)
        self.shuttle_vel = np.zeros(3, dtype=float)
        self.shuttle_hist = deque(maxlen=6)

        self.step_count = 0
        self.hit_count = 0
        self.rally_count = 0
        self.best_hits = 0
        self.last_hit_step = -100
        self.last_hitter: Optional[str] = None
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0

        # 国际比赛计分
        self.game_point = 21      # 每局分数
        self.deuce_max = 30       # 29平后先到30分赢
        self.sets_to_win = 2      # 三局两胜
        self.left_sets = 0
        self.right_sets = 0
        self.set_score = [0, 0]   # 当前局比分
        self.serving = "left"
        self.match_over = False

        # 轨迹绘制
        self.trail_update_counter = 0
        self.trail_update_interval = 10

        self._build_world()
        self.reset_rally()

    # ------------------------------------------------------------------
    # PPO loading (optional)
    # ------------------------------------------------------------------
    def _load_policy(self, model_path: Optional[str], vecnorm_path: Optional[str]):
        try:
            import gymnasium as gym
            from gymnasium import spaces
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            OBS_DIM = 18 + 2 + 6 + 3 + 1 + 1
            ACTION_DIM = 4

            if not model_path or not os.path.exists(model_path):
                print("Warning: PPO model not found. Using deterministic swing only.")
                self.use_policy_swing = False
                return

            self.model = PPO.load(model_path)
            print("Loaded PPO model:", model_path)

            if vecnorm_path and os.path.exists(vecnorm_path):
                class DummyEnv(gym.Env):
                    def __init__(self):
                        super().__init__()
                        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
                        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,))
                    def reset(self, *, seed=None, options=None):
                        return np.zeros(OBS_DIM, dtype=np.float32), {}
                    def step(self, action):
                        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}

                dummy = DummyVecEnv([lambda: DummyEnv()])
                self.vecnorm = VecNormalize.load(vecnorm_path, dummy)
                self.vecnorm.training = False
                self.vecnorm.norm_reward = False
                print("Loaded VecNormalize:", vecnorm_path)
        except ImportError:
            print("Warning: stable-baselines3/gymnasium not available. Using deterministic swing only.")
            self.use_policy_swing = False

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        return obs

    def _make_obs(self, side: str) -> np.ndarray:
        target, t_hit = self._estimate_intercept(side)
        hist = np.array(list(self.shuttle_hist), dtype=float)
        if len(hist) < 6:
            hist = np.tile(self.shuttle_pos, (6, 1))

        if side == "left":
            base_pos, _ = p.getBasePositionAndOrientation(self.left_robot)
            base_xy = np.array(base_pos[:2], dtype=float)
            racket_pos = self.left_racket_pos.copy()
            racket_vel = self.left_racket_vel.copy()
            phase = self.last_phase_left
        else:
            base_pos, _ = p.getBasePositionAndOrientation(self.right_robot)
            base_xy = np.array(base_pos[:2], dtype=float)
            racket_pos = self.right_racket_pos.copy()
            racket_vel = self.right_racket_vel.copy()
            phase = self.last_phase_right
            # mirror to left coordinate
            hist = hist.copy(); hist[:, 0] *= -1.0
            base_xy = base_xy.copy(); base_xy[0] *= -1.0
            racket_pos = racket_pos.copy(); racket_pos[0] *= -1.0
            racket_vel = racket_vel.copy(); racket_vel[0] *= -1.0
            target = target.copy(); target[0] *= -1.0

        obs = np.concatenate([
            hist.reshape(-1) * np.array([0.25, 0.5, 0.5] * 6),
            base_xy * np.array([0.25, 0.5]),
            racket_pos * np.array([0.25, 0.5, 0.5]),
            racket_vel * 0.1,
            target * np.array([0.25, 0.5, 0.5]),
            np.array([t_hit, phase], dtype=float),
        ]).astype(np.float32)
        return obs

    def _policy_swing_action(self, side: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        obs = self._make_obs(side).reshape(1, -1)
        obs = self._normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _build_world(self):
        for path in [self.left_urdf, self.right_urdf]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}. Run 2.py first.")
        p.loadURDF("plane.urdf")
        self._add_court()
        self._add_net()

        self.left_robot = p.loadURDF(self.left_urdf, [-2.45, 0.0, 1.30], [0, 0, 0, 1],
                                     useFixedBase=self.fixed_base)
        self.right_robot = p.loadURDF(self.right_urdf, [2.45, 0.0, 1.30], [0, 0, 1, 0],
                                      useFixedBase=self.fixed_base)

        self.left_map = self._make_joint_map(self.left_robot)
        self.right_map = self._make_joint_map(self.right_robot)
        self.left_racket_link = self._find_link(self.left_robot, ["left_racket", "racket", "left_hand"])
        self.right_racket_link = self._find_link(self.right_robot, ["right_racket", "racket", "right_hand"])
        self.shuttle_id = self._create_shuttle()

        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

    def _add_court(self):
        green = [0.08, 0.42, 0.18, 1.0]; green2 = [0.10, 0.50, 0.22, 1.0]; white = [1.0, 1.0, 1.0, 1.0]
        z = 0.006
        def add_box(center, half_extents, rgba):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=center)
        add_box([0, 0, z / 2], [4.15, 1.95, z / 2], green)
        add_box([-2.55, 0, z + 0.001], [1.35, 1.85, 0.001], green2)
        add_box([2.55, 0, z + 0.001], [1.35, 1.85, 0.001], green2)
        def line(pos, size): add_box(pos, size, white)
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
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2], rgbaColor=[0.2, 0.8, 0.2, 0.45])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h / 2])
        p.createMultiBody(0, col, vis, [0, 0, h / 2])

    def _create_shuttle(self):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.038, length=0.09, rgbaColor=[0.95, 0.95, 1.0, 1.0])
        return p.createMultiBody(0.005, col, vis, [0, 0, 1.4])

    def _make_joint_map(self, robot_id: int) -> Dict[str, int]:
        return {p.getJointInfo(robot_id, i)[1].decode("utf-8"): i for i in range(p.getNumJoints(robot_id))}

    def _find_link(self, robot_id: int, names: List[str]) -> int:
        for name in names:
            for i in range(p.getNumJoints(robot_id)):
                if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
                    return i
        raise RuntimeError(f"Cannot find any link from {names}")

    # ------------------------------------------------------------------
    # Reset and serve
    # ------------------------------------------------------------------
    def reset_rally(self):
        self.rally_count += 1
        self.best_hits = max(self.best_hits, self.hit_count)
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
        p.resetBaseVelocity(self.left_robot, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.right_robot, [0, 0, 0], [0, 0, 0])
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

        self._serve_to(self.serving)   # 当前发球方发球
        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())
        self._update_racket_memory()

        print(f"\nRALLY {self.rally_count} | Serving: {self.serving} | Set: {self.set_score} | Sets: L {self.left_sets} - R {self.right_sets}")

    def _serve_to(self, receiver: str):
        if receiver == "left":
            start = np.array([3.20, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.45, 1.75)])
            target = np.array([-1.65, self.rng.uniform(-0.45, 0.45), self.rng.uniform(1.38, 1.62)])
        else:
            start = np.array([-3.20, self.rng.uniform(-0.55, 0.55), self.rng.uniform(1.45, 1.75)])
            target = np.array([1.65, self.rng.uniform(-0.45, 0.45), self.rng.uniform(1.38, 1.62)])
        T = self.rng.uniform(0.95, 1.12)
        self.shuttle_pos = start
        self.shuttle_vel = self._ballistic_velocity(start, target, T)
        p.resetBasePositionAndOrientation(self.shuttle_id, start, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)

    def _reset_neutral(self, robot_id: int, joint_map: Dict[str, int]):
        neutral = {
            "left_hip": 0.0, "left_knee": 0.20, "left_ankle": -0.06,
            "right_hip": 0.0, "right_knee": 0.20, "right_ankle": -0.06,
            "left_shoulder_pitch": 0.25, "left_elbow": 0.35, "left_wrist": 0.0,
            "right_shoulder_pitch": 0.25, "right_elbow": 0.35, "right_wrist": 0.0,
        }
        for name, value in neutral.items():
            if name not in joint_map:
                continue
            jid = joint_map[name]
            p.resetJointState(robot_id, jid, value, 0.0)
            p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL,
                                    targetPosition=value, force=180, maxVelocity=6.0)

    # ------------------------------------------------------------------
    # Prediction and trail
    # ------------------------------------------------------------------
    def _receiver_side(self) -> str:
        vx = float(self.shuttle_vel[0])
        if vx < -0.05:
            return "left"
        if vx > 0.05:
            return "right"
        return "left" if self.shuttle_pos[0] < 0 else "right"

    def _other_side(self, side: str) -> str:
        return "right" if side == "left" else "left"

    def _estimate_intercept(self, side: str) -> Tuple[np.ndarray, float]:
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
        target[2] = float(np.clip(target[2], 1.18, 1.75))
        return target, t_hit

    def draw_prediction_trail(self):
        sim_pos = self.shuttle_pos.copy()
        sim_vel = self.shuttle_vel.copy()
        future_points = [sim_pos.copy()]
        for _ in range(30):
            sim_pos, sim_vel = self.shuttle_phys.update(sim_pos, sim_vel, 0.02)
            future_points.append(sim_pos.copy())
        for i in range(len(future_points) - 1):
            p.addUserDebugLine(future_points[i], future_points[i + 1],
                               lineColorRGB=[0.2, 0.6, 1.0], lineWidth=2.0, lifeTime=0.3)
        p.addUserDebugText("●", future_points[-1], textColorRGB=[0.2, 0.6, 1.0], textSize=0.7, lifeTime=0.3)

    # ------------------------------------------------------------------
    # Movement and swing
    # ------------------------------------------------------------------
    def _compute_phase_power(self, side: str) -> Tuple[float, float]:
        _, t_hit = self._estimate_intercept(side)
        if t_hit >= 0.85:
            phase = 0.12
        elif t_hit >= 0.45:
            phase = np.interp(t_hit, [0.85, 0.45], [0.12, 0.32])
        elif t_hit >= 0.10:
            phase = np.interp(t_hit, [0.45, 0.10], [0.32, 0.55])
        else:
            phase = 0.62
        power = 0.75
        pa = self._policy_swing_action(side)
        if pa is not None and len(pa) >= 4:
            phase = 0.75 * phase + 0.25 * float((pa[2] + 1.0) * 0.5)
            power = 0.60 * power + 0.40 * float((pa[3] + 1.0) * 0.5)
        return float(np.clip(phase, 0.05, 0.90)), float(np.clip(power, 0.25, 1.0))

    def _apply_active(self, side: str):
        target, t_hit = self._estimate_intercept(side)
        if side == "left":
            robot = self.left_robot; joint_map = self.left_map; desired_x = target[0] - 0.55; racket_side = "left"
        else:
            robot = self.right_robot; joint_map = self.right_map; desired_x = target[0] + 0.55; racket_side = "right"
        desired_y = target[1]

        if self.use_policy_footwork:
            pa = self._policy_swing_action(side)
            if pa is None:
                vx_cmd = 0.0; vy_cmd = 0.0
            else:
                vx_cmd = float(pa[0]); vy_cmd = float(pa[1])
                if side == "right": vx_cmd *= -1.0
        else:
            pos, _ = p.getBasePositionAndOrientation(robot)
            pos = np.array(pos, dtype=float)
            t_safe = max(float(t_hit), 0.12)
            vx_cmd = np.clip((desired_x - pos[0]) / t_safe / self.footwork_speed_x, -1.0, 1.0)
            vy_cmd = np.clip((desired_y - pos[1]) / t_safe / self.footwork_speed_y, -1.0, 1.0)

        self._move_robot_by_command(robot, vx_cmd, vy_cmd, side)
        phase, power = self._compute_phase_power(side)
        targets = self._swing_primitive(phase, power, racket_side)
        self._apply_leg_stance(targets, phase)
        self._apply_joint_targets(robot, joint_map, targets)

        if side == "left":
            self.last_phase_left = phase
        else:
            self.last_phase_right = phase

    def _apply_passive(self, side: str):
        if side == "left":
            self._move_base_toward(self.left_robot, [-2.45, 0.0], "left", self.passive_return_step)
            phase, power, racket_side = 0.10, 0.20, "left"
            robot, joint_map = self.left_robot, self.left_map
        else:
            self._move_base_toward(self.right_robot, [2.45, 0.0], "right", self.passive_return_step)
            phase, power, racket_side = 0.10, 0.20, "right"
            robot, joint_map = self.right_robot, self.right_map
        targets = self._swing_primitive(phase, power, racket_side)
        self._apply_leg_stance(targets, phase)
        self._apply_joint_targets(robot, joint_map, targets)

    def _move_robot_by_command(self, robot_id: int, vx_cmd: float, vy_cmd: float, side: str):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        pos = np.array(pos, dtype=float)
        pos[0] += float(vx_cmd) * self.footwork_speed_x * self.control_dt
        pos[1] += float(vy_cmd) * self.footwork_speed_y * self.control_dt
        if side == "left":
            pos[0] = np.clip(pos[0], self.left_x_min, self.left_x_max)
            orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], self.right_x_min, self.right_x_max)
            orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -self.y_limit, self.y_limit)
        p.resetBasePositionAndOrientation(robot_id, pos, orn)

    def _move_base_toward(self, robot_id: int, target_xy: List[float], side: str, max_step: float):
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        pos = np.array(pos, dtype=float)
        target = np.array(target_xy, dtype=float)
        pos[:2] += np.clip(target - pos[:2], -max_step, max_step)
        if side == "left":
            pos[0] = np.clip(pos[0], self.left_x_min, self.left_x_max)
            orn = [0, 0, 0, 1]
        else:
            pos[0] = np.clip(pos[0], self.right_x_min, self.right_x_max)
            orn = [0, 0, 1, 0]
        pos[1] = np.clip(pos[1], -self.y_limit, self.y_limit)
        p.resetBasePositionAndOrientation(robot_id, pos, orn)

    def _apply_leg_stance(self, targets: Dict[str, float], phase: float):
        targets.update({
            "left_hip": 0.05 * np.sin(phase * 2 * np.pi),
            "right_hip": -0.05 * np.sin(phase * 2 * np.pi),
            "left_knee": 0.22,
            "right_knee": 0.22,
            "left_ankle": -0.08,
            "right_ankle": -0.08,
        })

    def _apply_joint_targets(self, robot_id: int, joint_map: Dict[str, int], targets: Dict[str, float]):
        for name, value in targets.items():
            if name not in joint_map:
                continue
            p.setJointMotorControl2(robot_id, joint_map[name], p.POSITION_CONTROL,
                                    targetPosition=float(value), force=300, maxVelocity=12.0)

    def _swing_primitive(self, phase: float, power: float, racket_side: str) -> Dict[str, float]:
        amp = 0.82 + 0.78 * power
        if phase < 0.35:
            t = phase / 0.35
            shoulder = 0.20 + 1.30 * amp * t
            elbow = 0.30 + 0.80 * amp * t
            wrist = 0.00 + 0.50 * amp * t
        elif phase < 0.65:
            t = (phase - 0.35) / 0.30
            shoulder = 1.50 * amp - 2.35 * amp * t
            elbow = 1.10 * amp - 0.85 * amp * t
            wrist = 0.55 * amp - 1.05 * amp * t
        else:
            t = (phase - 0.65) / 0.35
            shoulder = -0.78 * amp + 1.00 * amp * t
            elbow = 0.25 + 0.15 * t
            wrist = -0.52 * amp + 0.50 * amp * t

        if racket_side == "left":
            return {
                "left_shoulder_pitch": float(np.clip(shoulder, -1.8, 1.8)),
                "left_elbow": float(np.clip(elbow, 0.0, 1.8)),
                "left_wrist": float(np.clip(wrist, -0.9, 0.9)),
                "right_shoulder_pitch": float(np.clip(-0.35 * shoulder, -1.8, 1.8)),
                "right_elbow": 0.35,
                "right_wrist": float(np.clip(-0.3 * wrist, -0.9, 0.9)),
            }
        else:
            return {
                "right_shoulder_pitch": float(np.clip(-shoulder, -1.8, 1.8)),
                "right_elbow": float(np.clip(elbow, 0.0, 1.8)),
                "right_wrist": float(np.clip(-wrist, -0.9, 0.9)),
                "left_shoulder_pitch": float(np.clip(0.35 * shoulder, -1.8, 1.8)),
                "left_elbow": 0.35,
                "left_wrist": float(np.clip(0.3 * wrist, -0.9, 0.9)),
            }

    # ------------------------------------------------------------------
    # Racket/shuttle logic
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
        state = p.getLinkState(robot_id, link_idx, computeLinkVelocity=1)
        pos = np.array(state[0], dtype=float)
        vel = np.zeros(3, dtype=float) if prev is None else (pos - prev) / self.control_dt
        return pos, vel

    def _maybe_hit(self, side: str):
        if self.step_count - self.last_hit_step < int(self.args.hit_cooldown_steps):
            return

        if side == "left":
            racket_pos = self.left_racket_pos
            racket_vel = self.left_racket_vel
            forward = float(racket_vel[0])
            approaching = self.shuttle_vel[0] < -0.05
            valid_half = self.shuttle_pos[0] < 0.35
            outgoing_target = np.array([1.65, self.rng.uniform(-0.60, 0.60), self.rng.uniform(1.40, 1.70)])
        else:
            racket_pos = self.right_racket_pos
            racket_vel = self.right_racket_vel
            forward = float(-racket_vel[0])
            approaching = self.shuttle_vel[0] > 0.05
            valid_half = self.shuttle_pos[0] > -0.35
            outgoing_target = np.array([-1.65, self.rng.uniform(-0.60, 0.60), self.rng.uniform(1.40, 1.70)])

        dist = float(np.linalg.norm(racket_pos - self.shuttle_pos))
        good_height = 0.25 < self.shuttle_pos[2] < 2.15
        forward_ok = forward > float(self.args.forward_threshold)

        if self.contact_assist:
            contact_ok = dist < self.hit_radius
        else:
            contact_ok = dist < min(self.hit_radius, 0.65)

        if not (approaching and valid_half and good_height and contact_ok and forward_ok):
            return

        t_flight = self.rng.uniform(float(self.args.return_time_min), float(self.args.return_time_max))
        vout = self._ballistic_velocity(self.shuttle_pos, outgoing_target, t_flight)
        vout[1] += 0.10 * racket_vel[1]
        vout[2] = max(vout[2], float(self.args.min_return_vz))
        speed = np.linalg.norm(vout)
        if speed > float(self.args.max_return_speed):
            vout = vout / speed * float(self.args.max_return_speed)

        self.shuttle_vel = vout
        self.hit_count += 1
        self.last_hit_step = self.step_count
        self.last_hitter = side
        print(f"{side.upper()} HIT #{self.hit_count}: dist={dist:.2f}")

    def _update_shuttle(self):
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(
            self.shuttle_pos, self.shuttle_vel, self.control_dt, force=None)
        self.shuttle_hist.append(self.shuttle_pos.copy())
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel, [0, 0, 0])

    def _ballistic_velocity(self, start: np.ndarray, target: np.ndarray, t_flight: float) -> np.ndarray:
        g = -9.8
        return np.array([
            (target[0] - start[0]) / t_flight,
            (target[1] - start[1]) / t_flight,
            (target[2] - start[2] - 0.5 * g * t_flight * t_flight) / t_flight,
        ], dtype=float)

    # ------------------------------------------------------------------
    # International badminton scoring
    # ------------------------------------------------------------------
    def _update_score_and_serve(self):
        """Determines who wins the rally and updates game/set/match accordingly."""
        if self.match_over:
            return
        scorer = None

        # net fault
        if abs(self.shuttle_pos[0]) < 0.06 and self.shuttle_pos[2] < self.net_height:
            if self.last_hitter == "left":
                scorer = "right"
            elif self.last_hitter == "right":
                scorer = "left"
            else:
                scorer = "right" if self.serving == "left" else "left"
        # out of bounds
        elif abs(self.shuttle_pos[0]) > 4.45 or abs(self.shuttle_pos[1]) > 2.55:
            if self.last_hitter == "left":
                scorer = "right"
            elif self.last_hitter == "right":
                scorer = "left"
            else:
                scorer = "left" if self.shuttle_vel[0] > 0 else "right"
        # shuttle landed
        elif self.shuttle_pos[2] < 0.06:
            if self.shuttle_pos[0] < 0:
                scorer = "right"
            else:
                scorer = "left"
        else:
            return   # no score (e.g., time limit)

        # Apply point
        if scorer == "left":
            self.set_score[0] += 1
            self.serving = "left"
        else:
            self.set_score[1] += 1
            self.serving = "right"

        print(f"Point {scorer.upper()}! Set: {self.set_score[0]} - {self.set_score[1]}")

        # Check set win
        a, b = self.set_score
        if (a >= 21 and a - b >= 2) or a == 30:
            if scorer == "left":
                self.left_sets += 1
            else:
                self.right_sets += 1
            print(f"*** SET WON by {scorer.upper()} *** Sets: L {self.left_sets} - R {self.right_sets}")
            self.set_score = [0, 0]
            # serve alternates after a set
            self.serving = "right" if scorer == "left" else "left"

        # Check match win
        if self.left_sets >= self.sets_to_win or self.right_sets >= self.sets_to_win:
            winner = "LEFT" if self.left_sets >= self.sets_to_win else "RIGHT"
            print(f"*** MATCH WON BY {winner}! ***")
            self.match_over = True
            # Reset for a new match (optional)
            self.left_sets = 0
            self.right_sets = 0
            self.set_score = [0, 0]
            self.serving = "left"
            self.match_over = False

    def _rally_done(self) -> bool:
        if self.shuttle_pos[2] < 0.06:
            print("RALLY END: landed.")
            self._update_score_and_serve()
            return True
        if abs(self.shuttle_pos[0]) > 4.45 or abs(self.shuttle_pos[1]) > 2.55:
            print("RALLY END: out.")
            self._update_score_and_serve()
            return True
        if abs(self.shuttle_pos[0]) < 0.06 and self.shuttle_pos[2] < self.net_height:
            print("RALLY END: net fail.")
            self._update_score_and_serve()
            return True
        if self.step_count > int(self.args.max_steps_per_rally):
            print("RALLY END: time limit.")
            self._update_score_and_serve()
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        try:
            while p.isConnected():
                receiver = self._receiver_side()
                passive = self._other_side(receiver)

                self._apply_active(receiver)
                self._apply_passive(passive)

                sim_steps = max(1, int(round(self.control_dt / self.sim_dt)))
                for _ in range(sim_steps):
                    if not p.isConnected():
                        break
                    p.stepSimulation()
                if not p.isConnected():
                    break

                self._update_racket_memory()
                self._maybe_hit(receiver)
                self._update_shuttle()
                self.step_count += 1

                self.trail_update_counter += 1
                if self.trail_update_counter >= self.trail_update_interval:
                    self.draw_prediction_trail()
                    self.trail_update_counter = 0

                if self._rally_done():
                    time.sleep(0.25)
                    self.reset_rally()

                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            p.disconnect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_urdf", type=str, default="models/humanoid_left.urdf")
    parser.add_argument("--right_urdf", type=str, default="models/humanoid_right.urdf")
    parser.add_argument("--model", type=str, default="runs_two_side/badminton_two_side_final.zip")
    parser.add_argument("--vecnorm", type=str, default="runs_two_side/vecnormalize_two_side_final.pkl")
    parser.add_argument("--use_policy_swing", action="store_true")
    parser.add_argument("--use_policy_footwork", action="store_true")
    parser.add_argument("--free_base", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hit_radius", type=float, default=1.35)
    parser.add_argument("--forward_threshold", type=float, default=-0.05)
    parser.add_argument("--hit_cooldown_steps", type=int, default=20)
    parser.add_argument("--no_contact_assist", action="store_true")
    parser.add_argument("--footwork_speed_x", type=float, default=4.8)
    parser.add_argument("--footwork_speed_y", type=float, default=3.8)
    parser.add_argument("--passive_return_step", type=float, default=0.065)
    parser.add_argument("--drag_coeff", type=float, default=0.004)
    parser.add_argument("--return_time_min", type=float, default=1.0)
    parser.add_argument("--return_time_max", type=float, default=1.25)
    parser.add_argument("--min_return_vz", type=float, default=5.15)
    parser.add_argument("--max_return_speed", type=float, default=13.5)
    parser.add_argument("--max_steps_per_rally", type=int, default=800)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = StableTwoRobotBadminton(args)
    demo.run()