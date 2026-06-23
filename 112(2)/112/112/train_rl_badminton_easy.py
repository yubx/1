#!/usr/bin/env python3
"""
完整的双机器人羽毛球对打演示，包含：
- 实时预测轨迹
- 按正式比赛规则计分（每球得分制）
- 可选 PPO 挥拍辅助
- 流畅运行，无卡顿
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

        # 场地限制
        self.left_x_min, self.left_x_max = -3.85, -0.45
        self.right_x_min, self.right_x_max = 0.45, 3.85
        self.y_limit = 1.80

        # 移动速度
        self.footwork_speed_x = float(args.footwork_speed_x)
        self.footwork_speed_y = float(args.footwork_speed_y)
        self.passive_return_step = float(args.passive_return_step)

        # 可选 PPO
        self.model = None
        self.vecnorm = None
        if self.use_policy_swing:
            self._load_policy(args.model, args.vecnorm)

        # 启动 PyBullet
        self.client = p.connect(p.GUI)
        if self.client < 0:
            raise RuntimeError("无法启动 PyBullet GUI")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(6.4, 35, -25, [0.0, 0.0, 1.0])

        # 机器人/球 ID
        self.left_robot = self.right_robot = self.shuttle_id = -1
        self.left_map: Dict[str, int] = {}
        self.right_map: Dict[str, int] = {}
        self.left_racket_link = self.right_racket_link = -1

        # 球拍状态
        self.prev_left_racket = self.prev_right_racket = None
        self.left_racket_pos = np.zeros(3)
        self.right_racket_pos = np.zeros(3)
        self.left_racket_vel = np.zeros(3)
        self.right_racket_vel = np.zeros(3)

        # 羽毛球
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=float(args.drag_coeff))
        self.shuttle_pos = np.zeros(3)
        self.shuttle_vel = np.zeros(3)
        self.shuttle_hist = deque(maxlen=6)

        # 计分
        self.step_count = self.hit_count = self.rally_count = 0
        self.best_hits = 0
        self.last_hit_step = -100
        self.last_hitter: Optional[str] = None
        self.last_phase_left = 0.0
        self.last_phase_right = 0.0

        self.left_score = 0
        self.right_score = 0
        self.serving = "left"

        # 预测轨迹绘制间隔
        self.trail_update_counter = 0
        self.trail_update_interval = 10

        self._build_world()
        self.reset_rally()

    # ---------- PPO 加载 ----------
    def _load_policy(self, model_path, vecnorm_path):
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            import gymnasium as gym
            from gymnasium import spaces
            OBS_DIM = 18 + 2 + 6 + 3 + 1 + 1
            ACTION_DIM = 4

            if model_path and os.path.exists(model_path):
                self.model = PPO.load(model_path)
                print("PPO 模型已加载:", model_path)
            else:
                print("未找到模型文件，使用确定性挥拍")
                self.use_policy_swing = False
                return

            if vecnorm_path and os.path.exists(vecnorm_path):
                class DummyEnv(gym.Env):
                    def __init__(self):
                        super().__init__()
                        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
                        self.action_space = spaces.Box(-1, 1, shape=(ACTION_DIM,))
                    def reset(self, *args, **kwargs):
                        return np.zeros(OBS_DIM, dtype=np.float32), {}
                    def step(self, action):
                        return np.zeros(OBS_DIM), 0.0, False, False, {}
                dummy = DummyVecEnv([lambda: DummyEnv()])
                self.vecnorm = VecNormalize.load(vecnorm_path, dummy)
                self.vecnorm.training = False
                self.vecnorm.norm_reward = False
                print("VecNormalize 已加载:", vecnorm_path)
            else:
                print("VecNormalize 未找到，策略可能不稳定")
        except ImportError:
            print("缺少 stable-baselines3/gymnasium，禁用 PPO")
            self.use_policy_swing = False

    def _normalize_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        return obs

    def _make_obs(self, side):
        target, t_hit = self._estimate_intercept(side)
        hist = np.array(list(self.shuttle_hist), dtype=float)
        if len(hist) < 6:
            hist = np.tile(self.shuttle_pos, (6, 1))

        if side == "left":
            base, _ = p.getBasePositionAndOrientation(self.left_robot)
            base_xy = np.array(base[:2])
            racket_pos = self.left_racket_pos.copy()
            racket_vel = self.left_racket_vel.copy()
            phase = self.last_phase_left
        else:
            base, _ = p.getBasePositionAndOrientation(self.right_robot)
            base_xy = np.array(base[:2])
            racket_pos = self.right_racket_pos.copy()
            racket_vel = self.right_racket_vel.copy()
            phase = self.last_phase_right
            # 镜像
            hist = hist.copy(); hist[:, 0] *= -1
            base_xy = base_xy.copy(); base_xy[0] *= -1
            racket_pos = racket_pos.copy(); racket_pos[0] *= -1
            racket_vel = racket_vel.copy(); racket_vel[0] *= -1
            target = target.copy(); target[0] *= -1

        obs = np.concatenate([
            hist.reshape(-1) * [0.25, 0.5, 0.5] * 6,
            base_xy * [0.25, 0.5],
            racket_pos * [0.25, 0.5, 0.5],
            racket_vel * 0.1,
            target * [0.25, 0.5, 0.5],
            np.array([t_hit, phase]),
        ]).astype(np.float32)
        return obs

    def _policy_swing_action(self, side):
        if self.model is None:
            return None
        obs = self._make_obs(side)
        obs = self._normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action).reshape(-1)

    # ---------- 场景 ----------
    def _build_world(self):
        for path in [self.left_urdf, self.right_urdf]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少 URDF 文件: {path}，请先运行 2.py 生成")
        p.loadURDF("plane.urdf")
        self._add_court()
        self._add_net()

        self.left_robot = p.loadURDF(self.left_urdf, [-2.45, 0, 1.30], [0,0,0,1], useFixedBase=self.fixed_base)
        self.right_robot = p.loadURDF(self.right_urdf, [2.45, 0, 1.30], [0,0,1,0], useFixedBase=self.fixed_base)
        self.left_map = self._make_joint_map(self.left_robot)
        self.right_map = self._make_joint_map(self.right_robot)
        self.left_racket_link = self._find_link(self.left_robot, ["left_racket", "racket", "left_hand"])
        self.right_racket_link = self._find_link(self.right_robot, ["right_racket", "racket", "right_hand"])
        self.shuttle_id = self._create_shuttle()
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

    def _add_court(self):
        green = [0.08, 0.42, 0.18, 1]; green2 = [0.1, 0.5, 0.22, 1]; white = [1,1,1,1]
        def add_box(c, h, rgba):
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=h, rgbaColor=rgba)
            p.createMultiBody(0, baseVisualShapeIndex=v, basePosition=c)
        add_box([0,0,0.003], [4.15, 1.95, 0.003], green)
        add_box([-2.55,0,0.007], [1.35, 1.85, 0.001], green2)
        add_box([2.55,0,0.007], [1.35, 1.85, 0.001], green2)
        def line(pos, size): add_box(pos, size, white)
        line([0,-1.85,0.016],[4,0.018,0.006]); line([0,1.85,0.016],[4,0.018,0.006])
        line([-4,0,0.016],[0.018,1.85,0.006]); line([4,0,0.016],[0.018,1.85,0.006])
        line([0,0,0.018],[0.018,1.85,0.006]); line([-1.2,0,0.018],[0.018,1.85,0.006])
        line([1.2,0,0.018],[0.018,1.85,0.006]); line([-2.7,0,0.018],[0.018,1.85,0.006])
        line([2.7,0,0.018],[0.018,1.85,0.006])

    def _add_net(self):
        h = 1.55
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h/2], rgbaColor=[0.2,0.8,0.2,0.45])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 2.0, h/2])
        p.createMultiBody(0, col, vis, [0,0,h/2])

    def _create_shuttle(self):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.038, height=0.09)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.038, length=0.09, rgbaColor=[0.95,0.95,1,1])
        return p.createMultiBody(0.005, col, vis, [0,0,1.4])

    def _make_joint_map(self, robot):
        return {p.getJointInfo(robot,i)[1].decode():i for i in range(p.getNumJoints(robot))}

    def _find_link(self, robot, names):
        for name in names:
            for i in range(p.getNumJoints(robot)):
                if p.getJointInfo(robot,i)[12].decode() == name:
                    return i
        raise RuntimeError(f"找不到链接: {names}")

    # ---------- 重置 & 发球 ----------
    def reset_rally(self):
        self.rally_count += 1
        self.best_hits = max(self.best_hits, self.hit_count)
        self.step_count = 0
        self.hit_count = 0
        self.last_hit_step = -100
        self.last_hitter = None
        self.prev_left_racket = self.prev_right_racket = None
        self.last_phase_left = self.last_phase_right = 0.0

        p.resetBasePositionAndOrientation(self.left_robot, [-2.45,0,1.30], [0,0,0,1])
        p.resetBasePositionAndOrientation(self.right_robot, [2.45,0,1.30], [0,0,1,0])
        self._reset_neutral(self.left_robot, self.left_map)
        self._reset_neutral(self.right_robot, self.right_map)

        self._serve_to(self.serving)
        self.shuttle_hist.clear()
        for _ in range(6):
            self.shuttle_hist.append(self.shuttle_pos.copy())
        self._update_racket_memory()
        print(f"新回合 {self.rally_count}, 发球方: {self.serving}, 比分 {self.left_score}-{self.right_score}")

    def _reset_neutral(self, robot, jmap):
        neutral = {n: 0.0 if "hip" in n else 0.2 if "knee" in n else -0.06 if "ankle" in n else 0.25 for n in JOINT_NAMES}
        neutral.update({"left_shoulder_pitch":0.25,"right_shoulder_pitch":0.25,"left_elbow":0.35,"right_elbow":0.35})
        for n, v in neutral.items():
            if n in jmap:
                p.resetJointState(robot, jmap[n], v, 0)
                p.setJointMotorControl2(robot, jmap[n], p.POSITION_CONTROL, targetPosition=v, force=180, maxVelocity=6)

    def _serve_to(self, receiver):
        if receiver == "left":
            start = np.array([3.2, self.rng.uniform(-0.55,0.55), self.rng.uniform(1.45,1.75)])
            target = np.array([-1.65, self.rng.uniform(-0.45,0.45), self.rng.uniform(1.38,1.62)])
        else:
            start = np.array([-3.2, self.rng.uniform(-0.55,0.55), self.rng.uniform(1.45,1.75)])
            target = np.array([1.65, self.rng.uniform(-0.45,0.45), self.rng.uniform(1.38,1.62)])
        T = self.rng.uniform(0.95,1.12)
        self.shuttle_pos = start
        self.shuttle_vel = self._ballistic_velocity(start, target, T)
        p.resetBasePositionAndOrientation(self.shuttle_id, start, [0,0,0,1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)

    def _ballistic_velocity(self, start, target, T):
        g = -9.8
        return np.array([(target[0]-start[0])/T, (target[1]-start[1])/T, (target[2]-start[2]-0.5*g*T*T)/T])

    # ---------- 计分 ----------
    def _update_score_and_serve(self):
        if abs(self.shuttle_pos[0]) < 0.06 and self.shuttle_pos[2] < self.net_height:
            if self.last_hitter == "left":
                self.right_score += 1; self.serving = "right"
            elif self.last_hitter == "right":
                self.left_score += 1; self.serving = "left"
            else:
                if self.serving == "left":
                    self.right_score += 1; self.serving = "right"
                else:
                    self.left_score += 1; self.serving = "left"
            return print(f"下网！比分 {self.left_score}-{self.right_score}, 发球方: {self.serving}")
        if abs(self.shuttle_pos[0]) > 4.45 or abs(self.shuttle_pos[1]) > 2.55:
            if self.last_hitter == "left":
                self.right_score += 1; self.serving = "right"
            elif self.last_hitter == "right":
                self.left_score += 1; self.serving = "left"
            else:
                if self.shuttle_vel[0] > 0:
                    self.left_score += 1; self.serving = "left"
                else:
                    self.right_score += 1; self.serving = "right"
            return print(f"出界！比分 {self.left_score}-{self.right_score}, 发球方: {self.serving}")
        if self.shuttle_pos[2] < 0.06:
            if self.shuttle_pos[0] < 0:
                self.right_score += 1; self.serving = "right"
            else:
                self.left_score += 1; self.serving = "left"
            print(f"落地得分！比分 {self.left_score}-{self.right_score}, 发球方: {self.serving}")

    def _rally_done(self):
        if self.shuttle_pos[2] < 0.06 or abs(self.shuttle_pos[0]) > 4.45 or abs(self.shuttle_pos[1]) > 2.55 or (abs(self.shuttle_pos[0])<0.06 and self.shuttle_pos[2]<self.net_height) or self.step_count > self.args.max_steps_per_rally:
            self._update_score_and_serve()
            return True
        return False

    # ---------- 预测轨迹 ----------
    def draw_prediction_trail(self):
        sim_pos, sim_vel = self.shuttle_pos.copy(), self.shuttle_vel.copy()
        points = [sim_pos.copy()]
        for _ in range(30):
            sim_pos, sim_vel = self.shuttle_phys.update(sim_pos, sim_vel, 0.02)
            points.append(sim_pos.copy())
        for i in range(len(points)-1):
            p.addUserDebugLine(points[i], points[i+1], [0.2,0.6,1], 2, 0.3)
        p.addUserDebugText("●", points[-1], [0.2,0.6,1], 0.7, 0.3)

    # ---------- 核心循环 ----------
    def run(self):
        try:
            while p.isConnected():
                receiver = self._receiver_side()
                passive = "right" if receiver=="left" else "left"
                self._apply_active(receiver)
                self._apply_passive(passive)
                for _ in range(int(round(self.control_dt/self.sim_dt))):
                    p.stepSimulation()
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
            pass
        finally:
            p.disconnect()

    # 其余方法请从前文补充完整（如 _receiver_side, _apply_active 等）s