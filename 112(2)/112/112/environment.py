import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import os
from humanoid_controller import HumanoidController
from biped_walk import BipedWalkGenerator
from physics_shuttle import ShuttlePhysics
from urdf_utils import create_urdf_with_fingers

class BadmintonEnv(gym.Env):
    def __init__(self, render=False, train_side='left'):
        super().__init__()
        self.render = render
        self.train_side = train_side
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)
        p.loadURDF("plane.urdf")
        self._add_net()
        
        # 生成带手指的URDF
        os.makedirs("models", exist_ok=True)
        left_path = "models/humanoid_left.urdf"
        right_path = "models/humanoid_right.urdf"
        if not os.path.exists(left_path):
            create_urdf_with_fingers(left_path, [0.2,0.6,0.8,1])
        if not os.path.exists(right_path):
            create_urdf_with_fingers(right_path, [0.8,0.2,0.2,1])
        
        self.left_robot = HumanoidController(left_path, [-2.5,0,0], [0,0,0,1])
        self.right_robot = HumanoidController(right_path, [2.5,0,0], [0,0,1,0])
        
        # 球拍 link 名称
        self.left_racket_link = 'left_racket'
        self.right_racket_link = 'right_racket'
        
        # 步态生成器
        self.walk_left = BipedWalkGenerator(step_length=0.15, step_height=0.06, period=1.0)
        self.walk_right = BipedWalkGenerator(step_length=0.15, step_height=0.06, period=1.0)
        
        # 羽毛球
        self.shuttle_id = self._create_shuttle()
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.012)
        
        # 动作空间: [水平方向偏移(-1~1), 力度(0~1), 手指弯曲(0~1)]
        self.action_space = spaces.Box(low=np.array([-1,0,0]), high=np.array([1,1,1]), dtype=np.float32)
        # 观测空间: 球的位置(3), 球的速度(3), 训练机器人球拍位置(3)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(9,), dtype=np.float32)
        
        self.step_count = 0
        self.last_hit_step = -50
        self.reset()
        
    def _add_net(self):
        net_height = 1.55
        net_width = 6.0
        net_thick = 0.05
        net_center = [0,0,net_height/2]
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[net_thick/2, net_width/2, net_height/2],
                                     rgbaColor=[0.2,0.8,0.2,0.6])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[net_thick/2, net_width/2, net_height/2])
        self.net_id = p.createMultiBody(0, collision, visual, net_center)
        
    def _create_shuttle(self):
        feather_r = 0.038
        feather_h = 0.09
        feather_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=feather_r, height=feather_h)
        feather_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=feather_r, length=feather_h,
                                          rgbaColor=[0.95,0.95,1,1])
        head_r = 0.028
        head_col = p.createCollisionShape(p.GEOM_SPHERE, radius=head_r)
        head_vis = p.createVisualShape(p.GEOM_SPHERE, radius=head_r, rgbaColor=[0.8,0.5,0.2,1])
        shuttle = p.createMultiBody(0.004, feather_col, feather_vis, [0,0,1.2])
        head_body = p.createMultiBody(0.001, head_col, head_vis, [0,0,1.2 + feather_r + head_r])
        p.createConstraint(shuttle, -1, head_body, -1, p.JOINT_FIXED,
                           [0,0,0], [0,0,0], [0,0, feather_r + head_r])
        return shuttle
        
    def reset(self):
        # 随机发球
        self.shuttle_pos = np.array([0.0, 0.0, 1.2])
        side = random.choice([-1,1])
        target_x = random.uniform(1.2,3.0) if side==1 else random.uniform(-3.0,-1.2)
        target_y = random.uniform(-1.5,1.5)
        dx = target_x - self.shuttle_pos[0]
        dy = target_y - self.shuttle_pos[1]
        T = random.uniform(0.8, 1.2)
        vx = dx / T
        vy = dy / T
        g = 9.8
        vz = (0.5*g*T*T - (self.shuttle_pos[2]-0.05))/T
        min_vz = math.sqrt(2*g*(1.55 - self.shuttle_pos[2])) if 1.55 > self.shuttle_pos[2] else 1.0
        vz = max(vz, min_vz+0.5)
        if abs(vx) < 2.0:
            vx = 2.5 if vx>0 else -2.5
        self.shuttle_vel = np.array([vx, vy, vz])
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0,0,0,1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)
        self.step_count = 0
        self.last_hit_step = -50
        # 重置机器人位置
        p.resetBasePositionAndOrientation(self.left_robot.robot_id, [-2.5,0,0], [0,0,0,1])
        p.resetBasePositionAndOrientation(self.right_robot.robot_id, [2.5,0,0], [0,0,1,0])
        self.walk_left.phase = 0
        self.walk_right.phase = 0
        return self._get_obs()
    
    def _get_obs(self):
        if self.train_side == 'left':
            racket_pos = self.left_robot.get_link_position(self.left_racket_link)
        else:
            racket_pos = self.right_robot.get_link_position(self.right_racket_link)
        obs = np.concatenate([self.shuttle_pos, self.shuttle_vel, racket_pos])
        return obs.astype(np.float32)
    
    def step(self, action):
        horiz_dir = action[0]
        power = action[1]
        finger_bend = action[2]
        dt = 1./240.
        
        # 更新步态和移动（主动预测移动，仅演示和学习时都启用）
        self._update_robots_movement(dt)
        
        # 控制手指（仅训练侧）
        finger_joints = []
        if self.train_side == 'left':
            finger_joints = ['left_finger1_joint', 'left_finger2_joint']
            robot = self.left_robot
        else:
            finger_joints = ['right_finger1_joint', 'right_finger2_joint']
            robot = self.right_robot
        for j in finger_joints:
            robot.set_joint_positions({j: finger_bend * 1.2}, velocity=5)
        
        force = None
        reward = 0.0
        
        # 击球检测与奖励
        if self.train_side == 'left' and self.shuttle_pos[0] < 0:
            racket_pos = self.left_robot.get_link_position(self.left_racket_link)
            if self._check_hit(racket_pos):
                force = self._compute_hit_force(racket_pos, horiz_dir, power)
                self.last_hit_step = self.step_count
                reward += 1.0   # 成功击球
        elif self.train_side == 'right' and self.shuttle_pos[0] > 0:
            racket_pos = self.right_robot.get_link_position(self.right_racket_link)
            if self._check_hit(racket_pos):
                force = self._compute_hit_force(racket_pos, horiz_dir, power)
                self.last_hit_step = self.step_count
                reward += 1.0
        
        # 更新羽毛球物理
        self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(self.shuttle_pos, self.shuttle_vel, dt, force)
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0,0,0,1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)
        p.stepSimulation()
        self.step_count += 1
        
        # 终止条件与额外奖励
        done = False
        if self.shuttle_pos[2] < 0.08:
            done = True
            reward -= 0.5   # 落地惩罚
        if (self.train_side == 'left' and self.shuttle_pos[0] > 0) or \
           (self.train_side == 'right' and self.shuttle_pos[0] < 0):
            reward += 0.2   # 球过网到对方半场
        if self.step_count - self.last_hit_step > 500:
            done = True
            reward -= 1.0   # 长时间不击球
        
        obs = self._get_obs()
        return obs, reward, done, {}
    
    def _check_hit(self, racket_pos):
        dist = np.linalg.norm(self.shuttle_pos - racket_pos)
        if dist < 0.25 and self.shuttle_vel[2] < 0 and self.shuttle_pos[2] < 0.8:
            if self.step_count - self.last_hit_step > 35:
                return True
        return False
    
    def _compute_hit_force(self, hit_pos, horiz_dir, power):
        speed_base = 8.0 + power * 8.0
        if self.train_side == 'left':
            vx = speed_base * 0.9
            vy = horiz_dir * speed_base * 0.5
        else:
            vx = -speed_base * 0.9
            vy = horiz_dir * speed_base * 0.5
        # 确保过网速度
        g = 9.8
        dx_net = 0 - hit_pos[0]
        if abs(dx_net) < 0.01:
            dx_net = 0.01
        t_net = dx_net / vx
        if t_net <= 0:
            t_net = 0.1
        vz_min = (1.55 - hit_pos[2] + 0.5*g*t_net**2) / t_net
        vz = max(vz_min, 2.0 + power * 5.0)
        target_vel = np.array([vx, vy, vz])
        mass = 0.005
        dt = 1./240.
        force = mass * (target_vel - self.shuttle_vel) / dt
        return force
    
    def _update_robots_movement(self, dt):
        # 更新步态相位
        self.walk_left.update_phase(dt)
        self.walk_right.update_phase(dt)
        # 为左右腿设置关节角度（简单站立摆动）
        for robot, walk, side in [(self.left_robot, self.walk_left, 'left'),
                                  (self.right_robot, self.walk_right, 'right')]:
            hip = walk.get_leg_angle(side, 'hip')
            knee = walk.get_knee_angle()
            ankle = -hip * 0.5
            robot.set_joint_positions({f"{side}_hip": hip, f"{side}_knee": knee, f"{side}_ankle": ankle})
            other = 'right' if side=='left' else 'left'
            o_hip = walk.get_leg_angle(other, 'hip')
            o_knee = walk.get_knee_angle()
            o_ankle = -o_hip * 0.5
            robot.set_joint_positions({f"{other}_hip": o_hip, f"{other}_knee": o_knee, f"{other}_ankle": o_ankle})
        
        # 主动移动：根据预测落点移动机器人（简化）
        target = self._predict_future_position(self.shuttle_pos, self.shuttle_vel, 0.5)
        if self.train_side == 'left':
            target[0] = np.clip(target[0], -4.0, -0.3)
            self._move_robot_to_target(self.left_robot, target)
        else:
            target[0] = np.clip(target[0], 0.3, 4.0)
            self._move_robot_to_target(self.right_robot, target)
    
    def _predict_future_position(self, pos, vel, dt_pred=0.5):
        g = 9.8
        future_pos = pos + vel * dt_pred
        future_pos[2] = pos[2] + vel[2] * dt_pred - 0.5 * g * dt_pred**2
        return future_pos
    
    def _move_robot_to_target(self, robot, target_pos, speed=1.2):
        current = robot.get_position()
        dx = target_pos[0] - current[0]
        dy = target_pos[1] - current[1]
        step = min(1.0, speed * (1./240.) * 2)
        new_x = current[0] + np.clip(dx, -step, step)
        new_y = current[1] + np.clip(dy, -step, step)
        if robot == self.left_robot:
            new_x = np.clip(new_x, -4.0, -0.3)
        else:
            new_x = np.clip(new_x, 0.3, 4.0)
        p.resetBasePositionAndOrientation(robot.robot_id, [new_x, new_y, current[2]], [0,0,0,1])
    
    def close(self):
        p.disconnect()