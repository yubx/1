#!/usr/bin/env python3
"""
双人形机器人羽毛球对打 - 高度拟人化（球形头盔头部、大眼、圆柱四肢、宽肩窄腰）
主动预测移动 + 明显挥手击球动作
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import random
import math

from humanoid_controller import HumanoidController
from biped_walk import BipedWalkGenerator
from physics_shuttle import ShuttlePhysics

class HumanoidBadmintonGame:
    def __init__(self):
        self.client = p.connect(p.GUI)
        if self.client < 0:
            print("GUI failed, fallback to DIRECT")
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        
        p.loadURDF("plane.urdf")
        self._add_net()
        
        self._ensure_urdf_exists()
        
        self.left_robot = HumanoidController("models/humanoid_left.urdf", [-2.5, 0, 0], [0,0,0,1])
        self.right_robot = HumanoidController("models/humanoid_right.urdf", [2.5, 0, 0], [0,0,1,0])
        
        # 获取球拍link索引
        self.left_racket_link = self._get_link_index(self.left_robot.robot_id, "left_racket")
        self.right_racket_link = self._get_link_index(self.right_robot.robot_id, "right_racket")
        if self.left_racket_link < 0:
            self.left_racket_link = self._get_link_index(self.left_robot.robot_id, "racket")
        if self.right_racket_link < 0:
            self.right_racket_link = self._get_link_index(self.right_robot.robot_id, "racket")
        
        # 创建羽毛球（真实视觉）
        self.shuttle_id = self._create_shuttle()
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.012)
        self.shuttle_pos = np.array([0.0, 0.0, 1.2])
        self.shuttle_vel = np.array([0.0, 0.0, 0.0])
        
        # 行走生成器
        self.walk_left = BipedWalkGenerator(step_length=0.15, step_height=0.06, period=1.0)
        self.walk_right = BipedWalkGenerator(step_length=0.15, step_height=0.06, period=1.0)
        
        self.last_hit_step = -50
        self.step = 0
        self.swing_left = False
        self.swing_right = False
        self.swing_timer = 0
        
        self._random_serve_with_over_net()
        print("初始化完成：高度拟人化机器人（球形头盔、大眼、圆柱四肢、宽肩窄腰）")
        
    def _add_net(self):
        net_height = 1.55
        net_width = 6.0
        net_thick = 0.05
        net_center = [0, 0, net_height/2]
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
        head_body = p.createMultiBody(0.001, head_col, head_vis, [0,0,1.2 + feather_h/2 + head_r])
        p.createConstraint(shuttle, -1, head_body, -1, p.JOINT_FIXED,
                           [0,0,0], [0,0,0], [0,0, feather_h/2 + head_r])
        return shuttle
        
    def _get_link_index(self, robot_id, link_name):
        for i in range(p.getNumJoints(robot_id)):
            if p.getJointInfo(robot_id, i)[12].decode('utf-8') == link_name:
                return i
        return -1
        
    def _ensure_urdf_exists(self):
        os.makedirs("models", exist_ok=True)
        left_path = "models/humanoid_left.urdf"
        right_path = "models/humanoid_right.urdf"
        if not os.path.exists(left_path):
            self._create_urdf(left_path, [0.2,0.6,0.8,1])   # 蓝色衣服
        if not os.path.exists(right_path):
            self._create_urdf(right_path, [0.8,0.2,0.2,1])  # 红色衣服
            
    def _create_urdf(self, path, color):
        """高度拟人化URDF：球形头盔头部、大眼、圆柱四肢、宽肩窄腰、腰带"""
        color_str = f"{color[0]} {color[1]} {color[2]} {color[3]}"
        skin_color = "0.9 0.85 0.7 1"
        eye_white = "1 1 1 1"
        eye_black = "0 0 0 1"
        mouth_color = "0.6 0.2 0.1 1"
        belt_color = "0.2 0.2 0.2 1"
        shoulder_color = "0.5 0.5 0.5 1"
        
        urdf = f'''<?xml version="1.0" ?>
<robot name="humanoid">
  <material name="cloth"><color rgba="{color_str}"/></material>
  <material name="skin"><color rgba="{skin_color}"/></material>
  <material name="eye_white"><color rgba="{eye_white}"/></material>
  <material name="eye_black"><color rgba="{eye_black}"/></material>
  <material name="mouth"><color rgba="{mouth_color}"/></material>
  <material name="belt"><color rgba="{belt_color}"/></material>
  <material name="shoulder_pad"><color rgba="{shoulder_color}"/></material>
  <material name="racket_mat"><color rgba="0.9 0.7 0.2 1"/></material>
  
  <!-- 头部：球形头盔 -->
  <link name="head">
    <inertial><mass value="1.5"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.015"/></inertial>
    <visual><geometry><sphere radius="0.16"/></geometry><material name="skin"/></visual>
    <collision><geometry><sphere radius="0.16"/></geometry></collision>
  </link>
  
  <!-- 左眼（加大） -->
  <link name="left_eye_white">
    <inertial><mass value="0.015"/><inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial>
    <visual><geometry><sphere radius="0.055"/></geometry><material name="eye_white"/></visual>
    <collision><geometry><sphere radius="0.055"/></geometry></collision>
  </link>
  <joint name="left_eye_joint" type="fixed"><parent link="head"/><child link="left_eye_white"/><origin xyz="0.08 0.1 0.12"/></joint>
  <link name="left_pupil">
    <inertial><mass value="0.008"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><geometry><sphere radius="0.025"/></geometry><material name="eye_black"/></visual>
    <collision><geometry><sphere radius="0.025"/></geometry></collision>
  </link>
  <joint name="left_pupil_joint" type="fixed"><parent link="left_eye_white"/><child link="left_pupil"/><origin xyz="0.04 0 0.03"/></joint>
  
  <!-- 右眼 -->
  <link name="right_eye_white">
    <inertial><mass value="0.015"/><inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial>
    <visual><geometry><sphere radius="0.055"/></geometry><material name="eye_white"/></visual>
    <collision><geometry><sphere radius="0.055"/></geometry></collision>
  </link>
  <joint name="right_eye_joint" type="fixed"><parent link="head"/><child link="right_eye_white"/><origin xyz="-0.08 0.1 0.12"/></joint>
  <link name="right_pupil">
    <inertial><mass value="0.008"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><geometry><sphere radius="0.025"/></geometry><material name="eye_black"/></visual>
    <collision><geometry><sphere radius="0.025"/></geometry></collision>
  </link>
  <joint name="right_pupil_joint" type="fixed"><parent link="right_eye_white"/><child link="right_pupil"/><origin xyz="-0.04 0 0.03"/></joint>
  
  <!-- 嘴巴（弧形，用盒子模拟） -->
  <link name="mouth">
    <inertial><mass value="0.008"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><geometry><box size="0.07 0.025 0.025"/></geometry><material name="mouth"/></visual>
    <collision><geometry><box size="0.07 0.025 0.025"/></geometry></collision>
  </link>
  <joint name="mouth_joint" type="fixed"><parent link="head"/><child link="mouth"/><origin xyz="0 0.06 0.08"/></joint>
  
  <!-- 颈部（圆柱） -->
  <link name="neck">
    <inertial><mass value="0.8"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <visual><geometry><cylinder radius="0.09" length="0.12"/></geometry><material name="skin"/></visual>
    <collision><geometry><cylinder radius="0.09" length="0.12"/></geometry></collision>
  </link>
  <joint name="neck_joint" type="revolute">
    <parent link="head"/><child link="neck"/>
    <origin xyz="0 0 -0.15"/><axis xyz="0 0 1"/>
    <limit lower="-0.3" upper="0.3" effort="20" velocity="1"/>
  </joint>
  
  <!-- 胸部（宽肩） -->
  <link name="chest">
    <inertial><mass value="8"/><inertia ixx="0.12" ixy="0" ixz="0" iyy="0.12" iyz="0" izz="0.12"/></inertial>
    <visual><geometry><box size="0.44 0.36 0.48"/></geometry><material name="cloth"/></visual>
    <collision><geometry><box size="0.44 0.36 0.48"/></geometry></collision>
  </link>
  <joint name="chest_neck" type="fixed"><parent link="neck"/><child link="chest"/><origin xyz="0 0 -0.2"/></joint>
  
  <!-- 髋部（窄腰） -->
  <link name="hip">
    <inertial><mass value="5"/><inertia ixx="0.09" ixy="0" ixz="0" iyy="0.09" iyz="0" izz="0.09"/></inertial>
    <visual><geometry><box size="0.38 0.32 0.28"/></geometry><material name="cloth"/></visual>
    <collision><geometry><box size="0.38 0.32 0.28"/></geometry></collision>
  </link>
  <joint name="hip_chest" type="fixed"><parent link="chest"/><child link="hip"/><origin xyz="0 0 -0.34"/></joint>
  
  <!-- 腰带视觉 -->
  <link name="belt">
    <inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><geometry><box size="0.42 0.34 0.05"/></geometry><material name="belt"/></visual>
    <collision><geometry><box size="0.42 0.34 0.05"/></geometry></collision>
  </link>
  <joint name="belt_joint" type="fixed"><parent link="hip"/><child link="belt"/><origin xyz="0 0 0.12"/></joint>
  
  <!-- ========== 左腿 ========== -->
  <link name="left_thigh">
    <inertial><mass value="3.5"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/></inertial>
    <visual><geometry><cylinder radius="0.09" length="0.36"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.09" length="0.36"/></geometry></collision>
  </link>
  <joint name="left_hip" type="revolute">
    <parent link="hip"/><child link="left_thigh"/>
    <origin xyz="-0.14 -0.14 -0.14"/><axis xyz="1 0 0"/>
    <limit lower="-0.9" upper="0.9" effort="120" velocity="2"/>
  </joint>
  <link name="left_calf">
    <inertial><mass value="2.5"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    <visual><geometry><cylinder radius="0.075" length="0.38"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.075" length="0.38"/></geometry></collision>
  </link>
  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/><child link="left_calf"/>
    <origin xyz="0 0 -0.18"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1.3" effort="100" velocity="2"/>
  </joint>
  <link name="left_foot">
    <inertial><mass value="0.8"/><inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.008"/></inertial>
    <visual><geometry><box size="0.14 0.24 0.1"/></geometry><material name="cloth"/></visual>
    <collision><geometry><box size="0.14 0.24 0.1"/></geometry></collision>
  </link>
  <joint name="left_ankle" type="revolute">
    <parent link="left_calf"/><child link="left_foot"/>
    <origin xyz="0 0 -0.19"/><axis xyz="1 0 0"/>
    <limit lower="-0.4" upper="0.4" effort="60" velocity="2"/>
  </joint>
  
  <!-- 右腿 -->
  <link name="right_thigh">
    <inertial><mass value="3.5"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/></inertial>
    <visual><geometry><cylinder radius="0.09" length="0.36"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.09" length="0.36"/></geometry></collision>
  </link>
  <joint name="right_hip" type="revolute">
    <parent link="hip"/><child link="right_thigh"/>
    <origin xyz="-0.14 0.14 -0.14"/><axis xyz="1 0 0"/>
    <limit lower="-0.9" upper="0.9" effort="120" velocity="2"/>
  </joint>
  <link name="right_calf">
    <inertial><mass value="2.5"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    <visual><geometry><cylinder radius="0.075" length="0.38"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.075" length="0.38"/></geometry></collision>
  </link>
  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/><child link="right_calf"/>
    <origin xyz="0 0 -0.18"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1.3" effort="100" velocity="2"/>
  </joint>
  <link name="right_foot">
    <inertial><mass value="0.8"/><inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.008"/></inertial>
    <visual><geometry><box size="0.14 0.24 0.1"/></geometry><material name="cloth"/></visual>
    <collision><geometry><box size="0.14 0.24 0.1"/></geometry></collision>
  </link>
  <joint name="right_ankle" type="revolute">
    <parent link="right_calf"/><child link="right_foot"/>
    <origin xyz="0 0 -0.19"/><axis xyz="1 0 0"/>
    <limit lower="-0.4" upper="0.4" effort="60" velocity="2"/>
  </joint>
  
  <!-- ========== 左臂 ========== -->
  <!-- 肩垫装饰 -->
  <link name="left_shoulder_pad">
    <inertial><mass value="0.3"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual><geometry><box size="0.12 0.12 0.08"/></geometry><material name="shoulder_pad"/></visual>
    <collision><geometry><box size="0.12 0.12 0.08"/></geometry></collision>
  </link>
  <joint name="left_shoulder_pad_joint" type="fixed"><parent link="chest"/><child link="left_shoulder_pad"/><origin xyz="0.26 -0.22 0.22"/></joint>
  
  <link name="left_upper_arm">
    <inertial><mass value="1.5"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.015"/></inertial>
    <visual><geometry><cylinder radius="0.075" length="0.32"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.075" length="0.32"/></geometry></collision>
  </link>
  <joint name="left_shoulder_pitch_joint" type="revolute">
    <parent link="chest"/><child link="left_upper_arm"/>
    <origin xyz="0.24 -0.2 0.22"/><axis xyz="0 1 0"/>
    <limit lower="-1.6" upper="1.8" effort="100" velocity="3"/>
  </joint>
  <link name="left_forearm">
    <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><cylinder radius="0.065" length="0.3"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.065" length="0.3"/></geometry></collision>
  </link>
  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/><child link="left_forearm"/>
    <origin xyz="0 0 -0.16"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="1.8" effort="80" velocity="3"/>
  </joint>
  <link name="left_hand">
    <inertial><mass value="0.4"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/></inertial>
    <visual><geometry><box size="0.1 0.12 0.08"/></geometry><material name="skin"/></visual>
    <collision><geometry><box size="0.1 0.12 0.08"/></geometry></collision>
  </link>
  <joint name="left_wrist" type="revolute">
    <parent link="left_forearm"/><child link="left_hand"/>
    <origin xyz="0 0 -0.15"/><axis xyz="0 1 0"/>
    <limit lower="-0.8" upper="0.8" effort="60" velocity="3"/>
  </joint>
  <link name="left_racket">
    <inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><geometry><box size="0.05 0.24 0.01"/></geometry><material name="racket_mat"/></visual>
    <collision><geometry><box size="0.05 0.24 0.01"/></geometry></collision>
  </link>
  <joint name="left_racket_joint" type="fixed"><parent link="left_hand"/><child link="left_racket"/><origin xyz="0 0 -0.12"/></joint>
  
  <!-- 右臂 -->
  <link name="right_shoulder_pad">
    <inertial><mass value="0.3"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual><geometry><box size="0.12 0.12 0.08"/></geometry><material name="shoulder_pad"/></visual>
    <collision><geometry><box size="0.12 0.12 0.08"/></geometry></collision>
  </link>
  <joint name="right_shoulder_pad_joint" type="fixed"><parent link="chest"/><child link="right_shoulder_pad"/><origin xyz="0.26 0.22 0.22"/></joint>
  
  <link name="right_upper_arm">
    <inertial><mass value="1.5"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.015"/></inertial>
    <visual><geometry><cylinder radius="0.075" length="0.32"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.075" length="0.32"/></geometry></collision>
  </link>
  <joint name="right_shoulder_pitch_joint" type="revolute">
    <parent link="chest"/><child link="right_upper_arm"/>
    <origin xyz="0.24 0.2 0.22"/><axis xyz="0 1 0"/>
    <limit lower="-1.6" upper="1.8" effort="100" velocity="3"/>
  </joint>
  <link name="right_forearm">
    <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><cylinder radius="0.065" length="0.3"/></geometry><material name="cloth"/></visual>
    <collision><geometry><cylinder radius="0.065" length="0.3"/></geometry></collision>
  </link>
  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/><child link="right_forearm"/>
    <origin xyz="0 0 -0.16"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="1.8" effort="80" velocity="3"/>
  </joint>
  <link name="right_hand">
    <inertial><mass value="0.4"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/></inertial>
    <visual><geometry><box size="0.1 0.12 0.08"/></geometry><material name="skin"/></visual>
    <collision><geometry><box size="0.1 0.12 0.08"/></geometry></collision>
  </link>
  <joint name="right_wrist" type="revolute">
    <parent link="right_forearm"/><child link="right_hand"/>
    <origin xyz="0 0 -0.15"/><axis xyz="0 1 0"/>
    <limit lower="-0.8" upper="0.8" effort="60" velocity="3"/>
  </joint>
  <link name="right_racket">
    <inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><geometry><box size="0.05 0.24 0.01"/></geometry><material name="racket_mat"/></visual>
    <collision><geometry><box size="0.05 0.24 0.01"/></geometry></collision>
  </link>
  <joint name="right_racket_joint" type="fixed"><parent link="right_hand"/><child link="right_racket"/><origin xyz="0 0 -0.12"/></joint>
</robot>'''
        with open(path, "w") as f:
            f.write(urdf)
        print(f"高度拟人化URDF生成: {path}")
        
    def predict_future_position(self, pos, vel, dt_pred=0.5):
        g = 9.8
        future_pos = pos + vel * dt_pred
        future_pos[2] = pos[2] + vel[2] * dt_pred - 0.5 * g * dt_pred**2
        return future_pos
        
    def move_robot_to_target(self, robot, target_pos, speed=1.2):
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
        
    def get_racket_position(self, robot_id, link_idx):
        if link_idx >= 0:
            pos, _ = p.getLinkState(robot_id, link_idx)[:2]
            return np.array(pos)
        return np.array([0,0,0])
        
    def update_robots_movement(self, dt):
        # 腿部摆动 + 脚踝补偿
        self.walk_left.update_phase(dt)
        self.walk_right.update_phase(dt)
        for robot, walk, side in [(self.left_robot, self.walk_left, 'left'), (self.right_robot, self.walk_right, 'right')]:
            hip = walk.get_leg_angle(side, 'hip')
            knee = walk.get_knee_angle()
            ankle = -hip * 0.4
            robot.set_joint_positions({f"{side}_hip": hip, f"{side}_knee": knee, f"{side}_ankle": ankle})
            other = 'right' if side=='left' else 'left'
            o_hip = walk.get_leg_angle(other, 'hip')
            o_knee = walk.get_knee_angle()
            o_ankle = -o_hip * 0.4
            robot.set_joint_positions({f"{other}_hip": o_hip, f"{other}_knee": o_knee, f"{other}_ankle": o_ankle})
        
        # 主动移动
        if self.shuttle_pos[0] < 0:
            target = self.predict_future_position(self.shuttle_pos, self.shuttle_vel, 0.5)
            target[0] = np.clip(target[0], -4.0, -0.2)
            target[1] = np.clip(target[1], -2.0, 2.0)
            self.move_robot_to_target(self.left_robot, target)
        else:
            target = self.predict_future_position(self.shuttle_pos, self.shuttle_vel, 0.5)
            target[0] = np.clip(target[0], 0.2, 4.0)
            target[1] = np.clip(target[1], -2.0, 2.0)
            self.move_robot_to_target(self.right_robot, target)
            
    def check_and_hit(self, robot, racket_pos, shuttle_pos, shuttle_vel, robot_side):
        if robot_side == 'left':
            if shuttle_pos[0] > -0.3: return None
        else:
            if shuttle_pos[0] < 0.3: return None
        if abs(shuttle_pos[0] - racket_pos[0]) > 0.15:
            return None
        if shuttle_pos[2] < 0.6 and shuttle_vel[2] < 0:
            dist = np.linalg.norm(racket_pos - shuttle_pos)
            if dist < 0.3 and (self.step - self.last_hit_step) > 35:
                if robot_side == 'left':
                    self.swing_left = True
                else:
                    self.swing_right = True
                self.swing_timer = 0
                if robot_side == 'left':
                    horiz_dir = np.array([1.0, random.uniform(-0.5,0.5)])
                else:
                    horiz_dir = np.array([-1.0, random.uniform(-0.5,0.5)])
                target_vel = self.ensure_over_net_velocity(racket_pos, horiz_dir)
                desired_speed = 12.0
                current = np.linalg.norm(target_vel)
                if current > 0:
                    target_vel = target_vel / current * desired_speed
                mass = 0.005
                dt = 1./240.
                force = mass * (target_vel - shuttle_vel) / dt
                self.last_hit_step = self.step
                print(f"{robot_side.upper()} 击球！移动到位，挥拍！ 速度={target_vel}")
                return force
        return None
        
    def ensure_over_net_velocity(self, hit_pos, target_dir_2d):
        g = 9.8
        horiz_speed = max(2.5, np.linalg.norm(target_dir_2d[:2]))
        horiz_dir = target_dir_2d[:2] / (np.linalg.norm(target_dir_2d[:2])+1e-6)
        vx = horiz_dir[0] * horiz_speed
        vy = horiz_dir[1] * horiz_speed
        dx_net = 0 - hit_pos[0]
        if abs(dx_net) < 0.01:
            dx_net = 0.01
        t_net = dx_net / vx
        if t_net <= 0:
            t_net = 0.1
        vz_min = (1.55 - hit_pos[2] + 0.5*g*t_net**2) / t_net
        vz = max(vz_min, 2.0)
        return np.array([vx, vy, vz])
        
    def update_swing_animation(self, dt):
        # 左臂挥拍
        if self.swing_left:
            self.swing_timer += dt
            if self.swing_timer < 0.1:
                t = self.swing_timer / 0.1
                pitch = 0.2 + t * 1.0
                elbow = 0.2 + t * 1.3
                wrist = 0.0 + t * 0.8
                self.left_robot.set_joint_positions({"left_shoulder_pitch_joint": pitch, "left_elbow": elbow, "left_wrist": wrist},
                                                    velocity=10, force=200)
            elif self.swing_timer < 0.25:
                t2 = (self.swing_timer - 0.1) / 0.15
                pitch = 1.2 * (1 - t2)
                elbow = 1.5 * (1 - t2)
                wrist = 0.8 * (1 - t2)
                self.left_robot.set_joint_positions({"left_shoulder_pitch_joint": pitch, "left_elbow": elbow, "left_wrist": wrist},
                                                    velocity=5, force=100)
            else:
                self.left_robot.set_joint_positions({"left_shoulder_pitch_joint": 0.2, "left_elbow": 0.2, "left_wrist": 0.0},
                                                    velocity=2, force=50)
                self.swing_left = False
        # 右臂挥拍
        if self.swing_right:
            self.swing_timer += dt
            if self.swing_timer < 0.1:
                t = self.swing_timer / 0.1
                pitch = 0.2 + t * 1.0
                elbow = 0.2 + t * 1.3
                wrist = 0.0 + t * 0.8
                self.right_robot.set_joint_positions({"right_shoulder_pitch_joint": pitch, "right_elbow": elbow, "right_wrist": wrist},
                                                     velocity=10, force=200)
            elif self.swing_timer < 0.25:
                t2 = (self.swing_timer - 0.1) / 0.15
                pitch = 1.2 * (1 - t2)
                elbow = 1.5 * (1 - t2)
                wrist = 0.8 * (1 - t2)
                self.right_robot.set_joint_positions({"right_shoulder_pitch_joint": pitch, "right_elbow": elbow, "right_wrist": wrist},
                                                     velocity=5, force=100)
            else:
                self.right_robot.set_joint_positions({"right_shoulder_pitch_joint": 0.2, "right_elbow": 0.2, "right_wrist": 0.0},
                                                     velocity=2, force=50)
                self.swing_right = False
                
    def _random_serve_with_over_net(self):
        start = np.array([0.0, 0.0, 1.2])
        side = random.choice([-1, 1])
        target_x = random.uniform(1.2, 3.0) if side==1 else random.uniform(-3.0, -1.2)
        target_y = random.uniform(-1.5, 1.5)
        dx = target_x - start[0]
        dy = target_y - start[1]
        T = random.uniform(0.8, 1.2)
        vx = dx / T
        vy = dy / T
        g = 9.8
        vz = (0.5*g*T*T - (start[2]-0.05))/T
        min_vz = math.sqrt(2*g*(1.55 - start[2])) if 1.55 > start[2] else 1.0
        vz = max(vz, min_vz+0.5)
        if abs(vx) < 2.0:
            vx = 2.5 if vx>0 else -2.5
        self.shuttle_vel = np.array([vx, vy, vz])
        self.shuttle_pos = start.copy()
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0,0,0,1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)
        print(f"发球: 目标({target_x:.1f},{target_y:.1f}) 初速({vx:.1f},{vy:.1f},{vz:.1f})")
        
    def run(self):
        last_time = time.time()
        try:
            while True:
                now = time.time()
                dt = min(0.033, now - last_time)
                last_time = now
                if dt <= 0: dt = 0.016
                
                self.update_robots_movement(dt)
                self.update_swing_animation(dt)
                
                left_racket = self.get_racket_position(self.left_robot.robot_id, self.left_racket_link)
                right_racket = self.get_racket_position(self.right_robot.robot_id, self.right_racket_link)
                
                # 手臂预摆：无偏航关节，直接使用肩俯仰对准（简单处理）
                # 为了拟人，不做复杂偏航，让球拍自然跟随
                
                force = None
                if self.shuttle_pos[0] < 0:
                    force = self.check_and_hit(self.left_robot, left_racket, self.shuttle_pos, self.shuttle_vel, 'left')
                else:
                    force = self.check_and_hit(self.right_robot, right_racket, self.shuttle_pos, self.shuttle_vel, 'right')
                
                self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(self.shuttle_pos, self.shuttle_vel, dt, force)
                if self.shuttle_pos[2] < 0.08:
                    print("落地，重新发球")
                    self._random_serve_with_over_net()
                    self.last_hit_step = -50
                    
                p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0,0,0,1])
                p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)
                p.stepSimulation()
                self.step += 1
                time.sleep(1./240.)
        except KeyboardInterrupt:
            print("退出")
        finally:
            p.disconnect()

if __name__ == "__main__":
    game = HumanoidBadmintonGame()
    game.run()