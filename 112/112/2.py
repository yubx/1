#!/usr/bin/env python3


import math
import os
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

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
        p.setTimeStep(1.0 / 240.0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=6.0,
            cameraYaw=35,
            cameraPitch=-25,
            cameraTargetPosition=[0.0, 0.0, 1.0],
        )

        p.loadURDF("plane.urdf")
        self._add_court_lines()
        self._add_net()

        self._ensure_urdf_exists()

        # 这个模型的 torso 是 base link，所以 z 不能是 0，否则会穿地。
        self.left_robot = HumanoidController(
            "models/humanoid_left.urdf",
            [-2.5, 0.0, 1.30],
            [0, 0, 0, 1],
        )
        self.right_robot = HumanoidController(
            "models/humanoid_right.urdf",
            [2.5, 0.0, 1.30],
            [0, 0, 1, 0],
        )

        self.left_racket_link = self._find_first_link(
            self.left_robot.robot_id,
            ["left_racket", "racket", "left_hand"],
        )
        self.right_racket_link = self._find_first_link(
            self.right_robot.robot_id,
            ["right_racket", "racket", "right_hand"],
        )

        print("Left racket link:", self.left_racket_link)
        print("Right racket link:", self.right_racket_link)

        self.shuttle_id = self._create_shuttle()
        self.shuttle_phys = ShuttlePhysics(g=-9.8, drag_coeff=0.012)
        self.shuttle_pos = np.array([0.0, 0.0, 1.2], dtype=float)
        self.shuttle_vel = np.array([0.0, 0.0, 0.0], dtype=float)

        self.walk_left = BipedWalkGenerator(step_length=0.15, step_height=0.06, period=1.0)
        self.walk_right = BipedWalkGenerator(step_length=0.15, step_height=0.06, period=1.0)

        self.last_hit_step = -80
        self.step = 0

        self.swing_left = False
        self.swing_right = False
        self.swing_timer_left = 0.0
        self.swing_timer_right = 0.0

        # 记录球拍上一帧位置，用来估计球拍速度。
        # 后面击球不再只靠“球接近身体”的规则，而是要求球拍真的运动到球附近。
        self.prev_left_racket_pos = None
        self.prev_right_racket_pos = None
        self.left_racket_vel = np.zeros(3, dtype=float)
        self.right_racket_vel = np.zeros(3, dtype=float)

        self._set_neutral_pose()
        self._random_serve_with_over_net()
        print("初始化完成：改进版人形机器人，手掌和球拍更明显。")

    # ------------------------------------------------------------------
    # 场地与物体
    # ------------------------------------------------------------------

    def _add_court_lines(self):
        """在地面画完整羽毛球场格子。

        坐标约定：
        - x 方向是左右半场长度方向，球网在 x=0。
        - y 方向是场地宽度方向。
        - 这里用 scale 缩放到当前仿真尺寸，避免场地过大。
        """
        scale = 0.60
        line_w = 0.035
        line_h = 0.004
        z = 0.006
        white = [1.0, 1.0, 1.0, 1.0]
        court_green = [0.08, 0.42, 0.18, 1.0]
        service_green = [0.10, 0.50, 0.22, 1.0]

        # 标准羽毛球尺寸近似值，按 scale 缩放。
        half_len = 6.70 * scale
        half_doubles = 3.05 * scale
        half_singles = 2.59 * scale
        short_service = 1.98 * scale
        long_service_doubles = 5.94 * scale

        def add_box(center, half_extents, rgba):
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=rgba,
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=center,
            )

        def add_line_x(x1, x2, y):
            cx = (x1 + x2) / 2.0
            lx = abs(x2 - x1) / 2.0
            add_box([cx, y, z + 0.002], [lx, line_w / 2.0, line_h], white)

        def add_line_y(x, y1, y2):
            cy = (y1 + y2) / 2.0
            ly = abs(y2 - y1) / 2.0
            add_box([x, cy, z + 0.002], [line_w / 2.0, ly, line_h], white)

        # 绿色地胶底色。
        add_box(
            [0, 0, z / 2.0],
            [half_len + 0.15, half_doubles + 0.15, z / 2.0],
            court_green,
        )

        # 左右发球区稍微加一点色差。
        add_box(
            [-(short_service + half_len) / 2.0, 0, z + 0.001],
            [(half_len - short_service) / 2.0, half_doubles, 0.001],
            service_green,
        )
        add_box(
            [(short_service + half_len) / 2.0, 0, z + 0.001],
            [(half_len - short_service) / 2.0, half_doubles, 0.001],
            service_green,
        )

        # 1. 双打外边界。
        add_line_x(-half_len, half_len, -half_doubles)
        add_line_x(-half_len, half_len, half_doubles)
        add_line_y(-half_len, -half_doubles, half_doubles)
        add_line_y(half_len, -half_doubles, half_doubles)

        # 2. 单打边线。
        add_line_x(-half_len, half_len, -half_singles)
        add_line_x(-half_len, half_len, half_singles)

        # 3. 前发球线。
        add_line_y(-short_service, -half_doubles, half_doubles)
        add_line_y(short_service, -half_doubles, half_doubles)

        # 4. 双打后发球线。
        add_line_y(-long_service_doubles, -half_doubles, half_doubles)
        add_line_y(long_service_doubles, -half_doubles, half_doubles)

        # 5. 发球区中线。注意不是整场贯穿，只从前发球线到底线。
        add_line_x(-half_len, -short_service, 0.0)
        add_line_x(short_service, half_len, 0.0)

        # 6. 网下中线，用淡灰色辅助观察。
        add_box(
            [0, 0, z + 0.003],
            [line_w / 2.0, half_doubles, line_h],
            [0.8, 0.8, 0.8, 0.45],
        )

    def _add_net(self):
        net_height = 1.55
        net_width = 4.0
        net_thick = 0.04
        net_center = [0, 0, net_height / 2]

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[net_thick / 2, net_width / 2, net_height / 2],
            rgbaColor=[0.2, 0.8, 0.2, 0.45],
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[net_thick / 2, net_width / 2, net_height / 2],
        )
        self.net_id = p.createMultiBody(0, collision, visual, net_center)

    def _create_shuttle(self):
        feather_r = 0.038
        feather_h = 0.09
        feather_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=feather_r, height=feather_h)
        feather_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=feather_r,
            length=feather_h,
            rgbaColor=[0.95, 0.95, 1.0, 1.0],
        )

        head_r = 0.028
        head_col = p.createCollisionShape(p.GEOM_SPHERE, radius=head_r)
        head_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=head_r,
            rgbaColor=[0.8, 0.5, 0.2, 1.0],
        )

        shuttle = p.createMultiBody(0.004, feather_col, feather_vis, [0, 0, 1.2])
        head_body = p.createMultiBody(
            0.001,
            head_col,
            head_vis,
            [0, 0, 1.2 + feather_h / 2 + head_r],
        )
        p.createConstraint(
            shuttle,
            -1,
            head_body,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, feather_h / 2 + head_r],
        )
        return shuttle

    # ------------------------------------------------------------------
    # URDF 生成
    # ------------------------------------------------------------------

    def _ensure_urdf_exists(self):
        """每次运行都重新生成，方便调外观。"""
        os.makedirs("models", exist_ok=True)
        self._create_urdf(
            path="models/humanoid_left.urdf",
            robot_name="humanoid_left",
            cloth_rgba="0.2 0.6 0.9 1.0",
            dark_rgba="0.05 0.15 0.35 1.0",
            racket_side="left",
        )
        self._create_urdf(
            path="models/humanoid_right.urdf",
            robot_name="humanoid_right",
            cloth_rgba="0.85 0.15 0.15 1.0",
            dark_rgba="0.35 0.05 0.05 1.0",
            racket_side="right",
        )

    def _create_urdf(self, path, robot_name, cloth_rgba, dark_rgba, racket_side):
        """生成一个简化但更清楚的人形机器人 URDF。"""
        if racket_side not in {"left", "right"}:
            raise ValueError("racket_side must be 'left' or 'right'")

        skin = "0.90 0.78 0.62 1.0"
        glove = "1.0 0.85 0.05 1.0"
        black = "0.02 0.02 0.02 1.0"
        white = "0.95 0.95 0.95 1.0"
        racket_color = "0.05 0.05 0.05 1.0"

        # 让持拍手更明显：手掌大、黄色、球拍大。
        left_racket_xml = self._arm_xml(
            side="left",
            cloth_material="cloth",
            skin_material="skin",
            glove_material="glove",
            racket_material="racket_mat",
            has_racket=(racket_side == "left"),
        )
        right_racket_xml = self._arm_xml(
            side="right",
            cloth_material="cloth",
            skin_material="skin",
            glove_material="glove",
            racket_material="racket_mat",
            has_racket=(racket_side == "right"),
        )

        urdf = f'''<?xml version="1.0" ?>
<robot name="{robot_name}">
  <material name="cloth"><color rgba="{cloth_rgba}"/></material>
  <material name="dark_cloth"><color rgba="{dark_rgba}"/></material>
  <material name="skin"><color rgba="{skin}"/></material>
  <material name="glove"><color rgba="{glove}"/></material>
  <material name="black"><color rgba="{black}"/></material>
  <material name="white"><color rgba="{white}"/></material>
  <material name="racket_mat"><color rgba="{racket_color}"/></material>

  <!-- root link: torso -->
  <link name="torso">
    <inertial><mass value="12"/><inertia ixx="0.12" ixy="0" ixz="0" iyy="0.10" iyz="0" izz="0.08"/></inertial>
    <visual><geometry><box size="0.38 0.26 0.52"/></geometry><material name="cloth"/></visual>
    <collision><geometry><box size="0.38 0.26 0.52"/></geometry></collision>
  </link>

  <link name="pelvis">
    <inertial><mass value="6"/><inertia ixx="0.06" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.04"/></inertial>
    <visual><geometry><box size="0.34 0.28 0.22"/></geometry><material name="dark_cloth"/></visual>
    <collision><geometry><box size="0.34 0.28 0.22"/></geometry></collision>
  </link>
  <joint name="waist_joint" type="fixed"><parent link="torso"/><child link="pelvis"/><origin xyz="0 0 -0.37"/></joint>

  <link name="neck">
    <inertial><mass value="0.7"/><inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/></inertial>
    <visual><geometry><cylinder radius="0.055" length="0.12"/></geometry><material name="skin"/></visual>
    <collision><geometry><cylinder radius="0.055" length="0.12"/></geometry></collision>
  </link>
  <joint name="neck_joint" type="fixed"><parent link="torso"/><child link="neck"/><origin xyz="0 0 0.32"/></joint>

  <link name="head">
    <inertial><mass value="1.6"/><inertia ixx="0.016" ixy="0" ixz="0" iyy="0.016" iyz="0" izz="0.016"/></inertial>
    <visual><geometry><sphere radius="0.15"/></geometry><material name="skin"/></visual>
    <collision><geometry><sphere radius="0.15"/></geometry></collision>
  </link>
  <joint name="head_joint" type="fixed"><parent link="neck"/><child link="head"/><origin xyz="0 0 0.14"/></joint>

  <link name="left_eye"><inertial><mass value="0.02"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial><visual><geometry><sphere radius="0.032"/></geometry><material name="white"/></visual><collision><geometry><sphere radius="0.032"/></geometry></collision></link>
  <joint name="left_eye_joint" type="fixed"><parent link="head"/><child link="left_eye"/><origin xyz="0.055 -0.12 0.035"/></joint>
  <link name="right_eye"><inertial><mass value="0.02"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial><visual><geometry><sphere radius="0.032"/></geometry><material name="white"/></visual><collision><geometry><sphere radius="0.032"/></geometry></collision></link>
  <joint name="right_eye_joint" type="fixed"><parent link="head"/><child link="right_eye"/><origin xyz="-0.055 -0.12 0.035"/></joint>

  {self._leg_xml("left")}
  {self._leg_xml("right")}

  {left_racket_xml}
  {right_racket_xml}
</robot>
'''

        with open(path, "w", encoding="utf-8") as f:
            f.write(urdf)
        print(f"URDF generated: {path}")

    def _leg_xml(self, side):
        y = -0.11 if side == "left" else 0.11
        return f'''
  <!-- {side} leg -->
  <link name="{side}_thigh">
    <inertial><mass value="4"/><inertia ixx="0.035" ixy="0" ixz="0" iyy="0.035" iyz="0" izz="0.025"/></inertial>
    <visual><origin xyz="0 0 -0.18"/><geometry><cylinder radius="0.075" length="0.36"/></geometry><material name="cloth"/></visual>
    <collision><origin xyz="0 0 -0.18"/><geometry><cylinder radius="0.075" length="0.36"/></geometry></collision>
  </link>
  <joint name="{side}_hip" type="revolute">
    <parent link="pelvis"/><child link="{side}_thigh"/>
    <origin xyz="0 {y} -0.12"/><axis xyz="1 0 0"/>
    <limit lower="-0.9" upper="0.9" effort="120" velocity="2.5"/>
  </joint>

  <link name="{side}_calf">
    <inertial><mass value="3"/><inertia ixx="0.025" ixy="0" ixz="0" iyy="0.025" iyz="0" izz="0.018"/></inertial>
    <visual><origin xyz="0 0 -0.18"/><geometry><cylinder radius="0.06" length="0.36"/></geometry><material name="cloth"/></visual>
    <collision><origin xyz="0 0 -0.18"/><geometry><cylinder radius="0.06" length="0.36"/></geometry></collision>
  </link>
  <joint name="{side}_knee" type="revolute">
    <parent link="{side}_thigh"/><child link="{side}_calf"/>
    <origin xyz="0 0 -0.36"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="1.4" effort="100" velocity="2.5"/>
  </joint>

  <link name="{side}_foot">
    <inertial><mass value="1"/><inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.006"/></inertial>
    <visual><origin xyz="0 -0.04 -0.04"/><geometry><box size="0.16 0.28 0.08"/></geometry><material name="black"/></visual>
    <collision><origin xyz="0 -0.04 -0.04"/><geometry><box size="0.16 0.28 0.08"/></geometry></collision>
  </link>
  <joint name="{side}_ankle" type="revolute">
    <parent link="{side}_calf"/><child link="{side}_foot"/>
    <origin xyz="0 0 -0.36"/><axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="70" velocity="2.5"/>
  </joint>
'''

    def _arm_xml(self, side, cloth_material, skin_material, glove_material, racket_material, has_racket):
        y = -0.24 if side == "left" else 0.24
        racket_xml = ""
        if has_racket:
            racket_xml = f'''
  <link name="{side}_racket">
    <inertial><mass value="0.25"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual><geometry><box size="0.10 0.38 0.025"/></geometry><material name="{racket_material}"/></visual>
    <collision><geometry><box size="0.10 0.38 0.025"/></geometry></collision>
  </link>
  <joint name="{side}_racket_joint" type="fixed">
    <parent link="{side}_hand"/><child link="{side}_racket"/>
    <origin xyz="0 0 -0.25"/>
  </joint>
'''

        return f'''
  <!-- {side} arm -->
  <link name="{side}_upper_arm">
    <inertial><mass value="1.5"/><inertia ixx="0.014" ixy="0" ixz="0" iyy="0.014" iyz="0" izz="0.010"/></inertial>
    <visual><origin xyz="0 0 -0.16"/><geometry><cylinder radius="0.055" length="0.32"/></geometry><material name="{cloth_material}"/></visual>
    <collision><origin xyz="0 0 -0.16"/><geometry><cylinder radius="0.055" length="0.32"/></geometry></collision>
  </link>
  <joint name="{side}_shoulder_pitch" type="revolute">
    <parent link="torso"/><child link="{side}_upper_arm"/>
    <origin xyz="0 {y} 0.18"/><axis xyz="0 1 0"/>
    <limit lower="-1.8" upper="1.8" effort="120" velocity="4.0"/>
  </joint>

  <link name="{side}_forearm">
    <inertial><mass value="1.1"/><inertia ixx="0.012" ixy="0" ixz="0" iyy="0.012" iyz="0" izz="0.008"/></inertial>
    <visual><origin xyz="0 0 -0.15"/><geometry><cylinder radius="0.048" length="0.30"/></geometry><material name="{cloth_material}"/></visual>
    <collision><origin xyz="0 0 -0.15"/><geometry><cylinder radius="0.048" length="0.30"/></geometry></collision>
  </link>
  <joint name="{side}_elbow" type="revolute">
    <parent link="{side}_upper_arm"/><child link="{side}_forearm"/>
    <origin xyz="0 0 -0.32"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="1.8" effort="100" velocity="4.0"/>
  </joint>

  <link name="{side}_hand">
    <inertial><mass value="0.55"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <visual><geometry><box size="0.15 0.17 0.12"/></geometry><material name="{glove_material}"/></visual>
    <collision><geometry><box size="0.15 0.17 0.12"/></geometry></collision>
  </link>
  <joint name="{side}_wrist" type="revolute">
    <parent link="{side}_forearm"/><child link="{side}_hand"/>
    <origin xyz="0 0 -0.32"/><axis xyz="0 1 0"/>
    <limit lower="-0.9" upper="0.9" effort="80" velocity="4.0"/>
  </joint>

  {racket_xml}
'''

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------

    def _get_link_index(self, robot_id, link_name):
        for i in range(p.getNumJoints(robot_id)):
            link = p.getJointInfo(robot_id, i)[12].decode("utf-8")
            if link == link_name:
                return i
        return -1

    def _find_first_link(self, robot_id, names):
        for name in names:
            idx = self._get_link_index(robot_id, name)
            if idx >= 0:
                return idx
        print("Warning: no link found from candidates:", names)
        return -1

    def _set_neutral_pose(self):
        neutral = {
            "left_hip": 0.0,
            "left_knee": 0.15,
            "left_ankle": -0.05,
            "right_hip": 0.0,
            "right_knee": 0.15,
            "right_ankle": -0.05,
            "left_shoulder_pitch": 0.3,
            "left_elbow": 0.3,
            "left_wrist": 0.0,
            "right_shoulder_pitch": 0.3,
            "right_elbow": 0.3,
            "right_wrist": 0.0,
        }
        self.left_robot.set_joint_positions(neutral, velocity=3, force=80)
        self.right_robot.set_joint_positions(neutral, velocity=3, force=80)

    def predict_future_position(self, pos, vel, dt_pred=0.55):
        g = 9.8
        future_pos = pos + vel * dt_pred
        future_pos[2] = pos[2] + vel[2] * dt_pred - 0.5 * g * dt_pred**2
        return future_pos

    def move_robot_to_target(self, robot, target_pos, speed=1.6):
        current = np.array(robot.get_position(), dtype=float)
        dx = target_pos[0] - current[0]
        dy = target_pos[1] - current[1]
        max_step = min(0.04, speed * (1.0 / 240.0) * 2.5)

        new_x = current[0] + np.clip(dx, -max_step, max_step)
        new_y = current[1] + np.clip(dy, -max_step, max_step)

        if robot == self.left_robot:
            new_x = np.clip(new_x, -4.0, -0.45)
        else:
            new_x = np.clip(new_x, 0.45, 4.0)

        # 保持身体直立，避免移动过程中机器人倾倒得太夸张。
        orn = [0, 0, 0, 1] if robot == self.left_robot else [0, 0, 1, 0]
        p.resetBasePositionAndOrientation(robot.robot_id, [new_x, new_y, current[2]], orn)

    def get_racket_position(self, robot_id, link_idx):
        if link_idx >= 0:
            state = p.getLinkState(robot_id, link_idx, computeLinkVelocity=1)
            return np.array(state[0], dtype=float)
        return np.array([0.0, 0.0, 0.0], dtype=float)

    def get_racket_position_and_velocity(self, robot_id, link_idx, prev_pos, dt):
        """返回球拍位置和速度。

        PyBullet 的 link velocity 对固定子 link 有时不够稳定，所以这里优先用
        “当前位置 - 上一帧位置”估计速度。这样视觉挥拍和击球判定会绑定在一起。
        """
        pos = self.get_racket_position(robot_id, link_idx)
        if prev_pos is None or dt <= 1e-6:
            vel = np.zeros(3, dtype=float)
        else:
            vel = (pos - prev_pos) / dt
        return pos, vel

    # ------------------------------------------------------------------
    # 机器人运动与击球
    # ------------------------------------------------------------------

    def update_robots_movement(self, dt):
        self.walk_left.update_phase(dt)
        self.walk_right.update_phase(dt)

        for robot, walk in [(self.left_robot, self.walk_left), (self.right_robot, self.walk_right)]:
            for side in ["left", "right"]:
                hip = walk.get_leg_angle(side, "hip")
                knee = walk.get_knee_angle()
                ankle = -hip * 0.35
                robot.set_joint_positions(
                    {
                        f"{side}_hip": hip,
                        f"{side}_knee": knee,
                        f"{side}_ankle": ankle,
                    },
                    velocity=3,
                    force=120,
                )

        target = self.predict_future_position(self.shuttle_pos, self.shuttle_vel, 0.55)
        target[1] = np.clip(target[1], -1.8, 1.8)

        if self.shuttle_pos[0] < 0:
            target[0] = np.clip(target[0], -3.8, -0.55)
            self.move_robot_to_target(self.left_robot, target)
        else:
            target[0] = np.clip(target[0], 0.55, 3.8)
            self.move_robot_to_target(self.right_robot, target)

    def prepare_swing_if_needed(self, racket_pos, shuttle_pos, robot_side):
        """球快到本方时提前挥拍。

        之前只有真正击球那一刻才触发挥臂，视觉上会像“没有挥臂”。
        现在提前触发，等球到球拍附近时再真正施加击球力。
        """
        if robot_side == "left" and shuttle_pos[0] > -0.10:
            return
        if robot_side == "right" and shuttle_pos[0] < 0.10:
            return

        dist = np.linalg.norm(racket_pos - shuttle_pos)
        good_height = 0.45 < shuttle_pos[2] < 1.85
        enough_cooldown = (self.step - self.last_hit_step) > 25

        if dist < 1.15 and good_height and enough_cooldown:
            if robot_side == "left" and not self.swing_left:
                self.swing_left = True
                self.swing_timer_left = 0.0
            elif robot_side == "right" and not self.swing_right:
                self.swing_right = True
                self.swing_timer_right = 0.0

    def check_and_hit(self, racket_pos, racket_vel, shuttle_pos, shuttle_vel, robot_side):
        """用球拍位置和球拍速度判断击球。

        现在不再是“球到某区域就自动飞走”，而是必须满足：
        1. 球在本方；
        2. 球拍离球足够近；
        3. 球拍正在朝对方场地挥动；
        4. 冷却时间足够。
        """
        if robot_side == "left" and shuttle_pos[0] > -0.05:
            return None
        if robot_side == "right" and shuttle_pos[0] < 0.05:
            return None

        dist = np.linalg.norm(racket_pos - shuttle_pos)
        good_height = 0.40 < shuttle_pos[2] < 1.75
        enough_cooldown = (self.step - self.last_hit_step) > 35

        # 球拍必须向对方半场运动。这样击球和挥拍动作会绑定在一起。
        forward_speed = racket_vel[0] if robot_side == "left" else -racket_vel[0]
        swinging_forward = forward_speed > 0.35

        # 允许球拍稍远一点，因为这个简化模型的球拍是盒子，视觉中心不一定正好在拍面中心。
        if dist < 0.75 and good_height and enough_cooldown and swinging_forward:
            if robot_side == "left":
                horiz_dir = np.array([1.0, random.uniform(-0.40, 0.40)])
            else:
                horiz_dir = np.array([-1.0, random.uniform(-0.40, 0.40)])

            # 先计算保证过网的速度，再叠加一部分球拍速度。
            target_vel = self.ensure_over_net_velocity(racket_pos, horiz_dir)
            target_vel = self._normalize_speed(target_vel, desired_speed=13.0)
            target_vel[2] = max(target_vel[2], 5.4)

            # 球拍速度参与击球，让视觉动作和球的反弹有关联。
            racket_boost = np.array([racket_vel[0] * 0.45, racket_vel[1] * 0.25, abs(racket_vel[2]) * 0.18])
            target_vel = target_vel + racket_boost

            mass = 0.005
            impulse_dt = 1.0 / 240.0
            force = mass * (target_vel - shuttle_vel) / impulse_dt
            self.last_hit_step = self.step
            print(
                f"{robot_side.upper()} RACKET HIT: "
                f"dist={dist:.2f}, racket_vel={racket_vel.round(2)}, target_vel={target_vel.round(2)}"
            )
            return force

        return None

    def _normalize_speed(self, vel, desired_speed):
        norm = np.linalg.norm(vel)
        if norm < 1e-6:
            return vel
        return vel / norm * desired_speed

    def ensure_over_net_velocity(self, hit_pos, target_dir_2d):
        g = 9.8
        horiz_dir = target_dir_2d[:2] / (np.linalg.norm(target_dir_2d[:2]) + 1e-6)
        # 水平速度不能太低，否则飞到网前耗时太长，重力会把球拉低。
        horiz_speed = 4.6
        vx = horiz_dir[0] * horiz_speed
        vy = horiz_dir[1] * horiz_speed

        dx_net = 0.0 - hit_pos[0]
        if abs(dx_net) < 0.05:
            dx_net = 0.05 if dx_net >= 0 else -0.05

        t_net = dx_net / vx if abs(vx) > 1e-6 else 0.25
        if t_net <= 0:
            t_net = 0.25

        # 球网高度是 1.55，这里使用更高的安全高度，减少撞网。
        net_clearance = 2.05
        vz_min = (net_clearance - hit_pos[2] + 0.5 * g * t_net**2) / t_net
        # 在理论最小过网速度上加安全余量。
        vz = max(vz_min + 1.2, 4.8)
        return np.array([vx, vy, vz], dtype=float)

    def update_swing_animation(self, dt):
        if self.swing_left:
            self.swing_timer_left = self._update_one_swing(
                robot=self.left_robot,
                side="left",
                timer=self.swing_timer_left,
                dt=dt,
            )
            if self.swing_timer_left < 0:
                self.swing_left = False
                self.swing_timer_left = 0.0

        if self.swing_right:
            self.swing_timer_right = self._update_one_swing(
                robot=self.right_robot,
                side="right",
                timer=self.swing_timer_right,
                dt=dt,
            )
            if self.swing_timer_right < 0:
                self.swing_right = False
                self.swing_timer_right = 0.0

    def _update_one_swing(self, robot, side, timer, dt):
        timer += dt

        # 左右机器人的朝向不同，右机器人整体旋转了 180 度。
        # 这里用方向系数让两边都表现为“向对方场地挥拍”。
        direction = 1.0 if side == "left" else -1.0

        if timer < 0.16:
            # 蓄力：手臂明显后摆。
            t = timer / 0.16
            pitch = 0.15 + 1.45 * t
            elbow = 0.25 + 1.10 * t
            wrist = 0.0 + 0.75 * t
            velocity = 10
            force = 220
        elif timer < 0.30:
            # 击球：快速甩臂。这个阶段球拍速度最大。
            t = (timer - 0.16) / 0.14
            pitch = 1.60 - 2.25 * t
            elbow = 1.35 - 0.95 * t
            wrist = 0.75 - 1.35 * t
            velocity = 20
            force = 320
        elif timer < 0.52:
            # 随挥和回收：动作停留更久，肉眼更容易看到。
            t = (timer - 0.30) / 0.22
            pitch = -0.65 + 0.90 * t
            elbow = 0.40 + 0.05 * t
            wrist = -0.60 + 0.60 * t
            velocity = 7
            force = 150
        else:
            robot.set_joint_positions(
                {
                    f"{side}_shoulder_pitch": 0.25,
                    f"{side}_elbow": 0.35,
                    f"{side}_wrist": 0.0,
                },
                velocity=4,
                force=80,
            )
            return -1.0

        # direction 让右侧机器人动作在视觉上和左侧对称。
        robot.set_joint_positions(
            {
                f"{side}_shoulder_pitch": pitch * direction,
                f"{side}_elbow": max(0.05, elbow),
                f"{side}_wrist": wrist * direction,
            },
            velocity=velocity,
            force=force,
        )
        return timer

    # ------------------------------------------------------------------
    # 羽毛球发球与主循环
    # ------------------------------------------------------------------

    def _random_serve_with_over_net(self):
        start = np.array([0.0, 0.0, 1.35], dtype=float)
        side = random.choice([-1, 1])
        target_x = random.uniform(1.2, 3.0) if side == 1 else random.uniform(-3.0, -1.2)
        target_y = random.uniform(-1.5, 1.5)

        dx = target_x - start[0]
        dy = target_y - start[1]
        t_flight = random.uniform(0.85, 1.15)
        vx = dx / t_flight
        vy = dy / t_flight
        g = 9.8
        vz = (0.5 * g * t_flight**2 - (start[2] - 0.25)) / t_flight
        # 发球时也抬高一点，避免一开始就过不了网。
        vz = max(vz, 3.2)

        if abs(vx) < 2.2:
            vx = 2.5 if vx >= 0 else -2.5

        self.shuttle_pos = start.copy()
        self.shuttle_vel = np.array([vx, vy, vz], dtype=float)
        p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)
        print(f"SERVE: target=({target_x:.1f},{target_y:.1f}), vel=({vx:.1f},{vy:.1f},{vz:.1f})")

    def run(self):
        last_time = time.time()
        try:
            while True:
                now = time.time()
                dt = min(0.033, now - last_time)
                last_time = now
                if dt <= 0:
                    dt = 0.016

                self.update_robots_movement(dt)
                self.update_swing_animation(dt)

                left_racket, self.left_racket_vel = self.get_racket_position_and_velocity(
                    self.left_robot.robot_id,
                    self.left_racket_link,
                    self.prev_left_racket_pos,
                    dt,
                )
                right_racket, self.right_racket_vel = self.get_racket_position_and_velocity(
                    self.right_robot.robot_id,
                    self.right_racket_link,
                    self.prev_right_racket_pos,
                    dt,
                )
                self.prev_left_racket_pos = left_racket.copy()
                self.prev_right_racket_pos = right_racket.copy()

                # 先提前挥拍，再用球拍位置和速度判断是否真正击中。
                if self.shuttle_pos[0] < 0:
                    self.prepare_swing_if_needed(left_racket, self.shuttle_pos, "left")
                else:
                    self.prepare_swing_if_needed(right_racket, self.shuttle_pos, "right")

                force = None
                if self.shuttle_pos[0] < 0:
                    force = self.check_and_hit(left_racket, self.left_racket_vel, self.shuttle_pos, self.shuttle_vel, "left")
                else:
                    force = self.check_and_hit(right_racket, self.right_racket_vel, self.shuttle_pos, self.shuttle_vel, "right")

                self.shuttle_pos, self.shuttle_vel = self.shuttle_phys.update(
                    self.shuttle_pos,
                    self.shuttle_vel,
                    dt,
                    force,
                )

                if self.shuttle_pos[2] < 0.08 or abs(self.shuttle_pos[0]) > 4.3 or abs(self.shuttle_pos[1]) > 2.4:
                    print("落地或出界，重新发球")
                    self._random_serve_with_over_net()
                    self.last_hit_step = self.step - 80

                p.resetBasePositionAndOrientation(self.shuttle_id, self.shuttle_pos, [0, 0, 0, 1])
                p.resetBaseVelocity(self.shuttle_id, self.shuttle_vel)
                p.stepSimulation()
                self.step += 1
                time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            print("退出")
        finally:
            p.disconnect()


if __name__ == "__main__":
    game = HumanoidBadmintonGame()
    game.run()
