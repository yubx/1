import os

def create_urdf_with_fingers(path, color):
    color_str = f"{color[0]} {color[1]} {color[2]} {color[3]}"
    cloth_color = f"{color[0]*0.8} {color[1]*0.8} {color[2]*0.8} 1"
    skin_color = "0.9 0.85 0.7 1"
    urdf = f'''<?xml version="1.0" ?>
<robot name="humanoid_with_fingers">
  <material name="body"><color rgba="{color_str}"/></material>
  <material name="cloth"><color rgba="{cloth_color}"/></material>
  <material name="skin"><color rgba="{skin_color}"/></material>
  <material name="eye_white"><color rgba="1 1 1 1"/></material>
  <material name="eye_black"><color rgba="0 0 0 1"/></material>
  <material name="racket_mat"><color rgba="0.9 0.7 0.2 1"/></material>

  <!-- 躯干 -->
  <link name="torso">
    <inertial><mass value="10"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <visual><geometry><box size="0.4 0.3 0.6"/></geometry><material name="cloth"/></visual>
    <collision><geometry><box size="0.4 0.3 0.6"/></geometry></collision>
  </link>

  <!-- 颈部 -->
  <link name="neck">
    <inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><geometry><cylinder radius="0.07" length="0.08"/></geometry><material name="skin"/></visual>
    <collision><geometry><cylinder radius="0.07" length="0.08"/></geometry></collision>
  </link>
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/><child link="neck"/>
    <origin xyz="0 0 0.32"/><axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="1"/>
  </joint>

  <!-- 头部 -->
  <link name="head">
    <inertial><mass value="1.2"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><sphere radius="0.13"/></geometry><material name="skin"/></visual>
    <collision><geometry><sphere radius="0.13"/></geometry></collision>
  </link>
  <joint name="head_joint" type="fixed"><parent link="neck"/><child link="head"/><origin xyz="0 0 0.06"/></joint>

  <!-- 腿部 -->
  <link name="left_thigh"><inertial><mass value="3"/><inertia ixx="0.02"/></inertial><visual><geometry><cylinder radius="0.08" length="0.32"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.08" length="0.32"/></geometry></collision></link>
  <joint name="left_hip" type="revolute"><parent link="torso"/><child link="left_thigh"/><origin xyz="-0.12 -0.12 -0.3"/><axis xyz="1 0 0"/><limit lower="-0.8" upper="0.8" effort="100" velocity="2"/></joint>
  <link name="left_calf"><inertial><mass value="2"/><inertia ixx="0.01"/></inertial><visual><geometry><cylinder radius="0.07" length="0.32"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.07" length="0.32"/></geometry></collision></link>
  <joint name="left_knee" type="revolute"><parent link="left_thigh"/><child link="left_calf"/><origin xyz="0 0 -0.16"/><axis xyz="1 0 0"/><limit lower="0" upper="1.2" effort="80" velocity="2"/></joint>
  <link name="left_foot"><inertial><mass value="0.6"/><inertia ixx="0.005"/></inertial><visual><geometry><box size="0.12 0.2 0.08"/></geometry><material name="cloth"/></visual><collision><geometry><box size="0.12 0.2 0.08"/></geometry></collision></link>
  <joint name="left_ankle" type="revolute"><parent link="left_calf"/><child link="left_foot"/><origin xyz="0 0 -0.16"/><axis xyz="1 0 0"/><limit lower="-0.4" upper="0.4" effort="50" velocity="2"/></joint>

  <link name="right_thigh"><inertial><mass value="3"/><inertia ixx="0.02"/></inertial><visual><geometry><cylinder radius="0.08" length="0.32"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.08" length="0.32"/></geometry></collision></link>
  <joint name="right_hip" type="revolute"><parent link="torso"/><child link="right_thigh"/><origin xyz="-0.12 0.12 -0.3"/><axis xyz="1 0 0"/><limit lower="-0.8" upper="0.8" effort="100" velocity="2"/></joint>
  <link name="right_calf"><inertial><mass value="2"/><inertia ixx="0.01"/></inertial><visual><geometry><cylinder radius="0.07" length="0.32"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.07" length="0.32"/></geometry></collision></link>
  <joint name="right_knee" type="revolute"><parent link="right_thigh"/><child link="right_calf"/><origin xyz="0 0 -0.16"/><axis xyz="1 0 0"/><limit lower="0" upper="1.2" effort="80" velocity="2"/></joint>
  <link name="right_foot"><inertial><mass value="0.6"/><inertia ixx="0.005"/></inertial><visual><geometry><box size="0.12 0.2 0.08"/></geometry><material name="cloth"/></visual><collision><geometry><box size="0.12 0.2 0.08"/></geometry></collision></link>
  <joint name="right_ankle" type="revolute"><parent link="right_calf"/><child link="right_foot"/><origin xyz="0 0 -0.16"/><axis xyz="1 0 0"/><limit lower="-0.4" upper="0.4" effort="50" velocity="2"/></joint>

  <!-- 左臂+手指+球拍 -->
  <link name="left_upper_arm"><inertial><mass value="1.2"/><inertia ixx="0.01"/></inertial><visual><geometry><cylinder radius="0.07" length="0.28"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.07" length="0.28"/></geometry></collision></link>
  <joint name="left_shoulder_yaw" type="revolute"><parent link="torso"/><child link="left_shoulder_yaw_link"/><origin xyz="0.22 -0.18 0.22"/><axis xyz="0 0 1"/><limit lower="-1.2" upper="1.2" effort="80" velocity="2"/></joint>
  <link name="left_shoulder_yaw_link"><inertial><mass value="0.5"/><inertia ixx="0.005"/></inertial><visual><geometry><sphere radius="0.06"/></geometry><material name="cloth"/></visual><collision><geometry><sphere radius="0.06"/></geometry></collision></link>
  <joint name="left_shoulder_pitch" type="revolute"><parent link="left_shoulder_yaw_link"/><child link="left_upper_arm"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.8" effort="100" velocity="3"/></joint>
  <link name="left_forearm"><inertial><mass value="0.9"/><inertia ixx="0.008"/></inertial><visual><geometry><cylinder radius="0.06" length="0.26"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.06" length="0.26"/></geometry></collision></link>
  <joint name="left_elbow" type="revolute"><parent link="left_upper_arm"/><child link="left_forearm"/><origin xyz="0 0 -0.14"/><axis xyz="0 1 0"/><limit lower="0" upper="1.8" effort="80" velocity="3"/></joint>
  <link name="left_palm"><inertial><mass value="0.4"/><inertia ixx="0.003"/></inertial><visual><geometry><box size="0.08 0.1 0.06"/></geometry><material name="skin"/></visual><collision><geometry><box size="0.08 0.1 0.06"/></geometry></collision></link>
  <joint name="left_wrist" type="revolute"><parent link="left_forearm"/><child link="left_palm"/><origin xyz="0 0 -0.13"/><axis xyz="0 1 0"/><limit lower="-0.8" upper="0.8" effort="60" velocity="3"/></joint>
  <link name="left_finger1"><inertial><mass value="0.05"/><inertia ixx="0.0001"/></inertial><visual><geometry><box size="0.02 0.04 0.04"/></geometry><material name="skin"/></visual><collision><geometry><box size="0.02 0.04 0.04"/></geometry></collision></link>
  <joint name="left_finger1_joint" type="revolute"><parent link="left_palm"/><child link="left_finger1"/><origin xyz="0.03 0.04 -0.02"/><axis xyz="0 1 0"/><limit lower="0" upper="1.2" effort="20" velocity="2"/></joint>
  <link name="left_finger2"><inertial><mass value="0.05"/><inertia ixx="0.0001"/></inertial><visual><geometry><box size="0.02 0.04 0.04"/></geometry><material name="skin"/></visual><collision><geometry><box size="0.02 0.04 0.04"/></geometry></collision></link>
  <joint name="left_finger2_joint" type="revolute"><parent link="left_palm"/><child link="left_finger2"/><origin xyz="-0.01 0.04 -0.02"/><axis xyz="0 1 0"/><limit lower="0" upper="1.2" effort="20" velocity="2"/></joint>
  <link name="left_racket"><inertial><mass value="0.2"/><inertia ixx="0.001"/></inertial><visual><geometry><box size="0.05 0.22 0.01"/></geometry><material name="racket_mat"/></visual><collision><geometry><box size="0.05 0.22 0.01"/></geometry></collision></link>
  <joint name="left_racket_joint" type="fixed"><parent link="left_palm"/><child link="left_racket"/><origin xyz="0 0 -0.1"/></joint>

  <!-- 右臂（对称） -->
  <link name="right_upper_arm"><inertial><mass value="1.2"/><inertia ixx="0.01"/></inertial><visual><geometry><cylinder radius="0.07" length="0.28"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.07" length="0.28"/></geometry></collision></link>
  <joint name="right_shoulder_yaw" type="revolute"><parent link="torso"/><child link="right_shoulder_yaw_link"/><origin xyz="0.22 0.18 0.22"/><axis xyz="0 0 1"/><limit lower="-1.2" upper="1.2" effort="80" velocity="2"/></joint>
  <link name="right_shoulder_yaw_link"><inertial><mass value="0.5"/><inertia ixx="0.005"/></inertial><visual><geometry><sphere radius="0.06"/></geometry><material name="cloth"/></visual><collision><geometry><sphere radius="0.06"/></geometry></collision></link>
  <joint name="right_shoulder_pitch" type="revolute"><parent link="right_shoulder_yaw_link"/><child link="right_upper_arm"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.8" effort="100" velocity="3"/></joint>
  <link name="right_forearm"><inertial><mass value="0.9"/><inertia ixx="0.008"/></inertial><visual><geometry><cylinder radius="0.06" length="0.26"/></geometry><material name="cloth"/></visual><collision><geometry><cylinder radius="0.06" length="0.26"/></geometry></collision></link>
  <joint name="right_elbow" type="revolute"><parent link="right_upper_arm"/><child link="right_forearm"/><origin xyz="0 0 -0.14"/><axis xyz="0 1 0"/><limit lower="0" upper="1.8" effort="80" velocity="3"/></joint>
  <link name="right_palm"><inertial><mass value="0.4"/><inertia ixx="0.003"/></inertial><visual><geometry><box size="0.08 0.1 0.06"/></geometry><material name="skin"/></visual><collision><geometry><box size="0.08 0.1 0.06"/></geometry></collision></link>
  <joint name="right_wrist" type="revolute"><parent link="right_forearm"/><child link="right_palm"/><origin xyz="0 0 -0.13"/><axis xyz="0 1 0"/><limit lower="-0.8" upper="0.8" effort="60" velocity="3"/></joint>
  <link name="right_finger1"><inertial><mass value="0.05"/><inertia ixx="0.0001"/></inertial><visual><geometry><box size="0.02 0.04 0.04"/></geometry><material name="skin"/></visual><collision><geometry><box size="0.02 0.04 0.04"/></geometry></collision></link>
  <joint name="right_finger1_joint" type="revolute"><parent link="right_palm"/><child link="right_finger1"/><origin xyz="0.03 -0.04 -0.02"/><axis xyz="0 1 0"/><limit lower="0" upper="1.2" effort="20" velocity="2"/></joint>
  <link name="right_finger2"><inertial><mass value="0.05"/><inertia ixx="0.0001"/></inertial><visual><geometry><box size="0.02 0.04 0.04"/></geometry><material name="skin"/></visual><collision><geometry><box size="0.02 0.04 0.04"/></geometry></collision></link>
  <joint name="right_finger2_joint" type="revolute"><parent link="right_palm"/><child link="right_finger2"/><origin xyz="-0.01 -0.04 -0.02"/><axis xyz="0 1 0"/><limit lower="0" upper="1.2" effort="20" velocity="2"/></joint>
  <link name="right_racket"><inertial><mass value="0.2"/><inertia ixx="0.001"/></inertial><visual><geometry><box size="0.05 0.22 0.01"/></geometry><material name="racket_mat"/></visual><collision><geometry><box size="0.05 0.22 0.01"/></geometry></collision></link>
  <joint name="right_racket_joint" type="fixed"><parent link="right_palm"/><child link="right_racket"/><origin xyz="0 0 -0.1"/></joint>
</robot>'''
    with open(path, 'w') as f:
        f.write(urdf)
    print(f"URDF with fingers created: {path}")