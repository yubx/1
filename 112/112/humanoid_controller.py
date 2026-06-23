import pybullet as p
import numpy as np

class HumanoidController:
    def __init__(self, urdf_path, base_pos, base_orn):
        self.robot_id = p.loadURDF(urdf_path, base_pos, base_orn, useFixedBase=False)
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode('utf-8')
            self.joint_indices[name] = i
    
    def set_joint_positions(self, pos_dict, velocity=5, force=100):
        for name, pos in pos_dict.items():
            if name in self.joint_indices:
                p.setJointMotorControl2(self.robot_id, self.joint_indices[name],
                                        p.POSITION_CONTROL, targetPosition=pos,
                                        force=force, maxVelocity=velocity)
    
    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos)
    
    def get_link_position(self, link_name):
        if link_name in self.joint_indices:
            link_state = p.getLinkState(self.robot_id, self.joint_indices[link_name])
            return np.array(link_state[0])
        return np.zeros(3)