import numpy as np

class BipedWalkGenerator:
    def __init__(self, step_length=0.15, step_height=0.06, period=1.0):
        self.step_len = step_length
        self.step_ht = step_height
        self.period = period
        self.phase = 0.0
    
    def update_phase(self, dt):
        self.phase += 2 * np.pi * dt / self.period
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
    
    def get_leg_angle(self, side, joint):
        sign = 1.0 if side == 'left' else -1.0
        swing = np.sin(self.phase)
        if joint == 'hip':
            return 0.3 * swing * sign
        elif joint == 'knee':
            return 0.2 * (1 - abs(swing))
        else:
            return 0.0
    
    def get_knee_angle(self):
        return 0.2 * (1 - abs(np.sin(self.phase)))