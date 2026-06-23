import numpy as np

class ShuttlePhysics:
    def __init__(self, g=-9.8, drag_coeff=0.012):
        self.g = g
        self.k = drag_coeff
    
    def update(self, pos, vel, dt, force=None):
        if force is not None:
            mass = 0.005
            vel = vel + force * dt / mass
        gravity = np.array([0, 0, self.g])
        acc = gravity - self.k * vel
        vel = vel + acc * dt
        pos = pos + vel * dt
        return pos, vel