from gymnasium.core import ActionWrapper
from gymnasium import spaces
import numpy as np

class PadGripper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def action(self, action_4d):
        full_action = np.zeros(5, dtype=np.float32)
        full_action[:4] = action_4d
        return full_action