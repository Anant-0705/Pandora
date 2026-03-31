import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def act(self, observation, info=None):
        # Pick 3 valid discrete actions from the space
        return self.action_space.sample()
