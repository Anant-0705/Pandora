import gymnasium as gym
import numpy as np
from gymnasium import spaces

class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Normalizes observations to a standard scale to help neural networks learn more effectively.
    
    The raw observation space has widely varying scales:
    - year: 0-10000
    - population: 0-1e12
    - happiness: 0-1
    - inequality: 0-1
    - tech_level: 0-20
    - climate_health: 0-1
    - resources: 0-1e6 each (food, water, energy, minerals)
    
    This wrapper normalizes all observations to roughly [0, 1] range.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define normalization parameters for each dimension
        # [year, population, happiness, inequality, tech_level, climate_health, food, water, energy, minerals]
        self.obs_max = np.array([10000, 1e12, 1, 1, 20, 1, 1e6, 1e6, 1e6, 1e6], dtype=np.float32)
        self.obs_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Update observation space to normalized range
        self.observation_space = spaces.Box(
            low=np.zeros(10, dtype=np.float32),
            high=np.ones(10, dtype=np.float32),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Normalize observation to [0, 1] range."""
        # Clip to valid range first (in case of overflow)
        obs = np.clip(obs, self.obs_min, self.obs_max)
        
        # Normalize: (obs - min) / (max - min)
        normalized_obs = (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-8)
        
        return normalized_obs.astype(np.float32)


class RewardScalingWrapper(gym.RewardWrapper):
    """
    Scales rewards to a more manageable range for training.
    
    Multi-objective rewards can range from -100 to +500 (or more),
    which can cause instability in training. This wrapper clips and scales.
    """
    
    def __init__(self, env, scale=0.01, clip_range=(-100, 100)):
        super().__init__(env)
        self.scale = scale
        self.clip_min, self.clip_max = clip_range
    
    def reward(self, reward):
        """Clip and scale reward."""
        # Clip to reasonable range
        reward = np.clip(reward, self.clip_min, self.clip_max)
        
        # Scale down
        return reward * self.scale


class EarlyTerminationWrapper(gym.Wrapper):
    """
    Terminates episode early if population drops to critical levels.
    This helps the agent learn that extinction is bad without waiting
    for the full 100-step episode.
    """
    
    def __init__(self, env, min_population=100):
        super().__init__(env)
        self.min_population = min_population
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Early termination if population too low
        if info.get('population', 0) < self.min_population:
            done = True
            reward = -100  # Severe penalty for extinction
            
        return obs, reward, done, truncated, info
