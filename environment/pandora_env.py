import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.world_engine import WorldEngine

ACTIONS = [
    'BOOST_FOOD_PRODUCTION', 'INVEST_IN_WATER', 'EXPAND_TERRITORY',
    'PROMOTE_EQUALITY', 'ENFORCE_ORDER', 'ENCOURAGE_RELIGION',
    'PUSH_SCIENCE', 'INDUSTRIALIZE', 'GO_GREEN', 'INSTALL_AI_GOVERNANCE'
]

class PandoraEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, seed: int = 42):
        super().__init__()
        self.engine = WorldEngine(seed=seed)
        
        # We need an action space that allows picking 3 actions.
        self.action_space = spaces.MultiDiscrete([len(ACTIONS)] * 3)
        
        # Observation space configuration
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([10000, 1e12, 1, 1, 20, 1, 1e6, 1e6, 1e6, 1e6]),
            dtype=np.float32
        )
        
    def _get_obs(self):
        s = self.engine.state
        return np.array([
            s.year,
            s.population,
            s.happiness,
            s.inequality,
            s.tech_level,
            s.climate_health,
            s.resources.get('food', 0),
            s.resources.get('water', 0),
            s.resources.get('energy', 0),
            s.resources.get('minerals', 0)
        ], dtype=np.float32)

    def step(self, action):
        if hasattr(action, 'shape') and len(action.shape) == 0:
            act_val = int(action.item())
            action = [act_val // 100, (act_val // 10) % 10, act_val % 10]
        elif isinstance(action, (int, float)):
            act_val = int(action)
            action = [act_val // 100, (act_val // 10) % 10, act_val % 10]
            
        action_names = [ACTIONS[i] for i in action]
        
        state = self.engine.step(action_names)
        obs = self._get_obs()
        
        # Baseline reward loop
        reward = state.happiness * 10.0 + state.tech_level * 5.0
        
        done = state.year >= 10000 or state.population <= 0
        info = {
            'history_log': state.history_log, 
            'year': state.year,
            'population': state.population,
            'state_obj': state # Expose full state for programmatic grader
        }
        
        return obs, reward, done, False, info
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.seed = seed if seed else 42
        self.engine.reset()
        obs = self._get_obs()
        return obs, {}
