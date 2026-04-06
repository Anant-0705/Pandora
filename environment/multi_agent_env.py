from environment.pandora_env import PandoraEnv
from environment.wrappers import NormalizeObservationWrapper, RewardScalingWrapper, EarlyTerminationWrapper
import gymnasium as gym

class FlattenActionWrapper(gym.ActionWrapper):
    """Converts MultiDiscrete([12,12,12]) to Discrete(144) for DQN."""
    def __init__(self, env):
        super().__init__(env)
        # Updated for 12 actions: 12*12 = 144
        self.action_space = gym.spaces.Discrete(144)
        
    def action(self, act):
        if hasattr(act, "item"):
            act = act.item()
        # Simplify to 2 actions instead of 3 to reduce DQN complexity
        # Map to [action1, action2, action1] pattern for compatibility
        a1 = act // 12
        a2 = act % 12
        return [a1, a2, a1]  # Reuse first action as third

class MultiAgentPandoraEnv:
    def __init__(self, seed: int = 42):
        # Ensure identical starting timelines WITH WRAPPERS (critical for RL models!)
        
        # PPO environment with wrappers
        ppo_env = PandoraEnv(seed=seed)
        ppo_env = NormalizeObservationWrapper(ppo_env)
        ppo_env = RewardScalingWrapper(ppo_env, scale=0.01, clip_range=(-100, 100))
        ppo_env = EarlyTerminationWrapper(ppo_env, min_population=100)
        
        # DQN environment with wrappers + flattened action space
        dqn_env = PandoraEnv(seed=seed)
        dqn_env = NormalizeObservationWrapper(dqn_env)
        dqn_env = RewardScalingWrapper(dqn_env, scale=0.01, clip_range=(-100, 100))
        dqn_env = EarlyTerminationWrapper(dqn_env, min_population=100)
        dqn_env = FlattenActionWrapper(dqn_env)
        
        # LLM environment (raw, no wrappers needed for LLM)
        llm_env = PandoraEnv(seed=seed)
        
        self.envs = {
            'ppo': ppo_env,
            'dqn': dqn_env,
            'llm': llm_env,
        }
        self.current_turn = 0

    def step_all(self, actions_dict: dict):
        """
        Expects a dictionary bridging the agents:
        actions_dict = {'ppo': [0, 4, 3], 'dqn': [1, 2, 7], 'llm': [3, 9, 8]}
        """
        results = {}
        self.current_turn += 1
        
        for agent_id, action in actions_dict.items():
            if agent_id in self.envs:
                obs, reward, done, truncated, info = self.envs[agent_id].step(action)
                results[agent_id] = (obs, reward, done, info)

        return results
        
    def reset_all(self):
        """ Resets all independent timelines """
        obs_dict = {}
        info_dict = {}
        for agent_id, env in self.envs.items():
            obs, info = env.reset()
            obs_dict[agent_id] = obs
            info_dict[agent_id] = info
        self.current_turn = 0
        return obs_dict, info_dict
