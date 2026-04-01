from environment.pandora_env import PandoraEnv
import gymnasium as gym

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(1000)
        
    def action(self, act):
        if hasattr(act, "item"):
            act = act.item()
        return [act // 100, (act // 10) % 10, act % 10]

class MultiAgentPandoraEnv:
    def __init__(self, seed: int = 42):
        # Ensure identical starting timelines
        self.envs = {
            'ppo': PandoraEnv(seed=seed),
            'dqn': FlattenActionWrapper(PandoraEnv(seed=seed)),
            'llm': PandoraEnv(seed=seed),
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
