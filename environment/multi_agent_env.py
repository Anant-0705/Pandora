from environment.pandora_env import PandoraEnv

class MultiAgentPandoraEnv:
    def __init__(self, seed: int = 42):
        # Ensure identical starting timelines
        self.envs = {
            'ppo': PandoraEnv(seed=seed),
            'dqn': PandoraEnv(seed=seed),
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
