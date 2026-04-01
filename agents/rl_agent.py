import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from environment.pandora_env import PandoraEnv

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(1000)
        
    def action(self, act):
        if hasattr(act, "item"):
            act = act.item()
        return [act // 100, (act // 10) % 10, act % 10]

def make_env(algo_name, seed):
    env = PandoraEnv(seed=seed)
    if algo_name.lower() == 'dqn':
        env = FlattenActionWrapper(env)
    return env

def train_agent(algo_name: str, timesteps: int, seed: int = 42):
    # make_vec_env wraps our env for efficient multi-processing
    env = make_vec_env(lambda: make_env(algo_name, seed), n_envs=1, seed=seed)
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pandora_{algo_name.lower()}"
    
    print(f"Training {algo_name.upper()} Agent for {timesteps} timesteps...")
    
    if algo_name.lower() == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1)
    elif algo_name.lower() == 'dqn':
        model = DQN("MlpPolicy", env, verbose=1)
    else:
        raise ValueError("Unsupported algorithm. Choice: 'ppo', 'dqn'")
        
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    
    print(f"Model saved to {model_path}.zip")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL Agent.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()
    
    train_agent(args.algo, args.timesteps)
