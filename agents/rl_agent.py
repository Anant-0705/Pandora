import os
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from environment.pandora_env import PandoraEnv

def train_agent(algo_name: str, timesteps: int, seed: int = 42):
    # make_vec_env wraps out env for efficient multi-processing
    env = make_vec_env(lambda: PandoraEnv(seed=seed), n_envs=1, seed=seed)
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pandora_{algo_name.lower()}"
    
    print(f"Training {algo_name.upper()} Agent for {timesteps} timesteps...")
    
    if algo_name.lower() == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1)
    elif algo_name.lower() == 'dqn':
        # DQN doesn't natively support MultiDiscrete observation out of the box
        # But Stable-baselines3 generally allows it if flattened, or we can use PPO mostly
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
