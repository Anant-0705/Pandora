import os
import sys
import argparse
import importlib.util
import gymnasium as gym

# Add parent directory to path so we can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from environment.pandora_env import PandoraEnv
from environment.wrappers import NormalizeObservationWrapper, RewardScalingWrapper, EarlyTerminationWrapper

class FlattenActionWrapper(gym.ActionWrapper):
    """Converts MultiDiscrete([12,12,12]) to Discrete(1728) for DQN."""
    def __init__(self, env):
        super().__init__(env)
        # Updated for 12 actions: 12*12*12 = 1728
        self.action_space = gym.spaces.Discrete(144)  # 12x12 to reduce action space
        
    def action(self, act):
        if hasattr(act, "item"):
            act = act.item()
        # Simplify to 2 actions instead of 3 to reduce DQN complexity
        # Map to [action1, action2, action1] pattern for compatibility
        a1 = act // 12
        a2 = act % 12
        return [a1, a2, a1]  # Reuse first action as third

def make_env(algo_name, seed):
    """Create and wrap environment with normalization and scaling."""
    env = PandoraEnv(seed=seed)
    
    # Add wrappers for better learning
    env = NormalizeObservationWrapper(env)
    env = RewardScalingWrapper(env, scale=0.01, clip_range=(-100, 100))
    env = EarlyTerminationWrapper(env, min_population=100)
    
    # DQN needs flattened action space
    if algo_name.lower() == 'dqn':
        env = FlattenActionWrapper(env)
    
    return env

def train_agent(algo_name: str, timesteps: int, seed: int = 42, n_envs: int = 4):
    """
    Train an RL agent with optimized hyperparameters.
    
    Args:
        algo_name: 'ppo' or 'dqn'
        timesteps: Total training timesteps
            - PPO: Recommended 1M-1.5M for this sparse-reward problem
            - DQN: Recommended 300k-500k
        seed: Random seed
        n_envs: Number of parallel environments (increases learning speed)
    """
    # Make vectorized environment for parallel training
    env = make_vec_env(lambda: make_env(algo_name, seed), n_envs=n_envs, seed=seed)
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/pandora_{algo_name.lower()}"
    
    print(f"\n{'='*60}")
    print(f"Training {algo_name.upper()} Agent")
    print(f"{'='*60}")
    print(f"Total timesteps: {timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    tensorboard_log_dir = f"./logs/{algo_name}_tensorboard/" if tensorboard_available else None
    if not tensorboard_available:
        print("TensorBoard not installed; continuing without tensorboard logging.")

    if algo_name.lower() == 'ppo':
        # PPO Hyperparameters optimized for long-horizon sparse-reward tasks
        # NOTE: PPO needs MORE training time than DQN for this problem!
        # Recommended: 1M-1.5M timesteps for good results
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,          # Lower LR for stability
            n_steps=2048,                # Longer rollouts for credit assignment
            batch_size=128,              # Increased from 64 for more stable updates
            n_epochs=15,                 # Increased from 10 for better data utilization
            gamma=0.99,                  # High discount for long-term planning
            gae_lambda=0.97,             # Increased from 0.95 for better credit assignment
            clip_range=0.2,              # PPO clipping
            ent_coef=0.02,               # Increased from 0.01 for more exploration
            vf_coef=0.5,                 # Value function coefficient
            max_grad_norm=0.5,           # Gradient clipping
            policy_kwargs=dict(
                net_arch=[256, 256]      # Larger network for complex state space
            ),
            verbose=1,
            tensorboard_log=tensorboard_log_dir
        )
    elif algo_name.lower() == 'dqn':
        # DQN Hyperparameters optimized for discrete action space
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,          # Lower LR for stability
            buffer_size=100000,          # Large replay buffer
            learning_starts=10000,       # Start learning after collecting experience
            batch_size=64,               # Standard batch size
            gamma=0.99,                  # High discount for long-term planning
            train_freq=4,                # Update every 4 steps
            gradient_steps=1,            # One gradient step per update
            target_update_interval=1000, # Update target network frequency
            exploration_fraction=0.3,    # Longer exploration period (30% of training)
            exploration_initial_eps=1.0, # Start with full exploration
            exploration_final_eps=0.05,  # End with 5% exploration
            policy_kwargs=dict(
                net_arch=[256, 256]      # Larger network for complex state space
            ),
            verbose=1,
            tensorboard_log=tensorboard_log_dir
        )
    else:
        raise ValueError("Unsupported algorithm. Choice: 'ppo', 'dqn'")
    
    print(f"Starting training...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(model_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Model saved to {model_path}.zip")
    print(f"{'='*60}\n")
    
    # Close environment
    env.close()
    
    return model
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL Agent for Pandora Environment")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"],
                        help="Algorithm to use (ppo or dqn)")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps (default: 100,000; recommended: 1M+ for PPO, 300k+ for DQN)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    
    train_agent(args.algo, args.timesteps, args.seed, args.n_envs)
