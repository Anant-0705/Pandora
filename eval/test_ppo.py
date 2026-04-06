import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from environment.pandora_env import PandoraEnv
from environment.wrappers import NormalizeObservationWrapper, RewardScalingWrapper, EarlyTerminationWrapper

# Create environment with same wrappers as training
env = PandoraEnv()
env = NormalizeObservationWrapper(env)
env = RewardScalingWrapper(env, scale=0.01, clip_range=(-100, 100))
env = EarlyTerminationWrapper(env, min_population=100)

model = PPO.load("models/pandora_ppo")

obs, _ = env.reset()

total_reward = 0
step_count = 0

print("\n" + "="*60)
print("Testing PPO Model on Pandora Environment")
print("="*60 + "\n")

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic for testing
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1
    
    # Print progress every 10 steps (1000 years)
    if step % 10 == 0:
        print(f"Year {info.get('year', 0):>5} | Population: {info.get('population', 0):>12,} | Reward: {reward:>8.3f}")

    if done or truncated:
        break

print("\n" + "="*60)
print(f"Simulation Complete!")
print("="*60)
print(f"Steps completed: {step_count}")
print(f"Final year: {info.get('year', 0)}")
print(f"Final population: {info.get('population', 0):,}")
print(f"Total reward: {total_reward:.2f}")
print("="*60 + "\n")
