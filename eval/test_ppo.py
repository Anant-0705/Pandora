from stable_baselines3 import PPO
from environment.pandora_env import PandoraEnv

env = PandoraEnv()

model = PPO.load("models/pandora_ppo")

obs, _ = env.reset()

total_reward = 0

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

    if done or truncated:
        break

print("Total Reward:", total_reward)
