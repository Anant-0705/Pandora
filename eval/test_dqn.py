from stable_baselines3 import DQN
from environment.pandora_env import PandoraEnv
from agents.rl_agent import FlattenActionWrapper

env = PandoraEnv()
# We must wrap the env for DQN because DQN outputs a scalar action (0-999) 
# instead of the native MultiDiscrete([10, 10, 10]) expected by PandoraEnv.
env = FlattenActionWrapper(env)

model = DQN.load("models/pandora_dqn")

obs, _ = env.reset()

total_reward = 0

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

    if done or truncated:
        break

print("Total Reward:", total_reward)
