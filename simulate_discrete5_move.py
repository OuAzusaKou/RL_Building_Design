import numpy as np
from PIL import Image
from stable_baselines3 import DQN

from move_env.discrete5_move_env import Discrete5_Move_DQN_Env

env = Discrete5_Move_DQN_Env()

model = DQN.load("./dqn_saved_model/best_model.zip",env=env)

obs = env.reset()


# Evaluate the agent
episode_reward = 0
for _ in range(503):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()
    episode_reward += reward
    if done :
        print("Reward:", episode_reward)
        episode_reward = 0.0
        obs = env.render()
        print(env.state)
        obs = obs.copy()
        obs = np.squeeze(obs, 0)
        new_map = Image.fromarray(obs.astype('uint8'), mode='L')

        new_map.show()
        obs = env.reset()
