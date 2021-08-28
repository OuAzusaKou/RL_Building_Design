import numpy as np
import torch
from PIL import Image
from stable_baselines3 import PPO

from discrete_building_env import Discrete_Building_Env
from discrete_vector_action_shape_env import Discrete_Vector_ActionShape_Env
from discrete_vector_env import Discrete_Vector_Env

torch.backends.cudnn.enabled = False
env = Discrete_Vector_ActionShape_Env()

# Load saved model
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = PPO.load("./ppo_saved_model/best_model.zip",env=env)
obs = env.reset()

episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    if done :
        print("Reward:", episode_reward)
        episode_reward = 0.0
        obs = env.render()
        obs = np.squeeze(obs,0)
        new_map = Image.fromarray(obs.astype('uint8'),mode='L')

        new_map.save('./build_map/t'+'.gif')
        obs = env.reset()
        break