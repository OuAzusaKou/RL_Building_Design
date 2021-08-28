import os

import numpy as np
import torch
from stable_baselines3 import SAC, DDPG, TD3, PPO,DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

from building_env import Building_Env
from discrete_building_env import Discrete_Building_Env
from move_env.discrete5_move_env import Discrete5_Move_DQN_Env

torch.backends.cudnn.enabled = True

env = Discrete5_Move_DQN_Env()
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "./dqn_saved_model/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
print(env)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

obs = env.reset()
# The noise objects for DDPG
n_actions = env.action_space.shape
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.8 * np.ones(n_actions))
n_sampled_goal =4
# model = DQN("MlpPolicy", env, verbose=1,
#     exploration_initial_eps = 1,
#     learning_rate=6e-7,
#     buffer_size=int(1e8),
#     gamma=0.95,
#     policy_kwargs=dict(net_arch=[256,256,256,256,256]),
#     batch_size=512,tensorboard_log="./DQN_tensorboard/")
model = DQN.load("./dqn_saved_model/best_model.zip",env=env,
    exploration_initial_eps = 0.1,
    learning_rate=6e-7,
    buffer_size=int(1e8),
    gamma=0.95,
    policy_kwargs=dict(net_arch=[256,256,256,256,256]),
    batch_size=512,tensorboard_log="./DQN_tensorboard/")
#model = DQN.load("./ddpg_saved_model/best_model.zip",env=env,learning_rate=3e-10,action_noise=action_noise)
model.learn(total_timesteps=int(3e7),tb_log_name="first_run",callback=callback)
model.save("dqn_pendulum")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_pendulum")

obs = env.reset()


# Evaluate the agent
episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done :
        print("Reward:", episode_reward)
        episode_reward = 0.0
        obs = env.reset()