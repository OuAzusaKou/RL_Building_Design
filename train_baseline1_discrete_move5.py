import os

import gym
import numpy as np
from stable_baselines.bench import Monitor, load_results
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.results_plotter import ts2xy

from move_env.discrete5_move_env import Discrete5_Move_DQN_Env

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
log_dir = "./rnn_ppo_saved_model/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
print(env)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)


model = PPO2(MlpLstmPolicy, env, verbose=1,nminibatches=1)
model.learn(total_timesteps=int(3e7))
model.save("rnn_ppo2")

del model # remove to demonstrate saving and loading

model = PPO2.load("rnn_ppo2")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()