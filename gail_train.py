import os

import gym
import numpy as np
from stable_baselines import GAIL
from stable_baselines.bench import Monitor, load_results
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines.results_plotter import ts2xy

from discrete_vector_action_shape_rewardshaping_env import Discrete_Vector_ActionShape_RewardShaping_Env
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
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                         mean_reward))

        # New best model, you could save the agent here
        if mean_reward > self.best_mean_reward:
          self.best_mean_reward = mean_reward
          # Example for saving best model
          if self.verbose > 0:
            print("Saving new best model to {}".format(self.save_path))
          self.model.save(self.save_path)

    return True


# Create log dir
log_dir = "./gail_saved_model/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
print(env)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

dataset = ExpertDataset(expert_path='dummy_expert_env.npz', traj_limitation=-1, verbose=1)

model = GAIL('MlpPolicy', env, dataset, verbose=1, policy_kwargs=dict(net_arch=[256,256,256,256]),)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=int(1e6), callback=callback)
model.save("gail_pendulum")

del model # remove to demonstrate saving and loading

model = GAIL.load("gail_pendulum")

obs = env.reset()
while True:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  env.render()
  if dones:
    break