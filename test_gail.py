import gym

from stable_baselines import GAIL, SAC
from stable_baselines.bench import Monitor
from stable_baselines.gail import ExpertDataset, generate_expert_traj

# Generate expert trajectories (train expert)
from move_env.discrete5_move_env import Discrete5_Move_DQN_Env
log_dir='./test'
env = Discrete5_Move_DQN_Env()
#env = Monitor(env, log_dir)
env.reset()
def dummy_expert(_obs):

    return env.action_space.sample()
generate_expert_traj(dummy_expert, 'dummy_expert_env', env, n_episodes=10)

# Load the expert dataset
dataset = ExpertDataset(expert_path='dummy_expert_env.npz', traj_limitation=10, verbose=1)

model = GAIL('MlpPolicy', env, dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=1000)
model.save("gail_pendulum")

del model # remove to demonstrate saving and loading

model = GAIL.load("gail_pendulum")

env = gym.make('Pendulum-v0')
obs = env.reset()
while True:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  env.render()