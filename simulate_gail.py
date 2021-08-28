from stable_baselines import GAIL

from move_env.discrete5_move_env import Discrete5_Move_DQN_Env

env = Discrete5_Move_DQN_Env()

model = GAIL.load("gail_saved_model/best_model.zip")

obs = env.reset()
while True:
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()
  if done:
    print(done)
    break