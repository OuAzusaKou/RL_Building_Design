import numpy as np

#from garl_discrete_env import GAIL_Dis_Env
from move_env.discrete5_move_env import Discrete5_Move_DQN_Env

env = Discrete5_Move_DQN_Env()

obs = env.reset()
env.render()


def wait_for_instruction(obs):
    while True:
        action_buf = input('action_')
        action=(float(action_buf))
        action = np.array(action)
        print(action,type(action))
        try:
            if action>=0 and action<40:
                print('right_action')
                break
        except:
            print('input_error')
    return action


while True:
    action = wait_for_instruction(obs)
    obs, reward, done, info = env.step(action)
    print(done)
    env.render()
    if done:
        break