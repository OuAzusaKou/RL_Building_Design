import numpy as np
import gym
import pandas as pd
import torch
from PIL import Image
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from move_env.rule_score import Score_relation

ROOM_NAMES = [
    'room1', 'room2', 'room3','room4','room5','boundary'
]

MATRIX_TARGET = [
    [-1,2,2,2,2,6],
    [2,-1,1,1,2,6],
    [2,1,-1,2,2,6],
    [2,1,2,-1,2,6],
    [2,2,2,2,-1,6]
]

lis = []
for i in ROOM_NAMES[:6]:
    new_name = i + '_'
    lis.append(new_name)

RELATION_TARGET = pd.DataFrame(MATRIX_TARGET, columns = ROOM_NAMES[:6], index = lis[:5])
RELATION_TARGET.to_csv('../supervised_learning/data_cls/'+'train_6.csv')
RELATION_TARGET = pd.read_csv('../supervised_learning/data_cls/'+'train_6.csv',index_col=0)
print(RELATION_TARGET)



class Discrete5_Move_DQN_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """


    def __init__(self, grid_size = np.array([45,120])):
        super(Discrete5_Move_DQN_Env, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.cuber_num = 5
        self.grid_size = grid_size
        self.step_length = 6
        self.limit = np.zeros((self.cuber_num+1,4))

        self.limit[1] = np.array([36,42,42,54])

        self.limit[2] = np.array([26,42,42,54])

        self.limit[3] = np.array([15, 36, 15, 36])

        self.limit[4] = np.array([15, 36, 15, 36])

        self.limit[5] = np.array([9, 45, 9, 120])

        self.limit[0] = np.array([36,54,100,150])

        self.action_space = spaces.Discrete(self.cuber_num*8)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high= self.grid_size[1],shape=(self.cuber_num*4,), dtype=np.float32)

        self.count = 0

        self.state = np.zeros((self.cuber_num,4))

        self.Score_mach = Score_relation(0, 0, self.state, self.grid_size)

        #self.state[0] = self.grid_size[0]

        #self.state[1] = self.grid_size[1]


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.count = 0

        self.state = np.zeros((self.cuber_num,4))

        self.init_state()

        self.obs = np.zeros((1,self.grid_size[0],self.grid_size[1]))
        '''
        self.obs[0] = (self.state[0] - ( self.limit[0][0] + self.limit[0][1] ) / 2)\
                        /( (self.limit[0][1] - self.limit[0][0])/2 )

        self.obs[1] = (self.state[1] - ( self.limit[0][2] + self.limit[0][3] ) / 2)\
                        /( (self.limit[0][3] - self.limit[0][2])/2 )
        '''

        obs = self._get_obs()

        self.score_old = self.get_reward()
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return obs

    def step(self, action):

        done = False

        reward = 0

        info={}


        self.count = self.count + 1

        #print(action)

        self.excute_action(action)

        #print('state0',self.state[self.count-1, 0])
        #print('state1', self.state[self.count - 1, 1])
        #print('state2', self.state[self.count - 1, 2])
        #print('state3', self.state[self.count - 1, 3])

        '''
        x_buf_start = round(self.state[self.count - 1][0] - self.state[self.count - 1][2] / 2)

        x_buf_end = round(self.state[self.count - 1][0] + self.state[self.count - 1][2] / 2)

        y_buf_start = round(self.state[self.count - 1][1] - self.state[self.count - 1][3] / 2)

        y_buf_end = round(self.state[self.count - 1][1] + self.state[self.count - 1][3] / 2)

        if x_buf_start < 0 :
            x_buf_start = 0

        if x_buf_end > self.grid_size[0]:
            x_buf_end = int(self.grid_size[0])

        if y_buf_start < 0 :
            y_buf_start = 0

        if y_buf_end > self.grid_size[1]:
            y_buf_end = int(self.grid_size[1])
        '''
        #print('x_buf_start',x_buf_start)
        #print('x_buf_end',x_buf_end)
        #print('y_buf_start',y_buf_start)
        #print('y_buf_end',y_buf_end)
        '''
        if (sum(sum(self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end]))) > 0:

            done = True

            obs = self.obs.copy()

            info = {}

            obs_buf = np.zeros_like(self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end])
            obs_buf[self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end] > 0] = 1

            reward_buf = sum(sum(obs_buf))/200
            self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = self.count * 40

            #reward = reward - reward_buf

            return obs, reward, done, info
        else:

            reward = 50
        '''


        score_new =   self.get_reward()

        reward = self.score_old - score_new

        self.score_old = score_new.copy()

        #self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end] = self.count*40

        info = {}

        if self.count > 500:
            done = True

        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='console'):
        self.change_human_obs()
        print(self.obs)
        obs=(self.obs+40).copy()

        obs = np.squeeze(obs, 0)
        #new_map = Image.fromarray(obs.astype('uint8'), mode='L')
        #new_map.show()
        #new_map.save('./gail_img/t'+'.gif')
        return obs

    def close(self):
        pass

    def compute_episode_reward(self):
        obs_buf = np.zeros((1,self.grid_size[0],self.grid_size[1]))
        obs_buf[self.obs == 0] = 1
        print(300/(sum(sum(sum(obs_buf)))+1))

        reward_ = 10 + 50000/(sum(sum(sum(obs_buf)))+1)

        return reward_

    def _get_obs(self):
        """
        Helper to create the observation.

        :return: The current observation.
        """


        return self.state.reshape((20,)).copy()

    def init_state(self):

        for i in range(self.cuber_num):

            self.state[i,2] = self.limit[i+1,0]

            self.state[i,3] = self.limit[i+1,2]

        return

    def excute_action(self, action):
        #print(action)
        excute_num = int(action / 8 )
        #print(action)
        excute_act = int(action % 8 )

        if excute_act == 0:
            self.state[excute_num, 0] += self.step_length
            if self.state[excute_num, 0] > self.grid_size[0]:
                self.state[excute_num, 0] = self.grid_size[0]
        elif excute_act == 1:
            self.state[excute_num, 0] -= self.step_length
            if self.state[excute_num, 0] < 0:
                self.state[excute_num, 0] = 0
        elif excute_act == 2:
            self.state[excute_num, 1] += self.step_length
            if self.state[excute_num, 1] > self.grid_size[1]:
                self.state[excute_num, 1] = self.grid_size[1]
        elif excute_act == 3:
            self.state[excute_num, 1] -= self.step_length
            if self.state[excute_num, 1] < 0:
                self.state[excute_num, 1] = 0
        elif excute_act == 4:
            self.state[excute_num, 2] += self.step_length
            if self.state[excute_num, 2] > self.limit[excute_num+1, 1]:
                self.state[excute_num, 2] = self.limit[excute_num+1, 1]
        elif excute_act == 5:
            self.state[excute_num, 2] -= self.step_length
            if self.state[excute_num, 2] < self.limit[excute_num+1, 0]:
                self.state[excute_num, 2] = self.limit[excute_num+1, 0]
        elif excute_act == 6:
            self.state[excute_num, 3] += self.step_length
            if self.state[excute_num, 3] > self.limit[excute_num+1, 3]:
                self.state[excute_num, 3] = self.limit[excute_num+1, 3]
        elif excute_act == 7:
            self.state[excute_num, 3] -= self.step_length
            if self.state[excute_num, 3] < self.limit[excute_num+1, 2]:
                self.state[excute_num, 3] = self.limit[excute_num+1, 2]
        else:
            print('error')

        return

    def get_reward(self):

        reward_ = 0

        for i in range(5):
            for j in range(5):
                #print(RELATION_TARGET.iloc[i, j])
                self.Score_mach.reset_room(room1 = i, room2 = j ,state= self.state, grid_size= self.grid_size)
                if RELATION_TARGET.iloc[i, j] == -1:
                    pass
                elif RELATION_TARGET.iloc[i, j] == 2:
                    reward_ += self.Score_mach.need_externally_tangent()
                elif RELATION_TARGET.iloc[i, j] == 1:
                    reward_ += self.Score_mach.need_seperated()
            reward_+= self.Score_mach.need_inside_boundary(boundary= 0, room = i)
        area = 0
        for i in range(5):
            area += (self.state[i,2]) * (self.state[i,3])

        reward_ += self.grid_size[0]*self.grid_size[1] - area

        #print(reward_)
        return reward_

    def change_human_obs(self):
        self.obs = np.zeros((1,self.grid_size[0],self.grid_size[1]))
        for count in range(5):

            x_buf_start = round(self.state[count][0] - self.state[count][2] / 2)

            x_buf_end = round(self.state[count][0] + self.state[count][2] / 2)

            y_buf_start = round(self.state[count][1] - self.state[count][3] / 2)

            y_buf_end = round(self.state[count][1] + self.state[count][3] / 2)

            #print(x_buf_start.grad_fn)
            if x_buf_start < 0:
                x_buf_start = 0

            if x_buf_end > self.grid_size[0]:
                x_buf_end = int(self.grid_size[0])

            if y_buf_start < 0:
                y_buf_start = 0

            if y_buf_end > self.grid_size[1]:
                y_buf_end = int(self.grid_size[1])

            self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = (count+1) * 40

            print('obs',self.obs)

        return


#env = Discrete5_Move_DQN_Env()
# It will check your custom environment and output additional warnings if needed
#check_env(env)