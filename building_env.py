import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

class Building_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """


    def __init__(self, grid_size = np.array([42,120])):
        super(Building_Env, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.cuber_num = 5
        self.grid_size = grid_size

        self.limit = np.zeros((self.cuber_num+1,4))

        self.limit[1] = np.array([36,42,42,54])

        self.limit[2] = np.array([26,42,42,54])

        self.limit[3] = np.array([15, 36, 15, 36])

        self.limit[4] = np.array([15, 36, 15, 36])

        self.limit[5] = np.array([9, 54, 9, 150])

        self.limit[0] = np.array([36,54,100,150])

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1,self.grid_size[0],self.grid_size[1]), dtype=np.uint8)

        self.count = 0

        self.state = np.zeros((self.cuber_num,4))

        #self.state[0] = self.grid_size[0]

        #self.state[1] = self.grid_size[1]


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.count = 0

        self.state = np.zeros((self.cuber_num,4))

        self.obs = np.zeros((1,self.grid_size[0],self.grid_size[1]))
        '''
        self.obs[0] = (self.state[0] - ( self.limit[0][0] + self.limit[0][1] ) / 2)\
                        /( (self.limit[0][1] - self.limit[0][0])/2 )

        self.obs[1] = (self.state[1] - ( self.limit[0][2] + self.limit[0][3] ) / 2)\
                        /( (self.limit[0][3] - self.limit[0][2])/2 )
        '''

        obs = self.obs.copy()

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return obs

    def step(self, action):

        done = False

        reward = 0



        self.count = self.count + 1

        print(action)

        self.state[self.count-1, 0] = action[0]*((self.grid_size[0] - 0 )/2)\
                                    + (self.grid_size[0]/2)
        self.state[self.count-1, 1] = action[1]*((self.grid_size[1] - 0)/2)\
                                    + (self.grid_size[1]/2)
        #print('limit2',(self.limit[self.count][0]+self.limit[self.count][1]/2))
        #print('limit3',(self.limit[self.count][3]+self.limit[self.count][2]/2))
        self.state[self.count-1, 2] = action[2]*((self.limit[self.count][1]-self.limit[self.count][0])/2)\
                                    + ((self.limit[self.count][0]+self.limit[self.count][1])/2)
        self.state[self.count-1, 3] = action[3]*((self.limit[self.count][3]-self.limit[self.count][2])/2)\
                                    + ((self.limit[self.count][3]+self.limit[self.count][2])/2)
        #print('state0',self.state[self.count-1, 0])
        #print('state1', self.state[self.count - 1, 1])
        #print('state2', self.state[self.count - 1, 2])
        #print('state3', self.state[self.count - 1, 3])
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
        #print('x_buf_start',x_buf_start)
        #print('x_buf_end',x_buf_end)
        #print('y_buf_start',y_buf_start)
        #print('y_buf_end',y_buf_end)
        if (sum(sum(self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end]))) > 0:

            done = True

            obs = self.obs

            info = {}

            obs_buf = np.zeros_like(self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end])
            obs_buf[self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end] > 0] = 1

            reward_buf = sum(sum(obs_buf))/200
            self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = self.count * 40

            #reward = reward - reward_buf

            return obs, reward, done, info
        else:

            reward = 50

        self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end] = self.count*40

        if self.count == 5:

            done = True

            reward_=self.compute_episode_reward()

            reward = reward + reward_
        info = {}

        obs = self.obs
        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def compute_episode_reward(self):
        obs_buf = np.zeros((1,self.grid_size[0],self.grid_size[1]))
        obs_buf[self.obs == 0] = 1

        reward_ = 10 + 100/(sum(sum(sum(obs_buf)))+1)

        return reward_


env = Building_Env()
# It will check your custom environment and output additional warnings if needed
check_env(env)