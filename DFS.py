
from collections import OrderedDict

import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

class Discrete_Vector_DFS():
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """


    def __init__(self, grid_size = np.array([42,120])):
        super(Discrete_Vector_DFS, self).__init__()

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

        self.limit[5] = np.array([9, 42, 9, 120])

        self.limit[0] = np.array([36,54,100,150])

        #self.observation_space = spaces.Box(low=0, high=255,
        #                                    shape=(1,self.grid_size[0],self.grid_size[1]), dtype=np.uint8)

        self.count = 0

        self.state = np.zeros((self.cuber_num,4))
        self.obs = np.zeros((1,self.grid_size[0],self.grid_size[1]))
        #self.state[0] = self.grid_size[0]

        #self.state[1] = self.grid_size[1]




    def step(self, action):

        done = False
        info ={}

        reward = 0



        self.count = self.count + 1

        print(action)

        self.state[self.count-1, 0] = action[0]*6+action[2]*2
        self.state[self.count-1, 1] = action[1]*6+action[3]*2
        #print('limit2',(self.limit[self.count][0]+self.limit[self.count][1]/2))
        #print('limit3',(self.limit[self.count][3]+self.limit[self.count][2]/2))
        self.state[self.count-1, 2] = ((action[4]-2)/2)*((self.limit[self.count][1]-self.limit[self.count][0])/2)\
                                    + ((self.limit[self.count][0]+self.limit[self.count][1])/2)
        self.state[self.count-1, 3] = ((action[5]-2)/2)*((self.limit[self.count][3]-self.limit[self.count][2])/2)\
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
            done = True
            obs = self._get_obs()
            return obs, reward, done, info
        if x_buf_end > self.grid_size[0]:
            x_buf_end = int(self.grid_size[0])
            done = True
            obs = self._get_obs()
            return obs, reward, done, info
        if y_buf_start < 0 :
            y_buf_start = 0
            done = True
            obs = self._get_obs()
            return obs, reward, done, info
        if y_buf_end > self.grid_size[1]:
            y_buf_end = int(self.grid_size[1])
            done = True
            obs = self._get_obs()
            return obs, reward, done, info
        #print('x_buf_start',x_buf_start)
        #print('x_buf_end',x_buf_end)
        #print('y_buf_start',y_buf_start)
        #print('y_buf_end',y_buf_end)
        if (sum(sum(self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end]))) > 0:

            done = True

            info = {}

            obs_buf = np.zeros_like(self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end])
            obs_buf[self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end] > 0] = 1

            reward_buf = sum(sum(obs_buf))/200
            self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = self.count * 40
            #reward = reward - reward_buf

            return done
        else:

            reward = 50 + self.state[self.count-1,2]*self.state[self.count-1,3]/50

        self.obs[0,x_buf_start:x_buf_end,y_buf_start:y_buf_end] = self.count*40

        if self.count == 5:

            done = True

            reward_ = self.compute_episode_reward()

            reward = reward + reward_
        info = {}

        obs = self._get_obs()
        return done

    def render(self, mode='console'):
        obs=self.obs
        return obs


    def _get_obs(self):
        """
        Helper to create the observation.

        :return: The current observation.
        """


        return OrderedDict(
            [
                ("count", self.count),
                ("observation", self.state.copy()),

            ]
        )
    def compute_episode_reward(self):
        obs_buf = np.zeros((1,self.grid_size[0],self.grid_size[1]))
        obs_buf[self.obs == 0] = 1
        print(300/(sum(sum(sum(obs_buf)))+1))

        reward_ = 10 + 50000/(sum(sum(sum(obs_buf)))+1)

        return reward_

    def dfs(self,count):
        for x_center in range (int(self.grid_size[0]/6)):
            for y_center in range (int(self.grid_size[1]/6)):
                for x_offset in range(3):
                    for y_offset in range(3):
                        for x_shape in range(5):
                            for y_shape in range(5):
                                done=self.putin(x_center,y_center,x_offset,y_offset,x_shape,y_shape,count)
                                if done:
                                    x_buf_start = round(self.state[count][0] - self.state[count][2] / 2)

                                    x_buf_end = round(self.state[count][0] + self.state[count][2] / 2)

                                    y_buf_start = round(self.state[count][1] - self.state[count][3] / 2)

                                    y_buf_end = round(self.state[count][1] + self.state[count][3] / 2)
                                    self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = 0
                                    self.state[count] = np.zeros(4, )
                                    continue

                                if count == 4:
                                    self.compute_episode_reward()

                                    x_buf_start = round(self.state[count][0] - self.state[count][2] / 2)

                                    x_buf_end = round(self.state[count][0] + self.state[count][2] / 2)

                                    y_buf_start = round(self.state[count][1] - self.state[count][3] / 2)

                                    y_buf_end = round(self.state[count][1] + self.state[count][3] / 2)
                                    self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = 0
                                    print(self.state)
                                    self.state[count] = np.zeros(4, )
                                    return

                                self.dfs(count+1)
                                x_buf_start = round(self.state[count][0] - self.state[count][2] / 2)

                                x_buf_end = round(self.state[count][0] + self.state[count][2] / 2)

                                y_buf_start = round(self.state[count][1] - self.state[count][3] / 2)

                                y_buf_end = round(self.state[count][1] + self.state[count][3] / 2)
                                self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = 0
                                self.state[count] = np.zeros(4,)


    def putin(self,x_center,y_center,x_offset,y_offset,x_shape,y_shape,count):
        done = False

        self.state[count, 0] = x_center * 6 + x_offset * 2
        self.state[count, 1] = y_center * 6 + y_offset * 2
        # print('limit2',(self.limit[self.count][0]+self.limit[self.count][1]/2))
        # print('limit3',(self.limit[self.count][3]+self.limit[self.count][2]/2))
        self.state[count, 2] = ((x_shape - 2) / 2) * (
                    (self.limit[count+1][1] - self.limit[count+1][0]) / 2) \
                                        + ((self.limit[count+1][0] + self.limit[count+1][1]) / 2)

        self.state[count, 3] = ((y_shape - 2) / 2) * (
                    (self.limit[count+1][3] - self.limit[count+1][2]) / 2) \
                                        + ((self.limit[count+1][3] + self.limit[count+1][2]) / 2)
        #print(self.state[count,3])
        # print('state0',self.state[self.count-1, 0])
        # print('state1', self.state[self.count - 1, 1])
        # print('state2', self.state[self.count - 1, 2])
        # print('state3', self.state[self.count - 1, 3])
        x_buf_start = round(self.state[count][0] - self.state[count][2] / 2)

        x_buf_end = round(self.state[count][0] + self.state[count][2] / 2)

        y_buf_start = round(self.state[count][1] - self.state[count][3] / 2)

        y_buf_end = round(self.state[count][1] + self.state[count][3] / 2)

        if x_buf_start < 0:
            x_buf_start = 0
            done = True
            return done
        if x_buf_end > self.grid_size[0]:
            x_buf_end = int(self.grid_size[0])
            done = True

            return  done
        if y_buf_start < 0:
            y_buf_start = 0
            done = True
            return done
        if y_buf_end > self.grid_size[1]:
            y_buf_end = int(self.grid_size[1])
            done = True
            return done
        # print('x_buf_start',x_buf_start)
        # print('x_buf_end',x_buf_end)
        # print('y_buf_start',y_buf_start)
        # print('y_buf_end',y_buf_end)
        if (sum(sum(self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end]))) > 0:

            done = True

            info = {}

            obs_buf = np.zeros_like(self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end])
            obs_buf[self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] > 0] = 1

            reward_buf = sum(sum(obs_buf)) / 200
            #self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = self.count * 40
            #obs = self._get_obs()
            # reward = reward - reward_buf

            return done

        self.obs[0, x_buf_start:x_buf_end, y_buf_start:y_buf_end] = count * 40



        return done


dd= Discrete_Vector_DFS()
dd.dfs(0)
