import numpy as np
import torch
from torch import nn

from move_env.discrete5_move_env import Discrete5_Move_DQN_Env
from move_env.rule_score import Score_relation
import csv

class Degree_loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss()

  def forward(self, x, y):

    y=torch.div(y,180)
    #print(y.size())
    x=x.flatten()
    #print(x.size())
    a=torch.stack([(x-y).pow(2),(x-y-2).pow(2),(x-y+2).pow(2)],dim=1)
    b=torch.min(a,dim=1)[0]
    #print(b.size())
    return torch.mean(b)


class rl_reward_loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.cuber_num = 5
    self.grid_size = torch.from_numpy(np.array([45,120]))
    self.state = np.zeros((self.cuber_num, 4))
    self.Score_mach = Score_relation(0, 0, self.state, self.grid_size)
    self.env = Discrete5_Move_DQN_Env()


  def forward(self, x, RELATION_TARGET):
    self.state = x.reshape((self.cuber_num,4))
    print(self.state)
    RELATION_TARGET = RELATION_TARGET.reshape((5,6))
    loss1 = self.get_reward(RELATION_TARGET)
    # self.env.reset()
    # self.env.state = self.state.reshape((5, 4))
    # obs = self.env.render()
    # print(obs)
    # #loss2 = sum(sum(obs < 41)) * 0.05
    #loss2 = sum(obs[:,0] < 41)*5
    loss2 = 0
    area = 0
    min_weight = []
    min_height = []
    max_weight = []
    max_height = []
    for count in range(5):

        area += (self.state[count, 2]) * (self.state[count, 3])

        x_buf_start = torch.round(self.state[count][0] - self.state[count][2] / 2)

        x_buf_end = torch.round(self.state[count][0] + self.state[count][2] / 2)

        y_buf_start = torch.round(self.state[count][1] - self.state[count][3] / 2)

        y_buf_end = torch.round(self.state[count][1] + self.state[count][3] / 2)

        # print(x_buf_start.grad_fn)
        if x_buf_start < 0:
            x_buf_start = 0

        if x_buf_end > self.grid_size[0]:
            x_buf_end = (self.grid_size[0])

        if y_buf_start < 0:
            y_buf_start = 0

        if y_buf_end > self.grid_size[1]:
            y_buf_end = (self.grid_size[1])

        min_weight.append(x_buf_start)
        max_weight.append(x_buf_end)
        min_height.append(y_buf_start)
        max_height.append(y_buf_end)

    # for i in range(5):
    #     area += (self.state[i, 2]) * (self.state[i, 3])
    #
    #     min_weight.append(self.state[i,2])



    loss3 = min(min_height) + min(min_weight) - max(max_weight)
    print('min',min(min_height))
    loss2 += self.grid_size[0] * self.grid_size[1] - area
    loss2 = loss2*0.5

    if loss2 < 0:
        loss2 = 0


    #loss2 = torch.tensor(loss2,=True)
    #loss2.requires_grad = True
    print('loss2',loss2)
    print('loss1',loss1)
    print('loss3',loss3)
    return loss1+loss2+loss3


  def get_reward(self,RELATION_TARGET):

      reward_ = 0

      for i in range(5):
          for j in range(5):
              #print(RELATION_TARGET[i, j])
              self.Score_mach.reset_room(room1 = i, room2 = j ,state= self.state, grid_size= self.grid_size)
              if RELATION_TARGET[i, j] == -1:
                  pass
              elif RELATION_TARGET[i, j] == 2:
                  reward_ += self.Score_mach.need_externally_tangent()
              elif RELATION_TARGET[i, j] == 1:
                  reward_ += self.Score_mach.need_seperated()
              reward_ += self.Score_mach.union_set()*0.1
          reward_+= self.Score_mach.need_inside_boundary(boundary= 0, room = i)
      #print(reward_)
      return reward_