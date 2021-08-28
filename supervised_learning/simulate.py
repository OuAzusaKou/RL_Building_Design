import numpy as np
import pandas as pd
import torch
from PIL import Image

from move_env.discrete5_move_env import Discrete5_Move_DQN_Env
from supervised_learning.model import GEN_MLP



def input_transform(data1,data2):
  input_data1 = np.array(data1).reshape((-1))
  #print('input_data1',type(data1))

  input_data2 = (data2.iloc[0, 0])

  input_data3 = (data2.iloc[0, 1])
  #print(input_data2)
  #print(np.array(data2.strip('\n')))
  input_data2 = np.asmatrix(input_data2)
  #print(np.asarray(data2).reshape((-1)))
  input_data2 = np.asarray(input_data2).reshape((-1))

  input_data3 = np.asmatrix(input_data3)
  #print(np.asarray(data2).reshape((-1)))
  input_data3 = np.asarray(input_data3).reshape((-1))

  input_data = np.concatenate([input_data1,input_data2,input_data3])

  #data2_1 = np.array(data2.iloc[0,0]).reshape((-1))

  #print(data2_1)

  #data2_2 = np.array(data2.iloc[0,1]).reshape((-1))

  #input_data2 = np.concatenate([data2_1,data2_2])

  #print(input_data2)
  #print(input_data.shape)
  return input_data

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = GEN_MLP().to(device)

model.load_state_dict(torch.load('snn_model_weights.pth'))

data1_path = './data_cls/train_1.csv'
data2_path ='./data_cls/limit_1.csv'
data1 = pd.read_csv(data1_path, index_col=0)
data2 = pd.read_csv(data2_path)

input_data = input_transform(data1,data2)

input_data = torch.tensor(input_data,dtype=torch.float).unsqueeze(0)

state = model(input_data)

print(state)

env = Discrete5_Move_DQN_Env()


env.state = state.detach().numpy().reshape((5,4))

obs = env.render()

print('obs_shape',obs.shape)

print(env.state)
obs = obs.copy()
#obs = np.squeeze(obs, 0)
new_map = Image.fromarray(obs.astype('uint8'), mode='L')

new_map.show()