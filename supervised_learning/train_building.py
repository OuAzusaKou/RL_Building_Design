import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from tensorboardX import SummaryWriter

from supervised_learning.dataset import CustomSoundDataset
from supervised_learning.loss import rl_reward_loss
from supervised_learning.model import GEN_MLP
from supervised_learning.test_loop import test_loop
from supervised_learning.train_loop import train_loop
from supervised_learning.weightinit import weight_init

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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

def out_transform(data):
  output_data = np.array(data).reshape((5, 6))
  #print(output_data)
  return output_data

if __name__ == '__main__':
  torch.cuda.empty_cache()
  # print(waveform.size())

  # define transformation
  batch_size = 256


  torch.set_default_tensor_type(torch.FloatTensor)
  learning_rate = 1e-3
  #model = NeuralNetwork()
  model = GEN_MLP().to(device)
  loss_fn = rl_reward_loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001)
  train_labels_dir = './train_data_label.csv'
  test_labels_dir = './train_data_label.csv'
  data_dir = './data_cls'
  trained_Flag = False

  training_data = CustomSoundDataset(train_labels_dir, data_dir, transform=[input_transform],
                                     target_transform=[out_transform])
  test_data = CustomSoundDataset(test_labels_dir, data_dir, transform=[input_transform],
                                 target_transform=[out_transform])
  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

  epochs = 50000

  writer = SummaryWriter('runs/scalar_example')
  model.apply(weight_init)

  if trained_Flag == True:
    model.load_state_dict(torch.load('snn_model_weights.pth'))
  for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")

    train_loop(train_dataloader, model, loss_fn, optimizer, writer, t, device)

    if t % 100 == 0:
      torch.save(model.state_dict(), 'snn_model_weights.pth')
      model.load_state_dict(torch.load('snn_model_weights.pth'))
      #model.eval()
      #test_loop(test_dataloader, model, loss_fn, writer, t, device)
  print("Done!")

