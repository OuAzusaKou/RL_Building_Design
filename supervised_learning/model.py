import numpy as np
import torch
from torch import nn

input_size = 30 + 20 + 2

hidden_size = 64

output_size = 20

class GEN_MLP(nn.Module):
  def __init__(self):
    super(GEN_MLP, self).__init__()

    # self.MLP1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
    #
    # self.MLP2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
    #
    # self.MLP3 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
    #
    # self.MLP4 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
    # #self.MLP = nn.Linear(in_features= hidden_size, out_features=1, bias=True)
    #
    # self.MLP5 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
    #
    # self.MLP6 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
    #
    # self.relu = nn.ReLU()
    #
    # self.tanh = nn.Tanh()
    self.mlplayer = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    # 最后一层不需要添加激活函数
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),

      nn.Linear(hidden_size, output_size),

      nn.Tanh(),
    )



  def forward(self, x):
    batch_size = len(x)



    x_identical = x.clone()

    # x = self.MLP1(x)
    #
    # x = self.relu(x)
    #
    # x = self.MLP2(x)
    #
    # x = self.relu(x)
    #
    # x = self.MLP3(x)
    #
    # x = self.relu(x)
    #
    # x = self.MLP4(x)
    #
    # x =self.relu(x)
    #
    # x = self.MLP5(x)
    #
    # x = self.relu(x)
    #
    # x = self.MLP6(x)
    #
    # output_ = self.tanh(x)
    output_ = self.mlplayer(x)

    output = torch.zeros_like(output_)

    for i in range(5):

      output[:, i*4 + 0] = x_identical[:, -2] / 2 + x_identical[:, -2] / 2 * output_[:,i*4 + 0]

      output[:, i*4 + 1] = x_identical[:, -1] /2 + x_identical[:, -1] / 2 * output_[:,i*4 + 1]

      output[:, i*4 + 2] = (x_identical[:, 30 + i*4 + 0] + x_identical[:, 30 + i * 4 + 1]) / 2 + \
                           (x_identical[:, 30 + i * 4 + 1] - x_identical[:, 30 + i*4 + 0]) / 2 * output_[:, i*4 + 2]

      output[:, i*4 + 3] = (x_identical[:, 30 + i * 4 + 2] + x_identical[:, 30 + i * 4 + 3]) / 2 + \
                           (x_identical[:, 30 + i * 4 + 3] - x_identical[:, 30 + i * 4 + 2]) / 2 * output_[:, i*4 + 3]

    #output = self.relu(x)


    # batch_size = len(x)
    # # print('input',x)
    # #print('input_size',x.size())
    # x = self.conv2d1(x)
    # x = self.RelU(x)
    # # print('x_relu',x)
    # #print('cnn1_size',x.size())
    # x = self.BatchNorm1(x)
    # #print('x_batchnorm_size',x.size())
    # x = self.maxpooling1(x)
    # #print('maxpooling1_size',x.size())
    # x = self.conv2d2(x)
    # x = self.RelU(x)
    # #print('cnn2_size',x.size())
    # # print(x.size())
    # x = self.BatchNorm2(x)
    # # print('x_batchnorm',x)
    # x = self.maxpooling2(x)
    # #print('maxpooling2_size',x.size())
    #
    # # print('x_maxpooling',x)
    # #print(x.size())
    # #out, h = self.GRU(x.view(batch_size, output_channels2, -1).permute(0, 2, 1))
    # # print('out_gru',out)
    # # print('outGRU.size',out.size())
    # #x = self.MLP(out[:, -1, :])
    # # print('xout',x)
    # output = self.snn(x.view(batch_size, output_channels2,-1))
    # #output = self.tanh(x)
    # # print('tanh',output)
    # # print('output.size',output.size())

    return output