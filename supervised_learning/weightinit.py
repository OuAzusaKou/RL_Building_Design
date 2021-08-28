from torch import nn


def weight_init(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_normal_(m.weight)
    nn.init.constant_(m.bias, 0)
  # 也可以判断是否为conv2d，使用相应的初始化方式
  elif isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out')
  # 是否为批归一化层
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.GRU):
    nn.init.kaiming_normal_(m.weight_ih_l0, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(m.weight_hh_l0, mode='fan_out', nonlinearity='relu')
    # nn.init.kaiming_normal_(m.bias_ih_l0, mode='fan_out',nonlinearity='relu')
    # nn.init.kaiming_normal_(m.bias_hh_l0, mode='fan_out',nonlinearity='relu')
    #nn.init.kaiming_normal_(m.weight_ih_l1, mode='fan_out', nonlinearity='relu')
    #nn.init.kaiming_normal_(m.weight_hh_l1, mode='fan_out', nonlinearity='relu')
    # nn.init.kaiming_normal_(m.bias_ih_l1, mode='fan_out',nonlinearity='relu')
    # nn.init.kaiming_normal_(m.bias_hh_l1, mode='fan_out',nonlinearity='relu')
    #nn.init.kaiming_normal_(m.weight_ih_l2, mode='fan_out', nonlinearity='relu')
    #nn.init.kaiming_normal_(m.weight_hh_l2, mode='fan_out', nonlinearity='relu')
    #nn.init.kaiming_normal_(m.weight_ih_l3, mode='fan_out', nonlinearity='relu')
    #nn.init.kaiming_normal_(m.weight_hh_l3, mode='fan_out', nonlinearity='relu')