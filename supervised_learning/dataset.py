import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomSoundDataset(Dataset):
  def __init__(self, labels_dir, data_dir, transform=None, target_transform=None):
    self.labels_dir = pd.read_csv(labels_dir)

    self.data_dir = data_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.labels_dir)

  def __getitem__(self, idx):
    data1_path = os.path.join(self.data_dir, self.labels_dir.iloc[idx, 0])
    data2_path = os.path.join(self.data_dir, self.labels_dir.iloc[idx, 1])
    #print(sound_dir)
    #print(self.labels_dir.iloc[idx,:])
    #print(sound_path)
    data1 = pd.read_csv(data1_path,index_col=0)
    data2 = pd.read_csv(data2_path)
    label = pd.read_csv(data1_path,index_col=0)
    #label = self.labels_dir.iloc[idx, 2]
    if self.transform:
      for trans in self.transform:
        data = trans(data1,data2)
    if self.target_transform:
      for trans in self.target_transform:
        label = trans(label)
    # if label == 1:
    #   label = torch.tensor([0,1])
    # else :
    #   label = torch.tensor([1,0])
    sample = {"data": data, "label": label}
    #print(sample['sound'].size)
    return sample