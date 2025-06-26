import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MoonDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class RealNVP(nn.Modules):
    def __init__(self):
        super(RealNVP, self).__init__()

    def forward(self, x):
        return x
