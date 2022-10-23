import torch
from torch import nn
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import Sigmoid
import sys
import os
from collections import Counter
import string
import numpy as np
import argparse

class LoadData(torch.utils.data.Dataset):
    def __init__(self,args):
        self.args = args
        self.data = self.get_data(args.data_path)

    def get_data(self, data_path):
        ret = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                category, cores, memory, disk, time = line.split(', ')
                l = [int(category), int(cores), int(memory), int(disk), float(time)]
                ret.append(l)
        return ret

    def __len__(self):
        return len(self.data) - self.args.seqLength

    def __getitem__(self, index):
        #category_new, category, cores, memory, disk, time
        #category, *cons = self.data[index]
        X = []
        y = []
        for i in range(self.args.seqLength):
            category, *cons = self.data[index+i]
            if index+i == 0:
                X.append([category] + self.data[index+i])
                y.append(cons)
            else:
                X.append([category] + self.data[index+i-1])
                y.append(cons)
        return torch.Tensor(X), torch.Tensor(y)

        # category, *cons = self.data[index]    
        # if index == 0:         
        #     X = torch.Tensor([category] + self.data[index])
        #     X = X[None, :]
        #     y = torch.Tensor(cons)
        #     y = y[None, :]
        #     return X, y
        # else:
        #     X = torch.Tensor([category] + self.data[index-1])
        #     X = X[None, :]
        #     y = torch.Tensor(cons)
        #     y = y[None, :]
        #     return X, y