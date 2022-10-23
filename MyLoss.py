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

class MyLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, output, label):
        batchedMaxAlloc = [[self.args.maxAlloc] for i in range(self.args.batchSize)]
        sig = nn.Sigmoid()
        ret = torch.Tensor(torch.mean((output-label)**2))
        #print('~~~', ret)
        ret += self.args.lambd * torch.mean((torch.Tensor(batchedMaxAlloc) - label + output) * sig(label-output))
        #print('~~after', ret)
        return ret