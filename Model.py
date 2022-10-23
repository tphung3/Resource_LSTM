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
from LoadData import LoadData
from MyLoss import MyLoss




class Model(nn.Module):
    def __init__(self, dataset, args):
        super(Model, self).__init__()

        # Define input dimension of RNN.
        self.inputSize = args.inputSize

        #cores, memory, disk, time
        self.hiddenSize = args.hiddenSize

        # Define the number of layers of the RNN.
        self.numLayers = args.numLayers

        self.rnnUnit = nn.LSTM(input_size=self.inputSize,
                            hidden_size=self.hiddenSize,
                            num_layers=self.numLayers,
        )

        self.fc = nn.Linear(self.hiddenSize, self.hiddenSize)

        #self.relu = nn.ReLU()

    def forward(self, X, prevState):
        output, state = self.rnnUnit(X, prevState)
        #print('prev_state is:', prevState)
        output = self.fc(output)
        #output = self.relu(output)
        return output, state

    def initState(self, seqLength):

        stateHidden = torch.zeros(self.numLayers, seqLength, self.hiddenSize)
        stateCurrent = torch.zeros(self.numLayers, seqLength, self.hiddenSize)
        return (stateHidden, stateCurrent)

    def evaluate(self, model, validate_data, device, prevState, args):
        model.eval()
        best_loss = float('inf')
        for data in validate_data:
            inputs, labels = data
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            predictions = model(inputs, prevState)[0]

            # inputs = inputs.detach().cpu().numpy()
            # labels = labels.detach().cpu().numpy()
            # predictions = predictions.detach().cpu().numpy()
            loss_f = MyLoss(args)

            loss = loss_f(predictions, labels)
            if loss < best_loss:
                best_loss = loss

        print(f'Best loss value per batch across validation dataset is {best_loss}')

        return best_loss
    
    def predict(self, X, state):
        output, _ = self.forward(X, state)
        return output