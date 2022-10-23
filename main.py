#import relevant libraries and code
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
from Model import Model
import copy

if __name__ == '__main__':

    #parse arguments for model
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="inference_task_resource.dat")
    parser.add_argument('--maxEpochs', type=int, default=70)
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--learningRate', type=float, default=1)
    parser.add_argument('--seqLength', type=int, default=15)
    parser.add_argument('--workingDir', type=str, default='.')
    parser.add_argument('--maxAlloc', type=list, default=[12, 24000, 24000, 1000])
    parser.add_argument('--trainSize', type=float, default=0.3)
    parser.add_argument('--validateSize', type=float, default=0.3)
    parser.add_argument('--inputSize', type=int, default=6)
    parser.add_argument('--hiddenSize', type=int, default=4)
    parser.add_argument('--numLayers', type=int, default=2)
    parser.add_argument('--lambd', type=float, default=2)
    args = parser.parse_args()

    #location to save best model
    try:
        os.mkdir(f"{args.workingDir}/checkpoints")
    except:
        pass

    #cpu or gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("currently using: ", device)

    #patition all dataset into train, validation, and test datasets
    dataset = LoadData(args)
    raw_data = dataset.data
    train_size = int(len(raw_data) * args.trainSize)
    validate_size = train_size + int(len(raw_data) * args.validateSize)
    test_size = len(raw_data) - train_size - validate_size
    train_data = LoadData(args)
    train_data.data = raw_data[:train_size]
    validate_data = LoadData(args)
    validate_data.data = raw_data[train_size:validate_size]
    test_data = LoadData(args)
    test_data.data = raw_data[validate_size:]
    train_data = DataLoader(train_data, batch_size=args.batchSize, drop_last=True)
    validate_data = DataLoader(validate_data, batch_size=args.batchSize, drop_last=True)
    test_data = DataLoader(test_data, batch_size=args.batchSize, drop_last=True)

    #define model
    model = Model(train_data, args)

    #train model
    def train(train_data, validate_data, model, args):

        #define loss function and optimizer
        loss_fn = MyLoss(args)
        optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

        #init current and hidden states
        stateHidden, stateCurrent = model.initState(args.seqLength)
        print('starting states:', stateHidden, stateCurrent)
        bestState = (stateHidden, stateCurrent)

        #record best loss
        bestLoss = float('inf')

        # Training loop starts here.
        for epoch in range(1, args.maxEpochs+1):      

            #switch to train mode
            model.train()

            for batch, (X, y) in enumerate(train_data):
                optimizer.zero_grad()
                y_pred, (stateHidden, stateCurrent) = model(X, (stateHidden, stateCurrent))
                stateHidden = stateHidden.detach()
                stateCurrent = stateCurrent.detach()
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                currLoss = loss.item()

                # Only save models with smallest loss per epoch.
                if currLoss < bestLoss:
                    bestLoss = currLoss
                    bestState = (stateHidden, stateCurrent)
                    torch.save(model.state_dict(), f'{args.workingDir}/checkpoints/-epoch_{epoch}.pth')

            print(f"Epoch ID: {epoch}, 'the best training loss': {bestLoss}")
            model.evaluate(model, validate_data, device, (stateHidden, stateCurrent), args)
        return bestState
    
    # We are ready to train our RNN model!
    bestState = train(train_data, validate_data, model, args)

    def test(test_data, model, args, bestState):
        model.eval()
        loss_fn = MyLoss(args)
        bestLoss = float('inf')
        for batch, (X, y) in enumerate(test_data):
            y_pred, *throw = model(X, bestState)
            loss = loss_fn(y_pred, y)
            if loss < bestLoss:
                bestLoss = loss
        print(f'Best loss value per batch across test dataset is {bestLoss}')

    test(test_data, model, args, bestState)

    element = torch.Tensor(next(iter(test_data))[0][2])
    element = element[None, :]
    print('input:', next(iter(test_data))[0][2])
    print('ground truth:', next(iter(test_data))[1][2])
    print('prediction:', model.predict(element, bestState))

    element = torch.Tensor(next(iter(test_data))[0][3])
    element = element[None, :]
    print('input:', next(iter(test_data))[0][3])
    print('ground truth:', next(iter(test_data))[1][3])
    print('prediction:', model.predict(element, bestState))

    print('----------------------vvvvv')

    element = torch.Tensor(next(iter(test_data))[0][7])
    print(element.size())
    element = element[None, :]
    print('input:', next(iter(test_data))[0][7])
    print('ground truth:', next(iter(test_data))[1][7])
    print('prediction:', model.predict(element, bestState))

    # element = torch.Tensor(next(iter(test_data))[0][6])
    # element = element[None, :]
    # print('input:', next(iter(test_data))[0][6])
    # print('ground truth:', next(iter(test_data))[1][6])
    # print('prediction:', model.predict(element, bestState))

    print('--------------------------')

    print('bestStateHidden:', bestState[0])
    print('###############')
    print('bestStateCurrent:', bestState[1].size())
    print('bestStateCurrent:', bestState[1])
    #print(bestState.size())

    fab_dat = torch.Tensor([[i*(i+1), i*(i+3),i+3,i+4, i*i, (i/2)*(i+10)]for i in range(args.seqLength)])
    print(fab_dat.size())
    fab_dat = fab_dat[None, :]
    print('input:', fab_dat)
    #print('ground truth:', next(iter(test_data))[1][7])
    print('prediction:', model.predict(fab_dat, bestState))