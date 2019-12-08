################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import sys
sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def acc(predictions, targets):
        accuracy = (predictions.argmax(dim=1) == targets).float().mean().item()
        return accuracy

    # Initialize the dataset and data loader (note the +1
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lstm = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes)
    rnn = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, device)

    optimizer_lstm = torch.optim.RMSprop(lstm.parameters(), lr=config.learning_rate)
    optimizer_rnn = torch.optim.RMSprop(rnn.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        print("step",step)
        # Initialize the model that we are going to use
        lstm_out = lstm.forward(batch_inputs)

        optimizer_lstm.zero_grad()
        loss_lstm = criterion(lstm_out, batch_targets)
        loss_lstm.backward()
        optimizer_lstm.step()


        rnn_out = rnn.forward(batch_inputs)

        optimizer_rnn.zero_grad()
        loss_rnn = criterion(rnn_out, batch_targets)
        loss_rnn.backward()
        optimizer_rnn.step()

        lstm_norms = []
        for h in lstm.all_h:
            lstm_norms.append(h.grad.norm().item())

        rnn_norms = []
        for h in rnn.all_h:
            rnn_norms.append(h.grad.norm().item())

        sequence = list(range(1, config.input_length + 1))
        plt.figure(figsize=(15, 6))
        plt.plot(sequence, rnn_norms, label="rnn")
        plt.plot(sequence, lstm_norms, label="lstm")
        plt.legend()
        plt.xlabel("sequence value")
        plt.ylabel("gradient norm")

        plt.show()

        break

    print('Done training.')

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=100, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=200, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)