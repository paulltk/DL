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


import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    #set variables
    config.model_type = "LSTM"

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    def one_hot(x, input_dim):
        one_hot_vec = (torch.arange(x.max() + 1) == x[..., None]).float()
        return one_hot_vec

    def acc(predictions, targets):
        accuracy = 0
        for prediction, target in zip(predictions, targets):
            if prediction.argmax() == target:
                accuracy += 1
        accuracy /= predictions.shape[0]
        return accuracy

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes)
    elif config.model_type == "LSTM":
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        if config.input_dim != 1:
            batch_inputs = one_hot(batch_inputs, config.input_dim)

        p = model.forward(batch_inputs)

        loss = criterion(p, batch_targets)
        accuracy = acc(p, batch_targets)

        optimizer.zero_grad()

        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        gradient_magnitudes = []
        inputs = list(range(1, config.input_dim +1))

        for hi in model.all_h:
            print(hi)


        break

    print('Done training.')
    #
    # plt.figure(figsize=(15, 8))
    #
    # mean_acc = list(np.array(all_accuracies).mean(axis=1))
    # maxstd_acc = list(np.array(all_accuracies).mean(axis=1) + np.array(all_accuracies).std(axis=1))
    # minstd_acc = list(np.array(all_accuracies).mean(axis=1) - np.array(all_accuracies).std(axis=1))
    #
    # mean_loss = list(np.array(all_losses).mean(axis=1))
    # maxstd_loss = list(np.array(all_losses).mean(axis=1) + np.array(all_losses).std(axis=1))
    # minstd_loss = list(np.array(all_losses).mean(axis=1) - np.array(all_losses).std(axis=1))
    #
    # plt.plot(T_options, all_accuracies, "o", color="black")
    # plt.plot(T_options, all_losses, "o", color="black")
    #
    # plt.plot(T_options, mean_acc, label="accuracy mean")
    # plt.fill_between(T_options, maxstd_acc, minstd_acc, alpha = 0.2, label="accuracy std")
    # plt.plot(T_options, mean_loss, label="loss mean")
    # plt.fill_between(T_options, maxstd_loss, minstd_loss, alpha = 0.2, label="loss std")
    #
    # plt.title('Accuracy and loss over different T\'s with %i seeds for every T' %(i+1))
    # plt.legend()
    # plt.xlabel("input_length")
    # plt.show()
    # plt.savefig("jaja.png")

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
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