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

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    #set variables
    T_options = list(range(5, 36, 2))
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


    all_accuracies = []
    all_losses = []
    all_train_steps = []


    for T in T_options:

        accuracies = np.array([])
        losses = np.array([])

        final_accs = []
        final_losses = []
        final_train_steps = []

        config.input_length = T

        for i in range(6):

            print("Iteration", i, "with T:", T,)

            # Initialize the dataset and data loader (note the +1)
            dataset = PalindromeDataset(config.input_length + 1)
            data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

            # Initialize the model that we are going to use
            if config.model_type == "RNN":
                model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, device)
            elif config.model_type == "LSTM":
                model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, device)
            model.to(device)

            # Setup the loss and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

            for step, (batch_inputs, batch_targets) in enumerate(data_loader):

                # Only for time measurement of step through network
                t1 = time.time()

                # Add more code here ...
                if config.input_dim != 1:
                    batch_inputs = one_hot(batch_inputs, config.input_dim)

                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                p = model.forward(batch_inputs)

                loss = criterion(p, batch_targets)
                accuracy = acc(p, batch_targets)

                optimizer.zero_grad()

                loss.backward()

                ############################################################################
                # QUESTION: what happens here and why?
                ############################################################################
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
                ############################################################################

                optimizer.step()

                # Just for time measurement
                t2 = time.time()
                examples_per_second = config.batch_size/float(t2-t1)

                if step % 10 == 0:
                    print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                          "Accuracy = {:.2f}, Loss = {:.3f}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                            config.train_steps, config.batch_size, examples_per_second,
                            accuracy, loss
                    ))

                    accuracies = np.append(accuracies, accuracy)
                    losses = np.append(losses, loss.item())

                if step == config.train_steps or (losses[-50:].std() < 0.01 and step>500):
                    # If you receive a PyTorch data-loader error, check this bug report:
                    # https://github.com/pytorch/pytorch/pull/9655

                    final_accs.append(accuracy)
                    final_losses.append(loss.item())
                    final_train_steps.append(step)

                    break

        all_accuracies.append(final_accs)
        all_losses.append(final_losses)
        all_train_steps.append(final_train_steps)

    print('Done training.')
    print(all_accuracies)
    print(all_losses)
    print(all_train_steps)

    with open("file.txt", "w") as output:
        output.write("accuracies \n")
        output.write(str(all_accuracies) + "\n")
        output.write("losses \n")
        output.write(str(all_losses) + "\n")
        output.write("train steps \n")
        output.write(str(all_train_steps) + "\n")

    #plt.figure(figsize=(19, 6))

    # mean_acc = list(np.array(all_accuracies).mean(axis=1))
    # maxstd_acc = list(np.array(all_accuracies).mean(axis=1) + np.array(all_accuracies).std(axis=1))
    # minstd_acc = list(np.array(all_accuracies).mean(axis=1) - np.array(all_accuracies).std(axis=1))
    #
    # mean_loss = list(np.array(all_losses).mean(axis=1))
    # maxstd_loss = list(np.array(all_losses).mean(axis=1) + np.array(all_losses).std(axis=1))
    # minstd_loss = list(np.array(all_losses).mean(axis=1) - np.array(all_losses).std(axis=1))

    mean_train_steps = list(np.array(all_train_steps).mean(axis=1))

    # plt.subplot(1, 3, 1)
    # plt.plot(T_options, all_accuracies, "o", color="orange")
    # plt.plot(T_options, mean_acc, label="accuracy mean", color="orange")
    # plt.fill_between(T_options, maxstd_acc, minstd_acc, alpha = 0.2, label="accuracy std", color="orange")
    # plt.xlabel("input_length")
    # plt.ylabel("accuracy")
    # plt.legend()
    #
    # plt.subplot(1, 3, 2)
    # plt.plot(T_options, all_losses, "o", color="orange")
    # plt.plot(T_options, mean_loss, label="loss mean", color="orange")
    # plt.fill_between(T_options, maxstd_loss, minstd_loss, alpha = 0.2, label="loss std", color="orange")
    # plt.xlabel("input_length")
    # plt.ylabel("loss")
    # plt.legend()
    #
    # plt.subplot(1, 3, 3)
    # plt.plot(T_options, mean_train_steps, label="average train steps", color="orange")
    # plt.xlabel("input_length")
    # plt.ylabel("train steps")
    # plt.legend()
    #
    # plt.savefig("ltsm2.png")
    #
    # plt.show()

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
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)