# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import random
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

################################################################################

def train(config):
    
    def acc(predictions, targets):
        accuracy = (predictions.argmax(dim=2) == targets).float().mean()
        return accuracy

    # Initialize the device which to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device", device)
    print("book:", config.txt_file)
    
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset._vocab_size,
                 config.lstm_num_hidden, config.lstm_num_layers, device).to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    gen_lengths = [20, 30, 100, 200]
    print("temperature:", config.temperature_int)

    all_accuracies = []
    all_losses = []
    all_train_steps = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = (torch.arange(dataset._vocab_size) == batch_inputs[..., None])  # create one-hot

        # Only for time measurement of step through network
        t1 = time.time()

        # set the data to device
        batch_inputs = batch_inputs.float().to(device)
        batch_targets = batch_targets.to(device)

        out, _ = model.forward(batch_inputs)  # forward pass

        loss = criterion(out.permute(0, 2, 1), batch_targets)  # calculate the loss
        accuracy = acc(out, batch_targets)  # calculate the accuracy

        optimizer.zero_grad() # throw away previous grads

        loss.backward() # calculate new gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)  # make sure the gradients do not explode

        optimizer.step() # update the weights

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            all_accuracies.append(accuracy.item())
            all_losses.append(loss.item())
            all_train_steps.append(step)


        if step % config.sample_every == 0:

            for gen_length in gen_lengths:
                print("Generated sentence with length of {}".format(gen_length))

                previous = random.randint(0, dataset._vocab_size - 1)  # get the first random letter
                letters = [previous]
                cell = None

                for i in range(gen_length):
                    # create input
                    input = torch.zeros(1, 1, dataset._vocab_size).to(device)
                    input[0, 0, previous] = 1

                    # do a forward pass
                    out, cell = model.forward(input, cell)

                    # get the next letter
                    out = out.squeeze()
                    if config.temperature is True:
                        out *= config.temperature_int
                        out = torch.softmax(out, dim=0)
                        previous = torch.multinomial(out, 1)[0].item()

                    else:
                        previous = out.argmax().item()

                    letters.append(previous)

                # convert to sentence
                sentence = dataset.convert_to_string(letters)
                print(sentence)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    with open("acc_loss_T_{}.txt".format(config.temperature_int), "w") as output:
        output.write("accuracies \n")
        output.write(str(all_accuracies) + "\n")
        output.write("losses \n")
        output.write(str(all_losses) + "\n")
        output.write("train steps \n")
        output.write(str(all_train_steps) + "\n")


    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--txt_file', type=str, default="assets/book_EN_democracy_in_the_US.txt", help="Path to a .txt file to train on")
    parser.add_argument('--txt_file', type=str, default="assets/book_EN_grimms_fairy_tails.txt", help="Path to a .txt file to train on")
    # parser.add_argument('--txt_file', type=str, default="assets/book_NL_darwin_reis_om_de_wereld.txt", help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')
    parser.add_argument('--temperature_int', type=float, default=1, help='Temperature integer')
    parser.add_argument('--temperature', type=bool, default=True, help='Use temperature?')

    config = parser.parse_args()

    # Train the model
    train(config)