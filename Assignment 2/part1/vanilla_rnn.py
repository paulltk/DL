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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.sec_length = seq_length
        self.device = device
        self.input_dim = input_dim
        self.Whx = nn.Parameter(nn.init.xavier_uniform_(torch.empty((input_dim, num_hidden))).to(device))
        self.Whh = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_hidden, num_hidden))).to(device))
        self.Wph = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_hidden, num_classes))).to(device))

        self.Bh = nn.Parameter(torch.zeros(num_hidden).to(device))
        self.Bp = nn.Parameter(torch.zeros(num_classes).to(device))

        self.h = torch.zeros(num_hidden)

    def forward(self, x):

        h = self.h.to(device)
        self.all_h = []

        for i in range(self.sec_length):
            if self.input_dim == 1:
                numbers = x[:, i].view(x.size(0), 1)
            else:
                numbers = x[:, i, :]

            h = (numbers @ self.Whx + h @ self.Whh + self.Bh).tanh().to(self.device)
            self.all_h.append(h.requires_grad_(True))

            p = (h @ self.Wph + self.Bp).to(self.device)

        return p