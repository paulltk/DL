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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()

        self.sec_length = seq_length
        self.input_dim = input_dim
        self.device = device

        self.Wgx = nn.Parameter(nn.init.xavier_uniform_(torch.empty((input_dim, num_hidden))).to(self.device))
        self.Wgh = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_hidden, num_hidden))).to(self.device))

        self.Wix = nn.Parameter(nn.init.xavier_uniform_(torch.empty((input_dim, num_hidden))).to(self.device))
        self.Wih = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_hidden, num_hidden))).to(self.device))

        self.Wfx = nn.Parameter(nn.init.xavier_uniform_(torch.empty((input_dim, num_hidden))).to(self.device))
        self.Wfh = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_hidden, num_hidden))).to(self.device))

        self.Wox = nn.Parameter(nn.init.xavier_uniform_(torch.empty((input_dim, num_hidden))).to(self.device))
        self.Woh = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_hidden, num_hidden))).to(self.device))

        self.Bg = nn.Parameter(torch.zeros(num_hidden).to(self.device))
        self.Bi = nn.Parameter(torch.zeros(num_hidden).to(self.device))
        self.Bf = nn.Parameter(torch.zeros(num_hidden).to(self.device))
        self.Bo = nn.Parameter(torch.zeros(num_hidden).to(self.device))

        self.Wph = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_hidden, num_classes))).to(self.device))

        self.Bp = nn.Parameter(torch.zeros(num_classes).to(self.device))

        self.h = torch.zeros(num_hidden).to(self.device)
        self.c = torch.zeros(num_hidden).to(self.device)

    def forward(self, x):

        self.all_h = []
        h = self.h
        c = self.c

        for i in range(self.sec_length):
            if self.input_dim == 1:
                numbers = x[:, i].view(x.size(0), 1)
            else:
                numbers = x[:, i, :]

            g = (numbers @ self.Wgx + h @ self.Wgh + self.Bg).tanh().to(self.device)
            i = (numbers @ self.Wix + h @ self.Wih + self.Bi).sigmoid().to(self.device)
            f = (numbers @ self.Wfx + h @ self.Wfh + self.Bf).sigmoid().to(self.device)
            o = (numbers @ self.Wox + h @ self.Woh + self.Bo).sigmoid().to(self.device)

            c = g * i + c * f

            h = (c * o).tanh()
            self.all_h.append(h.requires_grad_(True))

            p = h @ self.Wph + self.Bp

        return p