"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    super(ConvNet, self).__init__()

    channels = [n_channels, 64, 128, 256, 256, 512, 512, 512, 512] # conv layer sizes
    layers_sequence = ["c", "m", "c", "m", "c", "c", "m", "c", "c", "m", "c", "c", "m"] # conv and max pooling sequence
    layers = []

    i = 0
    for l in layers_sequence:
      if l is "c":
        layers.append(nn.Conv2d(channels[i], channels[i+1], stride=1, padding=1,  kernel_size=3))
        layers.append(nn.BatchNorm2d(channels[i+1]))
        layers.append(nn.ReLU())
        i += 1
      elif l is "m":
        layers.append(nn.MaxPool2d(stride=2, padding=1, kernel_size=3))
    layers.append(Flatten())
    layers.append(nn.Linear(channels[-1], n_classes))

    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    out = self.layers(x)

    return out
