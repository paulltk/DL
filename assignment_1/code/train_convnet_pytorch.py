"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
cuda = torch.device('cuda')

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  accuracy = 0
  for prediction, target in zip(predictions, targets):
    if prediction.argmax() == target.argmax():
      accuracy += 1
  accuracy /= predictions.shape[0]

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """
  # set cuda
  cuda = False

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  # import the test data
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  loss_function = torch.nn.CrossEntropyLoss()
  neuralnet = ConvNet(3, 10)
  if cuda:
    neuralnet.cuda() # run on GPU
  sgd_back = torch.optim.Adam(neuralnet.parameters(), lr=FLAGS.learning_rate)

  #lists with losses and accuracies
  train_losses = []
  train_accs = []
  test_losses = []
  test_accs = []
  graph_x = []

  # for i in range(50):
  for i in range(FLAGS.max_steps):
    neuralnet.train()
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    if cuda:
      x = torch.from_numpy(x).cuda()
      y = torch.from_numpy(y).cuda()
    else:
      x = torch.from_numpy(x)
      y = torch.from_numpy(y)

    # predict on the train data and calculate the gradients
    out = neuralnet.forward(x) #output after the softmax

    train_loss = loss_function(out, y.argmax(dim=1))

    sgd_back.zero_grad()
    train_loss.backward()
    sgd_back.step()

    # save the losses for every eval_freqth loop
    if i % FLAGS.eval_freq == 0 or i == (FLAGS.max_steps - 1):
    # if i % 1 == 0 or i == (FLAGS.max_steps - 1):
      neuralnet.eval()
      with torch.no_grad():
        test_x, test_y = cifar10['test'].next_batch(1000)
        if cuda:
          torch.cuda.empty_cache()
          test_x = torch.from_numpy(test_x).cuda()
          test_y = torch.from_numpy(test_y).cuda()
        else:
          test_x = torch.from_numpy(test_x)
          test_y = torch.from_numpy(test_y)

        train_out = neuralnet.forward(x)  # output after the softmax
        test_out = neuralnet.forward(test_x)
        if cuda:
          torch.cuda.empty_cache()

        train_loss = loss_function(train_out, y.argmax(dim=1))
        test_loss = loss_function(test_out, test_y.argmax(dim=1))
        train_acc = accuracy(out, y)
        test_acc = accuracy(test_out, test_y)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        graph_x.append(i)

        print("iteration:", i)
        print("Test accuracy:", test_accs[-1])
        print("Test loss:", test_losses[-1])

  plt.figure()
  plt.subplot(1, 2, 1)
  plt.plot(graph_x, train_losses, label="train loss")
  plt.plot(graph_x, test_losses, label="test loss")
  plt.title('Losses')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(graph_x, train_accs, label="train acc")
  plt.plot(graph_x, test_accs, label="test acc")
  plt.title('Accuracies')
  plt.legend()

  print("Final test accuracy:", test_accs[-1])
  print("Final test loss:", test_losses[-1])

  plt.savefig("conv_acc_and_loss.png")

  with open('conv_acc_and_loss.txt', 'w') as f:
    f.write("test_losses \n")
    f.write(str(test_losses) + "\n \n")
    f.write("test accs \n")
    f.write(str(test_accs))


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()