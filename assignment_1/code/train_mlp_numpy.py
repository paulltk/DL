"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

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
    accuracy += target[np.argmax(prediction)]
  accuracy /= predictions.shape[0]

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  x, y = cifar10['train'].next_batch(FLAGS.batch_size)
  x = x.reshape(x.shape[0], -1)
  test_x, test_y = cifar10['test'].images, cifar10['test'].labels
  test_x = test_x.reshape(test_x.shape[0], -1)

  loss_function = CrossEntropyModule()
  neuralnet = MLP(x.shape[1], dnn_hidden_units, y.shape[1], neg_slope)

  train_losses = []
  train_accs = []
  test_losses = []
  test_accs = []
  graph_x = []

  for i in range(FLAGS.max_steps):
    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    x = x.reshape(x.shape[0], -1)

    # predict on the train data and calculate the gradients
    out = neuralnet.forward(x) #output after the softmax
    dout = loss_function.backward(out, y)
    neuralnet.backward(dout)
    # update the weights and the biases
    for layer in neuralnet.layers[::2]:
      layer.params['weight'] -= FLAGS.learning_rate * (layer.grads['weight'])
      layer.params['bias'] -= FLAGS.learning_rate * (layer.grads['bias'])

    # save the losses for every eval_freqth loop
    if i % FLAGS.eval_freq == 0 or i == 1499:
      out = neuralnet.forward(x)  # output after the softmax
      test_out = neuralnet.forward(test_x)
      loss = loss_function.forward(out, y)
      test_loss = loss_function.forward(test_out, test_y)
      acc = accuracy(out, y)
      test_acc = accuracy(test_out, test_y)

      train_losses.append(loss)
      train_accs.append(acc)
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
  print(test_accs)
  print(test_losses)

  plt.show()


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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()