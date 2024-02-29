import time

import numpy as np

import DLA_Library
import cProfile

# !!!! IMPORTANT !!!
# This file will be completely re-made after re-implementation
# the XOR problem is not a suitable problem for DLA

#  data
X_train = [[1.0], [2.0], [3.0]]
y_train = [[0.0],  [0.0], [1.0]]
learning_rate = 100

# Create neurons
input_neurons = [DLA_Library.InputNeuron(0, (0, 1)) for i in range(1)]
hidden_neurons = [DLA_Library.ActiveNeuron(i + 1, (0.5, -0.5 + i), learning_rate, False) for i in range(100)]
output_neurons = [DLA_Library.ActiveNeuron(i + 3, (0, 0), learning_rate, True) for i in range(1)]

# create neural network
neural_network = DLA_Library.NeuralNetwork(input_neurons, hidden_neurons, output_neurons, 50)


# Training function
def train_epoch(network, x, y):
    print("input, output, expected, reward")
    total_reward = 0
    for input_value, expected_output in zip(x, y):
        output = network.propagate_input(input_value)
        reward = 1 - np.mean(abs(np.array(output) - np.array(expected_output)))
        network.reinforce(0, 50, reward)
        print(f"{input_value}, {output}, {expected_output}, {reward}")
        total_reward += reward
    return total_reward

# run training function for 500 epoches
for _ in range(500):
    train_epoch(neural_network, X_train, y_train)
