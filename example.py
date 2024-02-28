import time
import DLA_Library
import cProfile

# !!!! IMPORTANT !!!
# This file will be completely re-made after re-implementation
# the XOR problem is not a suitable problem for DLA

# XOR data
X_train = [[0, 0], [0, 1], [1, 1], [1, 0]]
y_train = [[0], [1], [0], [1]]
learning_rate = 0.1

# Create neurons
input_neurons = [DLA_Library.InputNeuron(i, (0, 0)) for i in range(10000)]
hidden_neurons = [DLA_Library.ActiveNeuron(i + 10000, (0, 1), learning_rate) for i in range(10000)]
output_neurons = [DLA_Library.ActiveNeuron(i + 20000, (0, 2), learning_rate) for i in range(25)]
network = input_neurons + hidden_neurons + output_neurons

# Initialize connections
for neuron in hidden_neurons + output_neurons:
    neuron.initialize_neuron(network)

# create the network manager for easier usage of neurons
neural_network = DLA_Library.NeuralNetwork(input_neurons, hidden_neurons, output_neurons, 50)


# Function to make predictions
def make_predictions(X, duration):
    neural_network.propagate_input(X)
    #neural_network.train(0, 50, 1)
    start_time = time.time()
    end_time = start_time + duration
    predictions_made = 0

    while time.time() < end_time:
        neural_network.propagate_input(X)
        neural_network.train(0, 50, 1)
        predictions_made += 1

    print(predictions_made)
    print(predictions_made / 10)


# Perform predictions for 10 seconds
duration = 10
make_predictions(X_train[1], duration)
#cProfile.run('make_predictions(X_train[1], duration)')


