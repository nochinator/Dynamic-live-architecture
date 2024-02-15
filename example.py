import time
import network_manager
import neurons

# !!!! IMPORTANT !!!
# This file will be completely re-made after re-implementation
# the XOR problem is not a suitable problem for DLA

# XOR data
X_train = [[0, 0], [0, 1], [1, 1], [1, 0]]
y_train = [[0], [1], [0], [1]]
learning_rate = 0.1

# Create neurons
input_neurons = [neurons.InputNeuron((0, 0)) for _ in range(1)]
hidden_neurons = [neurons.HiddenNeuron(1, (0, 1), learning_rate) for _ in range(999)]
output_neurons = [neurons.AnchorNeuron(1, (0, 2), learning_rate)]
network = input_neurons + hidden_neurons + output_neurons

# Initialize connections
hidden_neurons[0].initialize_neuron(input_neurons)
hidden_neurons[1].initialize_neuron(input_neurons)

output_neurons[0].initialize_neuron(hidden_neurons)

# create the network manager for easier usage of neurons
neural_network = network_manager.NeuralNetwork(
    input_neurons=input_neurons,
    hidden_neurons=hidden_neurons,
    output_neurons=output_neurons,
)

# Function to make predictions
def make_predictions(X, duration):
    start_time = time.time()
    end_time = start_time + duration
    predictions_made = 0

    while time.time() < end_time:
        neural_network.propagate_input(X)
        neural_network.train()
        predictions_made += 1

    return predictions_made


# Perform predictions for 10 seconds
duration = 10
total_predictions = make_predictions(X_train[1], duration)

print(total_predictions)
print(total_predictions / 10)
