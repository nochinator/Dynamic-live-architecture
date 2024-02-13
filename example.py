import network_manager
import neurons
import numpy as np


# XOR data
X_train = [[0, 0], [0, 1], [1, 1], [1, 0]]
y_train = [[0], [1], [0], [1]]
learning_rate = 0.1

# Create neurons
input_neurons = [neurons.InputNeuron((0, 0)) for _ in range(2)]
hidden_neurons = [neurons.HiddenNeuron((0, 1), learning_rate=learning_rate) for _ in range(2)]
output_neurons = [neurons.AnchorNeuron((0, 2), learning_rate=learning_rate)]

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

# Train the neural network on XOR data
epochs = 1000
for epoch in range(epochs):
    output = []
    for i in range(len(X_train)):
        # predict with the network
        result = neural_network.propagate_input(X_train[i])

        # train the network
        if result == y_train[i]:
            reward = [True]
        else:
            reward = [False]
        neural_network.reinforce(reward, 5, 0)
        output.append(result)
    print(f"\nEpoch {epoch}: Predictions - {[round(prediction[0], 3) for prediction in output]}, Expected - "
          f"{y_train.flatten()}", end=' ')
print("done")
