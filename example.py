import network_manager
import neuron
import numpy as np


new_neuron = neuron.Neuron

# XOR data
X_train = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
y_train = np.array([[0], [1], [0], [1]])
learning_rate = 0.1

# Create neurons
input_neurons = [new_neuron(memory_slots=5, time_sensitive_neuron=False, is_input_neuron=True) for _ in range(2)]
hidden_neurons = [new_neuron(memory_slots=5, time_sensitive_neuron=False, learning_rate=learning_rate) for _ in range(2)]
output_neurons = [new_neuron(memory_slots=5, time_sensitive_neuron=False, learning_rate=learning_rate)]

# Initialize connections
hidden_neurons[0].initialize_connections(input_neurons)
hidden_neurons[1].initialize_connections(input_neurons)

output_neurons[0].initialize_connections(hidden_neurons)

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
    print(f"\nEpoch {epoch}: Predictions - {[round(prediction[0], 3) for prediction in output]}, Expected - {y_train.flatten()}", end=' ')
print("done")
