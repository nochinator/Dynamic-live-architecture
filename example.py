import network_manager
import neuron
import numpy as np


new_neuron = neuron.Neuron
# memory XOR data
X_train = np.array([[0], [1], [1], [0]])
y_train = np.array([[0], [1], [0], [1]])
learning_rate = 0.2

# Create a neural network for XOR problem
input_neurons = [new_neuron(memory_slots=3, is_input_neuron=True)]

hidden_layers = [
    [new_neuron(memory_slots=3, learning_rate=learning_rate) for _ in range(3)]]

output_neurons = [new_neuron(memory_slots=3, learning_rate=learning_rate)]

# Initialize connections
hidden_layers[0][0].initialize_connections(input_neurons)
hidden_layers[0][1].initialize_connections([input_neurons[0], hidden_layers[0][0]])
hidden_layers[0][2].initialize_connections([input_neurons[0], hidden_layers[0][0]])

output_neurons[0].initialize_connections([hidden_layers[0][1], hidden_layers[0][2]])

neural_network = network_manager.NeuralNetwork(
    input_neurons=input_neurons,
    hidden_layers=hidden_layers,
    output_neurons=output_neurons,
)

# Train the neural network on XOR data
epochs = 1000
for epoch in range(epochs):
    output = []
    for i in range(len(X_train)):
        result = (neural_network.propagate_input(X_train[i]))
        # print(result)
        reward = [(result[0] - 0.5) * 2]
        neural_network.reinforce(reward, 10)
        output.append(result)
    print(f"\nEpoch {epoch}: Predictions - {[round(prediction[0], 3) for prediction in output]}, Expected - "
          f"{y_train.flatten()}", end=' ')
