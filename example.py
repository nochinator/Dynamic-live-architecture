import network_manager
import neuron
import numpy as np


def activation(x):
    return x ** 2 - 1


def weight_initialize(x):
    return np.random.uniform(0.4, 0.6, x)


new_neuron = neuron.Neuron
# XOR data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Create a neural network for XOR problem
input_neurons = np.array(
    [new_neuron(1, activation=activation, weight_initialization=weight_initialize, memory_slots=3,
                is_input_neuron=True) for _ in range(2)])

hidden_layers = np.array([
    [new_neuron(2, activation=activation, weight_initialization=weight_initialize, memory_slots=3)
     for _ in range(2)]])

output_neurons = np.array(
    [new_neuron(2, activation=activation, weight_initialization=weight_initialize, memory_slots=3)])

# Initialize connections
for neuron in hidden_layers[0]:
    neuron.initialize_connections(input_neurons)
output_neurons[0].initialize_connections(hidden_layers[0])

neural_network = network_manager.NeuralNetwork(
    input_neurons=input_neurons,
    hidden_layers=hidden_layers,
    output_neurons=output_neurons,
    learning_rate=0.01,
)

# Train the neural network on XOR data
epochs = 10000
for epoch in range(epochs):
    output = []
    for i in range(len(X_train)):
        result = (neural_network.propagate_input(X_train))
        reward = (result[0] - 0.5) * 2
        neural_network.reinforce(reward, 5)
        output.append(result)
    if epoch % 1000 == 0:
        print(
            f"\nEpoch {epoch}: Predictions - {[round(prediction[0], 3) for prediction in output]}, Expected - {y_train.flatten()}",
            end=' ')
