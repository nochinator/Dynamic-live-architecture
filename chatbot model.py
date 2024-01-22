import neuron
import network_manager
import numpy


new_neuron = neuron.Neuron
learning_rate = 0.1  # create dynamic adjustment system

# Create network
input_neurons = [new_neuron(memory_slots=3, is_input_neuron=True) for _ in range(100)]

hidden_layers = [[new_neuron(memory_slots=3, learning_rate=learning_rate) for _ in range(500)] for _ in range(3)]

output_neurons = [new_neuron(memory_slots=3, learning_rate=learning_rate)]

# Initialize connections
front_neurons = [input_neurons[0], (neuron for layer in hidden_layers for neuron in layer)]
for layer in range(len(hidden_layers)):
    for neuron in hidden_layers[layer]:
        neuron.initialize_connections(front_neurons)

output_neurons[0].initialize_connections((neuron for layer in hidden_layers for neuron in layer))

# create network
nn = network_manager.NeuralNetwork(input_neurons, hidden_layers, output_neurons)


