import pickle
import numpy as np


class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers, output_neurons, learning_rate):
        """
        create a neural network using the Neuron class.
        :param input_neurons: expected format: np.array[InputNeuron(), InputNeuron(), etc.]
        :param hidden_layers: expected format: np.array[[Neuron(), Neuron()], [Neuron(), Neuron()], etc.]
        :param output_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        :param learning_rate: how much to update the weights when training or reinforcing
        """
        self.input_neurons = input_neurons
        self.hidden_layers = hidden_layers
        self.output_neurons = output_neurons
        concatenated_hidden_layers = np.concatenate([hidden_layer.ravel() for hidden_layer in hidden_layers])
        self.active_neurons = np.concatenate((concatenated_hidden_layers, output_neurons))
        self.network = np.concatenate((input_neurons, self.active_neurons))

        self.learning_rate = learning_rate
        self.neuron_importance = []
        self.outputs = []

    def propagate_input(self, inputs):
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: array of shape *number of input neurons*
        :return: network outputs
        """
        outputs = []

        # Prime neurons
        print("\ninputs")

        for i, neuron in enumerate(self.input_neurons):
            neuron.prime(inputs[i])
        # Fire input neurons
        for neuron in self.input_neurons:
            neuron.fire()

        # Hidden neurons layer by layer
        for layer in self.hidden_layers:
            print("\nhidden")

            # Prime neurons in the layer
            for neuron in layer:
                neuron.prime()
            # Fire neurons in the layer
            for neuron in layer:
                neuron.fire()

        # Output neurons
        print("\noutputs")

        for neuron in self.output_neurons:
            # Prime output neuron
            neuron.prime()
        for neuron in self.output_neurons:
            # Fire output neuron
            neuron.fire()
            outputs.append(neuron.output)
        self.outputs = outputs
        return outputs

    def reinforce(self, reward, backpropogations, reference_output=None):
        """
        Train the network based on expected input and output
        :param reward: array of shape *number of output neurons, rewards each output separately, values between -1 and 1
        :param backpropogations: How many neurons to backpropogate through, higher values result in better fine-tuning
        but an exponential increase in compute required. Low values on large networks will result in some neurons
        never training
        :param reference_output: Used for training in hindsight. Leaving blank will reward the last predicted output
        :return None
        """
        # defaults
        if reference_output is None:
            reference_output = self.outputs

        # train each output neuron with the parameters
        for i, neuron in enumerate(self.output_neurons):
            neuron.train(reward[i], backpropogations, reference_output[i])

    def save_model(self, file_path):
        """
        Save the created neural network as a pkl. resulting file contains all data needed to reconstruct the network.
        :param file_path: path to save file at
        :return: None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_path):
        """
        Load a pkl file that contains a neural network. File must contain all data needed to reconstruct the network.
        :param file_path: path to load file from
        :return: None
        """
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model