import pickle
import neuron as n
from typing import List


class NeuralNetwork:
    def __init__(self, input_neurons: List[n.Neuron], hidden_neurons: List[n.Neuron],
                 output_neurons: List[n.Neuron]):
        """
        create a neural network using the Neuron class.
        :param input_neurons: expected format: np.array[InputNeuron(), InputNeuron(), etc.]
        :param hidden_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        :param output_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        """
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.network = input_neurons + hidden_neurons + output_neurons

    def propagate_input(self, inputs: List[float]) -> List[float]:
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: array of shape *number of input neurons*
        :return: network outputs
        """

        outputs = []

        # Prime neurons
        for i, neuron in enumerate(self.input_neurons):
            neuron.prime(inputs[i])
        for neuron in self.hidden_neurons + self.output_neurons:
            neuron.prime()

        # Fire neurons
        for neuron in self.network:
            neuron.fire()

        for neuron in self.output_neurons:
            # Fire output neuron
            outputs.append(neuron.output)

        return outputs

    def reinforce(self, reward: List[float], backpropagations: int) -> None:
        """
        Train the network based on expected input and output
        :param reward: array with rewards for each output separately, values between -1 and 1
        :param backpropagations: How many neurons to backpropogate through, higher values result in better fine-tuning
        but an exponential increase in compute required. Low values on large networks will result in some neurons
        never training
        :return: None
        """
        # train each output neuron with the parameters
        for i, neuron in enumerate(self.output_neurons):
            neuron.train(reward[i], backpropagations)

    def save_model(self, file_path):
        """
        Save the created neural network as a pkl. resulting file contains all data needed to reconstruct the network.
        :param file_path: path to save model to
        :return: None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_path):
        """
        Load a pkl file that contains a neural network. File must contain all data needed to reconstruct the network.
        :param file_path: path to load model from
        :return: None
        """
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
