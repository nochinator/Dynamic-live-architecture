import concurrent
import os
from concurrent import futures
import pickle
import neurons as n
from typing import List


class NeuralNetwork:
    def __init__(self, input_neurons: List[n.InputNeuron], hidden_neurons: List[n.HiddenNeuron],
                 output_neurons: List[n.AnchorNeuron] or List[n.HiddenNeuron], cores=None):
        """
        create a neural network using the Neuron class.
        :param input_neurons: expected format: np.array[InputNeuron(), InputNeuron(), etc.]
        :param hidden_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        :param output_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        :param cores:  number of virtual cpu cores to use, if unset then uses all cores, can be set when loading model
        """
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.network = input_neurons + hidden_neurons + output_neurons
        if cores is None:
            self.cores = os.cpu_count()
        else:
            self.cores = cores

    def propagate_input(self, inputs: List[float]) -> List[float]:
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: array of shape *number of input neurons*
        :return: network outputs
        """
        outputs = []

        # give input, firing inputs before priming reduces reaction speed by one cycle with no downside
        for i, neuron in enumerate(self.input_neurons):
            neuron.fire(inputs[i])

        # Prime neurons
        for neuron in self.hidden_neurons + self.output_neurons:
            neuron.prime()

        # fire neurons
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores) as executor:
            # Map the fire_neuron function to the neurons in parallel
            executor.map(n.HiddenNeuron.fire, self.network)
            executor.map(n.AnchorNeuron.fire, self.network)


        # collect output
        for neuron in self.output_neurons:
            outputs.append(neuron.output)
        return outputs

    def train(self):
        """
        Train the network with hebbian learning
        :return: None
        """
        # train each output neuron with the parameters
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores) as executor:
            executor.map(n.HiddenNeuron.train, (self.hidden_neurons + self.output_neurons))
            executor.map(n.AnchorNeuron.train, (self.hidden_neurons + self.output_neurons))

    def save_model(self, file_path):
        """
        Save the created neural network as a pkl. resulting file contains all data needed to reconstruct the network.
        :param file_path: path to save model to
        :return: None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_path, cores=None):
        """
        Load a pkl file that contains a neural network. File must contain all data needed to reconstruct the network.
        :param file_path: path to load model from
        :param cores: the number of virtual cpu cores to use, if unset then uses all cores
        :return: None
        """
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)

        if cores is None:
            cls.cores = os.cpu_count()
        else:
            cls.cores = cores
        return loaded_model
