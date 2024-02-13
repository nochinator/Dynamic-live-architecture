import concurrent
import os
from concurrent import futures
import pickle
import neurons as n
from typing import List


class NeuralNetwork:
    def __init__(self, input_neurons: List[n.Neuron], hidden_neurons: List[n.Neuron],
                 output_neurons: List[n.Neuron], cores=None):
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

        # Prime neurons
        for i, neuron in enumerate(self.input_neurons):
            neuron.prime(inputs[i])
        for neuron in self.hidden_neurons + self.output_neurons:
            neuron.prime()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores) as executor:
            # Map the fire_neuron function to the neurons in parallel
            executor.map(n.Neuron.fire, self.network)
        for neuron in self.output_neurons:
            # Fire output neuron
            outputs.append(neuron.output)
        return outputs

    def reinforce(self, reward: List[float], backpropagations: int, cycles: int) -> None:
        """
        Train the network based on expected input and output
        :param reward: array with rewards for each output separately, values between -1 and 1
        :param backpropagations: How many neurons to backpropogate through, higher values result in better fine-tuning
        but an exponential increase in compute required. Low values on large networks will result in some neurons
        never training
        :return: None
        """
        # train each output neuron with the parameters
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores) as executor:
            executor.map(n.Neuron.train, self.output_neurons, reward, [backpropagations]*len(self.output_neurons), [cycles]*len(self.output_neurons))

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
