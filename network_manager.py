import concurrent
import multiprocessing
import os
import threading
from concurrent import futures
import pickle

import numpy as np

import neurons as n
from typing import List


# global functions for efficient multi-threading/multi-coring
def prime_neuron(neuron):
    neuron.prime()


def fire_neuron(neuron):
    neuron.fire()


def train_neuron(neuron, cycle, context_size):
    neuron.train(cycle, context_size)


class NeuralNetwork:
    def __init__(self, input_neurons: List[n.InputNeuron], hidden_neurons: List[n.HiddenNeuron],
                 output_neurons: List[n.AnchorNeuron] or List[n.HiddenNeuron], cores=os.cpu_count()):
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
        self.internal_neurons = hidden_neurons + output_neurons
        self.network = input_neurons + hidden_neurons + output_neurons
        self.cores = cores

    def propagate_input(self, inputs: List[np.float32]) -> List[float]:
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: array of shape *number of input neurons*
        :return: network outputs
        """
        outputs = []

        # give input, firing inputs before priming others reduces reaction speed by one cycle with no downside
        for i, neuron in enumerate(self.input_neurons):
            neuron.fire(np.float32(inputs[i]))
        print("input ready")

        # Iterate over neurons and create a thread for each
        for neuron in self.internal_neurons:
            prime_neuron(neuron)

        print("primed")

        # Empty list to hold threads from firing
        threads = []

        # Iterate over neurons and create a thread for each
        for neuron in self.internal_neurons:
            thread = threading.Thread(target=fire_neuron, args=(neuron,))
            thread.start()
            thread.join()
            threads.append(thread)
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        print("fired")

        # collect outputs
        for neuron in self.output_neurons:
            outputs.append(neuron.output)
        print("outputing")
        return outputs


    def train(self, cycle, context_size):
        """
        Train the network with hebbian learning
        :param cycle: how many cycles back in memory to consider the start of context, 0 is the most recent cycle
        :param context_size: how many neurons back from the start of context to consider in context
        :return: None
        """
        # train each output neuron with the parameters
        with multiprocessing.Pool() as pool:
            # Map the train function to each neuron
            pool.starmap(train_neuron, [(neuron, cycle, context_size) for neuron in self.internal_neurons])



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
