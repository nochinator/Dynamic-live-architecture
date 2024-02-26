import numpy as np
import os
import pickle
from typing import List

"""
!!!IMPORTANT NOTE!!!
nearly all constants will be made a hyperparameter for beta release and full releases
unless explicitly stated or the value is 0/None or in most cases 1
"""


class InputNeuron:
    def __init__(self, position: tuple[float, float]):
        """
        Create an input neuron, will only provide input to rest of network, does not do any processing on the data.
        :param position: The initial position of the input neuron; will not move from this position.
        Note: Neuron number is not saved, but should still be counted. Start at 0 (0, 1, 2, 3...) like list indexes.
        """
        self.position = position
        self.output = np.float32(0)

    def fire(self, output: np.float32):
        """
        Does not do any processing, simply sets its output to the input.
        :param output: The output this neuron should provide.
        :return: None; get the output from neuron.output.
        """
        self.output = output


class ActiveNeuron:
    def __init__(self, neuron_number: int, position: tuple[float, float], learning_rate:
    np.float32, mobile_neuron: bool):
        """
        Create a hidden neuron.
        :param neuron_number: Unique identifier for this neuron, neurons must be numbered in order of creation.
        :param position: The starting position of the hidden neuron.
        :param learning_rate: How quickly to change the weights when training.
        :param mobile_neuron: Weather or not the neurons should move. Stationary neurons are often used for output.
        """
        # Save values of initialization
        self.number = neuron_number
        self.learning_rate = learning_rate
        self.position = position
        self.mobility = mobile_neuron

        # Prep variables for use
        self.output = np.float32(0)
        self.memory = None
        self.memory_index = 0
        self.network = None
        self.synaptic_weights = None

    def get_distances(self):
        # calculates the distance between every other neuron (including self) and the
        return np.linalg.norm(np.array(self.position) - np.array([neuron.position for neuron in self.network]), axis=1)

    def initialize_neuron(self, network):
        """
        Connect this neuron to neighboring neurons using random weights.
        :param network: List of every neuron in the network.
        :return: None
        """
        self.network = network

        # Calculate distances between neurons
        distances = self.get_distances()

        # Initialize synaptic weights with random values
        self.synaptic_weights = np.zeros(len(self.network))
        self.synaptic_weights = np.where(distances <= 1, np.random.uniform(0, 1), 0)

        # set weight for self to 0
        self.synaptic_weights[self.number] = 0

        # Normalize weights to sum up to 1
        total_weight = np.sum(self.synaptic_weights)
        if total_weight != 0:
            self.synaptic_weights /= total_weight

    def fire(self, state):
        """
        Make a prediction.

        Training after every fire is recommended.
        :param state: List of output from every neuron in the network IN NUMERICAL ORDER.
        :return: None; get the output from neuron.output
        """
        # calculate output
        self.output = np.dot(state, self.synaptic_weights)

    def train(self, context: List[float], reward: float):
        """
        Use hebbian learning to train the weights in the network and move the neurons around.
        :param context: An array of outputs from every neuron, averaged
        :param reward: The reward for the actions, acts as a simple scalar for weight changes. Positive or Negative
        :return: None
        note: context_size + cycle may NOT be more than (but can be equal to) the number of memory slots
        """
        distances = self.get_distances()

        # Update synaptic weights
        for i, weight in enumerate(self.synaptic_weights):
            if distances[i] < 1 or weight > 0.01:  # 1 is hyperparameter
                # Apply Hebbian-like learning rule to synaptic weights
                weight += self.learning_rate * context[i] * context[self.number] * reward

                # Ensure non-negative weights
                self.synaptic_weights[i] = max(np.float32(0), weight)

        # set weight for self to 0
        self.synaptic_weights[self.number] = 0

        # Normalize weights to add up to 1
        total_weight = sum(self.synaptic_weights)
        self.synaptic_weights /= total_weight

        # move neurons
        if self.mobility:
            for i, neuron in enumerate(self.network):
                # Calculate vectors
                vector1 = np.array(neuron.position) - np.array(self.position)
                vector2 = np.array(self.position) - np.array(neuron.position)

                # Normalize and scale vectors
                norm_vector1 = (vector1 / np.linalg.norm(vector1)) * self.synaptic_weights[i]
                norm_vector2 = (vector2 / np.linalg.norm(vector2)) * self.synaptic_weights[i]

                # Move neurons
                self.position = tuple(np.array(self.position) + norm_vector1)
                neuron.position = tuple(np.array(neuron.position) + norm_vector2)
        else:
            for i, neuron in enumerate(self.network):
                # Calculate vectors
                vector2 = np.array(self.position) - np.array(neuron.position)

                # Normalize and scale vectors
                norm_vector2 = (vector2 / np.linalg.norm(vector2)) * self.synaptic_weights[i]

                # Move neurons
                neuron.position = tuple(np.array(neuron.position) + norm_vector2)


class NeuralNetwork:
    def __init__(self, input_neurons: List[InputNeuron], hidden_neurons: List[ActiveNeuron], output_neurons:
                 List[ActiveNeuron], memory_slots: int):
        """
        create a neural network using the Neuron class.
        :param input_neurons: expected format: np.array[InputNeuron(), InputNeuron(), etc.]
        :param hidden_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        :param output_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        :param memory_slots: How many actions to remember.
        """
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.internal_neurons = hidden_neurons + output_neurons
        self.network = input_neurons + hidden_neurons + output_neurons

        # setup self.input_memory
        self.memory = np.zeros((memory_slots, len(self.network)), dtype=np.float32)
        self.memory_index = 0

    def propagate_input(self, inputs: List[np.float32]) -> List[float]:
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: Array of shape *number of input neurons*.
        :return: Outputs of the network in an array.
        """
        outputs = []

        # firing input neurons before building the state reduces reaction delay by 1 cycle at with disadvantage
        for i, neuron in enumerate(self.input_neurons):
            neuron.fire(inputs[i])

        state = []
        for neuron in self.network:
            state.append(neuron.output)
        for neuron in self.internal_neurons:
            neuron.fire(state)

        # collect outputs
        for neuron in self.output_neurons:
            outputs.append(neuron.output)

        # update memory
        self.memory[self.memory_index] = state
        self.memory_index = (self.memory_index + 1) % len(self.memory)

        return outputs

    def train(self, cycle, context_size, reward):
        """
        Train the network with hebbian learning.
        :param cycle: Number of cycles back in memory to consider the start of context, 0 is the most recent cycle.
        :param context_size: Number of neurons back from specified cycle to consider in context.
        :param reward: The reward to apply to weight changes, acts as a simple scalar for the value.
        :return: None
        note:
        context_size + cycle may NOT be more than (but can be equal to)
        the minimum number of memory slots in the network.
        """
        # Prepare to get context
        start_index = (self.memory_index - cycle) % len(self.memory)

        # Actually get context
        context = np.sum(self.memory[start_index: start_index + context_size: -1], axis=0) / context_size

        # train each output neuron with the parameters
        for neuron in self.internal_neurons:
            neuron.train(context, reward)

    def save_model(self, file_path):
        """
        Save the created neural network as a pkl. The Resulting file contains the network in its current state.
        This includes weights along with the current memory and outputs of every neuron.
        :param file_path: Path to save model to
        :return: None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_path):
        """
        Load a pkl file that contains a neural network.
        The file must have been made with the most recent version of the network.
        As of now, models from older versions are not compatible with newer versions.
        Feel free to help out with this on GitHub.
        :param file_path: Path to load model from.
        :return: The loaded model.
        """
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
