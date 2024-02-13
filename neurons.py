import numpy as np
import random


class InputNeuron:
    def __init__(self, position: tuple):
        """
        Create an input neuron, will only provide input to rest of network, does not do any processing on the data
        :param position: the initial position of the input neuron, it will not move from this position
        """
        self.position = position
        self.output = 0

    def fire(self, input=float):
        """
        does not do any processing, simply sets its output to the input
        :return: None, get the output from neuron.output
        """
        self.output = input


class HiddenNeuron:
    def __init__(self, position: tuple, learning_rate=0.1):
        """
        Create a hidden neuron.
        :param position: the starting position of the hidden neuron, the neuron will move around
        :param learning_rate: how quickly to change the weights when training
        """
        # Save values of initialization
        self.learning_rate = learning_rate
        self.position = position

        # Prep variables for use
        self.output = 0
        self.inputs = np.array([], dtype=np.float64)
        self.network = []
        self.neuron_connections = []

        # learnable things:
        self.synaptic_weights = np.array([], dtype=np.float64)
        self.bias = None

    def nearby(self, neuron):
        x1, y1 = neuron.position
        x2, y2 = self.position
        if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) <= 1:
            return True
        else:
            return False

    def initialize_neuron(self, network):
        """
        Connect this neuron to neighboring neurons with random weights
        :param network: list of neurons that this neuron can connect to
        :return: None
        """
        # Create connections and assign proper connection strengths
        self.network = network
        self.synaptic_weights = np.full(len(self.network), 0)
        for i in range(len(self.network)):
            if self.nearby(self.network[i]):
                self.synaptic_weights[i] = np.random.uniform(-0.5, 1.0)

        self.bias = random.uniform(0, 1.0)
        self.synaptic_weights[self.synaptic_weights < 0] = 0  # Ensure non-negative weights
        self.bias = max(0, self.bias)

        # Normalize weights to sum up to 1
        total_weight = np.sum(self.synaptic_weights) + self.bias
        if total_weight != 0:
            self.synaptic_weights /= total_weight
            self.bias /= total_weight

    def prime(self):
        """
        Always call before firing the neurons in the network, will get inputs auto-magically
        :return: None
        """
        self.inputs = np.array([neuron.output for neuron in self.neuron_connections])

    def fire(self):
        """
        Make a prediction with the neuron based on the input collected by the prime function

        training after every fire is recommended
        :return: None, get the output from neuron.output
        """
        self.output = np.dot(self.inputs, self.synaptic_weights) + self.bias
        self.output = max(0, min(self.output, 1))

    def train(self):
        """
        Use hebbian learning to train the weights in the network and move the neurons around. Requires no data.
        :return: None
        """
        # Update synaptic weights
        for i in range(len(self.network)):
            if self.nearby(self.network[i]):
                # Apply Hebbian-like learning rule to synaptic weights
                self.synaptic_weights[i] += self.learning_rate * self.inputs[i] * self.output

                # Ensure non-negative weights
                self.synaptic_weights[i] = max(0, self.synaptic_weights[i])

        # Update bias
        self.bias += self.learning_rate * self.output

        # Ensure non-negative bias
        self.bias = max(0, self.bias)

        # Normalize weights to add up to 1
        total_weight = sum(self.synaptic_weights) + self.bias
        self.synaptic_weights /= total_weight
        self.bias /= total_weight

        # move neurons
        for i in range(len(self.neuron_connections)):
            # Calculate vectors
            vector1 = np.array(self.neuron_connections[i].position) - np.array(self.position)
            vector2 = np.array(self.position) - np.array(self.neuron_connections[i].position)

            # Normalize and scale vectors
            norm_vector1 = (vector1 / np.linalg.norm(vector1)) * self.synaptic_weights[i]
            norm_vector2 = (vector2 / np.linalg.norm(vector2)) * self.synaptic_weights[i]

            # Move neurons
            self.position = tuple(np.array(self.position) + norm_vector1)
            self.neuron_connections[i].position = tuple(np.array(self.neuron_connections[i].position) + norm_vector2)


class AnchorNeuron:
    def __init__(self, position: tuple, learning_rate=0.1):
        """
        Create a stationary neuron. Typically used for output
        :param position: the position of the neuron, will not move
        :param learning_rate: how much to change the weights when training
        """
        # Save values of initialization
        self.position = position
        self.learning_rate = learning_rate

        # Prep variables for use
        self.network = []
        self.output = 0
        self.inputs = np.array([], dtype=np.float64)
        self.neuron_connections = []

        # learnable things:
        self.synaptic_weights = np.array([], dtype=np.float64)
        self.bias = None

    def initialize_neuron(self, network):
        """
        Connect this neuron to neighboring neurons with random weights
        :param network: list of neurons that this neuron can connect to
        :return: None
        """
        # Create connections and assign proper connection strengths
        self.network = network
        self.synaptic_weights = np.full(len(self.network), 0)
        for i in range(len(self.network)):
            if self.nearby(self.network[i]):
                self.synaptic_weights[i] = np.random.uniform(-0.5, 1.0)

        self.bias = random.uniform(0, 1.0)
        self.synaptic_weights[self.synaptic_weights < 0] = 0  # Ensure non-negative weights
        self.bias = max(0, self.bias)

        # Normalize weights to sum up to 1
        total_weight = np.sum(self.synaptic_weights) + self.bias
        if total_weight != 0:
            self.synaptic_weights /= total_weight
            self.bias /= total_weight

    def prime(self):
        """
        Always call before firing the neurons in the network, will get inputs auto-magically
        :return: None
        """
        self.inputs = np.array([neuron.output for neuron in self.neuron_connections])

    def fire(self):
        """
        Make a prediction with the neuron based on the input collected by the prime function
        :return: None, get the output from neuron.output
        """
        self.output = np.dot(self.inputs, self.synaptic_weights) + self.bias

    def train(self):
        """
        Use hebbian learning to train the weights in the network and move the neurons around. Requires no data.
        :return: None
        """
        # Update synaptic weights
        for i in range(len(self.network)):
            if self.nearby(self.network[i]):
                # Apply Hebbian-like learning rule to synaptic weights
                self.synaptic_weights[i] += self.learning_rate * self.inputs[i] * self.output

                # Ensure non-negative weights
                self.synaptic_weights[i] = max(0, self.synaptic_weights[i])

        # Update bias
        self.bias += self.learning_rate * self.output

        # Ensure non-negative bias
        self.bias = max(0, self.bias)

        # Normalize weights to add up to 1
        total_weight = sum(self.synaptic_weights) + self.bias
        self.synaptic_weights /= total_weight
        self.bias /= total_weight

        # move neurons
        for i in range(len(self.neuron_connections)):
            # Calculate vector of other neuorn, self will not move
            vector2 = np.array(self.position) - np.array(self.neuron_connections[i].position)

            # Normalize and scale vector
            norm_vector2 = (vector2 / np.linalg.norm(vector2)) * self.synaptic_weights[i]

            # Move neuron
            self.neuron_connections[i].position = tuple(np.array(self.neuron_connections[i].position) + norm_vector2)
