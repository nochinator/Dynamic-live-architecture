import numpy as np

import numpy as np


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
    def __init__(self, position: tuple, learning_rate=0.1, time_sensitive_neuron=True):
        """
        Create a neuron a biologically inspired hidden neuron.

        :param is_input_neuron: Set true if the neuron will be used on the input layer
        :param time_sensitive_neuron: If true then neuron will incorporate past inputs along with current inputs
        :param learning_rate: how much to change the weights when training
        """
        # Save values of initialization
        self.time_sensitive_neuron = time_sensitive_neuron
        self.learning_rate = learning_rate
        # self.memory_slots = memory_slots

        # Prep variables for use
        self.output = 0
        self.inputs = np.array([], dtype=np.float64)
        # self.input_memory = np.array([], dtype=np.float64)
        # self.output_memory = np.array([], dtype=np.float64)
        self.neuron_connections = []
        self.action_potential = 0
        self.decay = 0

        # learnable things:
        self.synaptic_weights = np.array([], dtype=np.float64)
        self.bias = None

    def initialize_connections(self, network):
        """
        Connect this neuron to neighboring neurons with random weights
        :param network: list of neurons that this neuron can connect to
        :return: None
        """
        # Create connections and assign proper connection strengths
        self.neuron_connections = network
        self.synaptic_weights = np.random.uniform(-1.0, 1.0, len(self.neuron_connections))
        self.bias = np.random.uniform(-0.1, 1.0)
        self.synaptic_weights[self.synaptic_weights < 0] = 0  # Ensure non-negative weights
        if self.bias < 0: self.bias = 0  # Ensure non-negative weights

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
        for i in range(len(self.synaptic_weights)):
            # Apply Hebbian-like learning rule to synaptic weights
            self.synaptic_weights[i] += self.learning_rate * self.inputs[i] * self.output

            # Ensure non-negative weights
            self.synaptic_weights[i] = max(0, self.synaptic_weights[i])

        # Update bias
        self.bias += self.learning_rate * self.output

        # Ensure non-negative bias
        self.bias = max(0, self.bias)

        # Normalize weights to add up to 1
        total_weight = sum(self.synaptic_weights)
        self.synaptic_weights /= total_weight


class OutputNeuron:
    def __init__(self, position: tuple, memory_slots=1, learning_rate=0.1, time_sensitive_neuron=True):
        """
        Create a neuron a biologically inspired hidden neuron.
        :param position: the position of the output neuron, will not move
        :param memory_slots: number of memory slots, higher improves training for hindsight but increases ram usage
        :param time_sensitive_neuron: If true then neuron will incorporate past inputs along with current inputs
        :param learning_rate: how much to change the weights when training
        """
        # Save values of initialization
        self.time_sensitive_neuron = time_sensitive_neuron
        self.learning_rate = learning_rate
        self.memory_slots = memory_slots

        # Prep variables for use
        self.output = 0
        self.inputs = np.array([], dtype=np.float64)
        self.input_memory = np.array([], dtype=np.float64)
        self.output_memory = np.array([], dtype=np.float64)
        self.neuron_connections = []
        self.action_potential = 0
        self.decay = 0

        # learnable things:
        self.synaptic_weights = np.array([], dtype=np.float64)
        self.bias = None

    def initialize_connections(self, network):
        """
        Connect this neuron to neighboring neurons with random weights
        :param network: list of neurons that this neuron can connect to
        :return: None
        """
        # Create connections and assign proper connection strengths
        self.neuron_connections = network
        self.synaptic_weights = np.random.uniform(-1.0, 1.0, len(self.neuron_connections))
        self.bias = np.random.uniform(-0.1, 1.0)
        self.synaptic_weights[self.synaptic_weights < 0] = 0  # Ensure non-negative weights
        if self.bias < 0: self.bias = 0  # Ensure non-negative weights

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

        # Get inputs
        self.inputs = np.array([neuron.output for neuron in self.neuron_connections])

        """# Shift all items in array to make room for new inputs in memory
        self.input_memory = np.roll(self.input_memory, axis=0, shift=1)

        # Remember inputs
        if len(self.input_memory) != 0:
            self.input_memory[0] = self.inputs
        else:
            self.input_memory = np.array([self.inputs])

        # Forget old inputs
        if len(self.input_memory) > self.memory_slots:
            self.input_memory = self.input_memory[:self.memory_slots]"""

    def fire(self):
        """
        Make a prediction with the neuron based on the input collected by the prime function
        :return: None, get the output from neuron.output
        """

            self.output = np.dot(self.inputs, self.synaptic_weights) + self.bias

            """# Shift all items in array to make room for new inputs in memory
            self.output_memory = np.roll(self.input_memory, axis=0, shift=1)

            # Remember outputs
            if len(self.output_memory) != 0:
                self.output_memory[0] = self.output
            else:
                self.output_memory = np.array([self.output])
            # Forget old inputs
            if len(self.output_memory) > self.memory_slots:
                self.output_memory = self.output_memory[:self.memory_slots]"""

    def train(self):
        """
        Use hebbian learning to train the weights in the network and move the neurons around. Requires no data.
        :return: None
        """
        # if and cycles < len(self.input_memory) and cycles < len(self.output_memory):
        # get input context
        # input_context = self.input_memory[cycles]
        # output_context = self.output_memory[cycles]

        # Update synaptic weights
        for i in range(len(self.synaptic_weights)):
            # Apply Hebbian-like learning rule to synaptic weights
            self.synaptic_weights[i] += self.learning_rate * input_context[i] * output_context * modifier

            # Ensure non-negative weights
            self.synaptic_weights[i] = max(0, self.synaptic_weights[i])

        # Update bias
        self.bias += self.learning_rate * output_context * modifier

        # Ensure non-negative bias
        self.bias = max(0, self.bias)

        # Normalize weights to add up to 1
        total_weight = sum(self.synaptic_weights)
        self.synaptic_weights /= total_weight