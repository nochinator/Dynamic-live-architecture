import numpy as np
import random


class InputNeuron:
    def __init__(self, position: tuple):
        """
        Create an input neuron, will only provide input to rest of network, does not do any processing on the data
        :param position: the initial position of the input neuron, it will not move from this position, use 2D tuples.
        """
        self.position = position
        self.output = np.float32(0)

    def fire(self, input: np.float32):
        """
        does not do any processing, simply sets its output to the input
        :return: None, get the output from neuron.output
        """
        self.output = input


class HiddenNeuron:
    def __init__(self, memory_slots: int, position: tuple, learning_rate=np.float32(0.1)):
        """
        Create a hidden neuron.
        :param position: the starting position of the hidden neuron, the neuron will move around
        :param learning_rate: how quickly to change the weights when training
        """
        # Save values of initialization
        self.input_memory = None
        self.output_memory = np.zeros(memory_slots, dtype=np.float32)
        self.memory_index = 0
        self.learning_rate = learning_rate
        self.position = position

        # Prep variables for use
        self.output = np.float32(0)
        self.inputs = None
        self.network = []
        self.synaptic_weights = None

    def nearby(self, neuron):
        x1, y1 = neuron.position
        x2, y2 = self.position
        distance = np.sqrt((x2 - x1) ** np.float32(2) + (y2 - y1) ** np.float32(2))
        return distance <= np.float32(1)

    def initialize_neuron(self, network):
        """
        Connect this neuron to neighboring neurons with random weights
        :param network: list of every neuron in the network
        :return: None
        """
        self.network = network

        # Calculate distances between neurons
        distances = np.linalg.norm(np.array(self.position) - np.array([neuron.position for neuron in network]), axis=1)

        # Initialize synaptic weights with random values
        self.synaptic_weights = np.where(distances <= 1, np.random.uniform(0, 1, len(distances)), 0)

        # Normalize weights to sum up to 1
        total_weight = np.sum(self.synaptic_weights)
        if total_weight != 0:
            self.synaptic_weights /= total_weight

        # setup self.input_memory
        self.input_memory = np.zeros((len(self.output_memory), len(self.network)), dtype=np.float32)

    def prime(self):
        """
        Always call before firing the neurons in the network, will get inputs auto-magically
        :return: None
        """
        # get inputs
        nearby_mask = np.array([self.nearby(neuron) for neuron in self.network], dtype=bool)
        nearby_outputs = np.array([neuron.output for neuron in np.array(self.network)[nearby_mask]], dtype=np.float32)
        self.inputs = np.zeros(len(self.network), dtype=np.float32)
        self.inputs[nearby_mask] = nearby_outputs
        self.inputs = np.reshape(self.inputs, (1, len(self.network)))  # Reshape inputs to match input_memory

        # update memory
        self.input_memory[self.memory_index] = self.inputs

    def fire(self):
        """
        Make a prediction with the neuron based on the input collected by the prime function

        training after every fire is recommended
        :return: None, get the output from neuron.output
        """
        # calculate output
        self.output = np.dot(self.inputs, self.synaptic_weights)

        # update memory
        self.output_memory[self.memory_index] = self.output

        # update memory position
        self.memory_index = (self.memory_index + 1) % len(self.output_memory)

    def train(self, cycle: int, context_size: int):
        """
        Use hebbian learning to train the weights in the network and move the neurons around. Requires no data.
        :param cycle: how many cycles back in memory to consider the start of context, 0 is the most recent cycle
        :param context_size: how many neurons back from the start of context to consider in context
        :return: None
        note: context_size + cycle can NOT be more than (but can be equal to) the number of memory slots
        """
        if context_size + cycle > len(self.output_memory):
            print("invalid training context, can not access data further back than amount of memory slots")
            return

        # Prepare to get context
        start_index = (self.memory_index - cycle) % len(self.output_memory)

        # Actually get context
        input_context = np.sum(self.input_memory[start_index: start_index - context_size: -1], axis=0) / context_size
        output_context = np.sum(self.output_memory[start_index: start_index - context_size: -1]) / context_size

        # Update synaptic weights
        for i, weight in enumerate(self.synaptic_weights):
            if self.nearby(self.network[i]):
                # Apply Hebbian-like learning rule to synaptic weights
                weight += self.learning_rate * input_context[i] * output_context

                # Ensure non-negative weights
                self.synaptic_weights[i] = max(np.float32(0), weight)

        # Normalize weights to add up to 1
        total_weight = sum(self.synaptic_weights)
        self.synaptic_weights /= total_weight

        # move neurons
        for i, neuron in enumerate(self.network):
            if self.nearby(neuron):
                # Calculate vectors
                vector1 = np.array(neuron.position) - np.array(self.position)
                vector2 = np.array(self.position) - np.array(neuron.position)

                # Normalize and scale vectors
                norm_vector1 = (vector1 / np.linalg.norm(vector1)) * self.synaptic_weights[i]
                norm_vector2 = (vector2 / np.linalg.norm(vector2)) * self.synaptic_weights[i]

                # Move neurons
                self.position = tuple(np.array(self.position) + norm_vector1)
                neuron.position = tuple(np.array(neuron.position) + norm_vector2)



class AnchorNeuron:
    def __init__(self, memory_slots: int, position: tuple, learning_rate=np.float32(0.1)):
        """
        Create an anchor neuron.
        :param position: the starting position of the hidden neuron, the neuron will NOT move around
        :param learning_rate: how quickly to change the weights when training
        """
        # Save values of initialization
        self.input_memory = None
        self.output_memory = np.zeros(memory_slots, dtype=np.float32)
        self.memory_index = 0
        self.learning_rate = learning_rate
        self.position = position

        # Prep variables for use
        self.output = np.float32(0)
        self.inputs = None
        self.network = []
        self.synaptic_weights = None

    def nearby(self, neuron):
        x1, y1 = neuron.position
        x2, y2 = self.position
        squared_distance = (x2 - x1) + (y2 - y1)
        return squared_distance <= 1

    def initialize_neuron(self, network):
        """
        Connect this neuron to neighboring neurons with random weights
        :param network: list every neuron in the network
        :return: None
        """
        self.network = network

        # Calculate distances between neurons
        distances = np.linalg.norm(np.array(self.position) - np.array([neuron.position for neuron in network]), axis=1)

        # Initialize synaptic weights with random values
        self.synaptic_weights = np.where(distances <= 1, np.random.uniform(0, 1, len(distances)), 0)

        # Normalize weights to sum up to 1
        total_weight = np.sum(self.synaptic_weights)
        if total_weight != 0:
            self.synaptic_weights /= total_weight

        # setup self.input_memory
        self.input_memory = np.zeros((len(self.output_memory), len(self.network)), dtype=np.float32)

    def prime(self):
        """
        Always call before firing the neurons in the network, will get inputs auto-magically
        :return: None
        """
        # get inputs
        self.inputs = np.array([neuron.output if self.nearby(neuron) else np.float32(0) for neuron in self.network],
                               dtype=np.float32)

        self.inputs = np.reshape(self.inputs, (1, len(self.network)))  # Reshape inputs to match input_memory

        # update memory
        self.input_memory[self.memory_index] = self.inputs

    def fire(self):
        """
        Make a prediction with the neuron based on the input collected by the prime function

        training after every fire is recommended
        :return: None, get the output from neuron.output
        """
        # calculate output
        self.output = np.dot(self.inputs, self.synaptic_weights)

        # update memory
        self.output_memory[self.memory_index] = self.output

        # update memory position
        self.memory_index = (self.memory_index + 1) % len(self.output_memory)

    def train(self, cycle: int, context_size: int):
        """
        Use hebbian learning to train the weights in the network and move the neurons around. Requires no data.
        :param cycle: how many cycles back in memory to consider the start of context, 0 is the most recent cycle
        :param context_size: how many neurons back from the start of context to consider in context
        :return: None
        note: context_size + cycle can NOT be more than (but can be equal to) the number of memory slots
        """
        if context_size + cycle > len(self.output_memory):
            print("invalid training context, can not access data further back than amount of memory slots")
            return

        # Prepare to get context
        start_index = (self.memory_index - cycle) % len(self.output_memory)

        # Actually get context
        input_context = np.sum(self.input_memory[start_index: start_index - context_size: -1], axis=0) / context_size
        output_context = np.sum(self.output_memory[start_index: start_index - context_size: -1]) / context_size

        # Update synaptic weights
        for i, weight in enumerate(self.synaptic_weights):
            if self.nearby(self.network[i]):
                # Apply Hebbian-like learning rule to synaptic weights
                weight += self.learning_rate * input_context[i] * output_context

                # Ensure non-negative weights
                self.synaptic_weights[i] = max(np.float32(0), weight)

        # Normalize weights to add up to 1
        total_weight = sum(self.synaptic_weights)
        self.synaptic_weights /= total_weight
