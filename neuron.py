import numpy as np


class Neuron:
    def __init__(self, memory_slots: int, learning_rate=0.1, is_input_neuron=False):
        """
        Create a neuron a self-managing neuron. Using input neurons to provide input to a network is advised.
        :param memory_slots: number of memory slots, higher improves training  for hindsight but increases ram usage
        :param is_input_neuron: Set true if the neuron will be used on the input layer
        :param learning_rate: how much to change the weights when training
        """
        # Save values of initialization
        self.learning_rate = learning_rate
        self.memory_slots = memory_slots
        self.is_input_neuron = is_input_neuron

        # Prep variables for use
        self.output = 0
        self.inputs = np.array([], dtype=np.float64)
        self.input_memory = np.array([], dtype=np.float64)
        self.output_memory = np.array([], dtype=np.float64)
        self.neuron_connections = []
        self.synaptic_weights = np.array([], dtype=np.float64)
        self.bias = 0

    def initialize_connections(self, network):
        """
        Connect this neuron to random neurons in the network.
        :param network: list of neurons that this neuron can connect too, np array is slower here
        :return: None
        """
        # Create random connections and assign proper connection strengths
        self.neuron_connections = network
        distributed_weights = len(network), 1 / len(network) + 1
        self.synaptic_weights = np.full(distributed_weights, dtype=np.float64)

    def prime(self, inputs=None):
        """
        Always call before firing the neurons in the network
        :param inputs: array of floats, if undefined neuron gets inputs automatically based on the connected neurons.
        :return: None
        """

        # Get inputs
        if inputs is None:
            self.inputs = np.array([neuron.output for neuron in self.neuron_connections])
        else:
            self.inputs = np.array([inputs], dtype=np.float64)

        # Shift all items in array to make room for new inputs in memory
        self.input_memory = np.roll(self.input_memory, axis=0, shift=1)

        # Remember inputs
        if len(self.input_memory) != 0:
            self.input_memory[0] = self.inputs
        else:
            self.input_memory = np.array([self.inputs])

        # Forget old inputs
        if len(self.input_memory) > self.memory_slots:
            self.input_memory = self.input_memory[:self.memory_slots]

    def fire(self):
        """
        Make a prediction with the neuron based on the input collected by the prime function
        :return: None, get the output from neuron.output
        """
        if not self.is_input_neuron:
            # Calculate sums, take averages, and apply activation function
            self.output = (np.sum(np.dot(self.inputs, self.synaptic_weights)) / np.sum(self.inputs)) ** 2 \
                if np.sum(self.inputs) != 0 else 0

            # Shift all items in array to make room for new inputs in memory
            self.output_memory = np.roll(self.input_memory, axis=0, shift=1)

            # Remember inputs
            if len(self.output_memory) != 0:
                self.output_memory[0] = self.output
            else:
                self.output_memory = np.array([self.output])
            # Forget old inputs
            if len(self.output_memory) > self.memory_slots:
                self.output_memory = self.output_memory[:self.memory_slots]
        # for input neurons
        else:
            self.output = self.inputs[0]

    def train(self, reward: float, backpropagations: int, reference_output=None):
        """
        Calculate changes based on the inputted value
        :param reward: The reward of the specific neuron
        :param backpropagations: how many neurons to back-propagate, higher improves learning, but requires more compute
        :param reference_output: Used to provide context to the neuron(s)
        :return: None
        """
        if not self.is_input_neuron:
            # check for context
            if reference_output is not None:
                memory_index = np.argmax(self.output_memory == reference_output)
            else:
                memory_index = 0

            for i in range(len(self.synaptic_weights)):
                # check if the neuron is connected or not
                if self.synaptic_weights[i] > 0:

                    # get context
                    reference = self.input_memory[memory_index, i]
                    # modify weights
                    self.synaptic_weights[i] += self.learning_rate * reward * reference

                    # Increment memory by 1 and pass the signal to each connected neuron
                    if backpropagations > 1:
                        # Back-propagate to the other neurons
                        connection_reward = reward / len(self.neuron_connections)  # under research functional but WIP
                        self.neuron_connections[i].train(connection_reward, backpropagations - 1, reference)
                # reconnecting, functional but WIP, reconnects randomly, but won't when finished
                else:
                    unconnected_neurons = np.where(self.synaptic_weights == 0)
                    for connection_index in unconnected_neurons[0]:
                        if reward < 0 and np.random.uniform(0, 1) > 0.9:
                            self.synaptic_weights[connection_index] = np.random.uniform(0, 0.05)

                # normalize weights to add up to 1
                total_weight = sum(self.synaptic_weights)
                self.synaptic_weights /= total_weight
