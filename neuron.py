import numpy as np


class Neuron:
    def __init__(self, memory_slots: int, learning_rate=0.1, time_sensitive_neuron=True, is_input_neuron=False):
        """
        Create a neuron a self-managing neuron. Using input neurons to provide input to a network is advised.
        :param memory_slots: number of memory slots, higher improves training  for hindsight but increases ram usage
        :param is_input_neuron: Set true if the neuron will be used on the input layer
        :param time_sensitive_neuron: If true then neuron will incorporate past inputs along with current inputs
        :param learning_rate: how much to change the weights when training
        """
        # Save values of initialization
        self.time_sensitive_neuron = time_sensitive_neuron
        self.learning_rate = learning_rate
        self.memory_slots = memory_slots
        self.is_input_neuron = is_input_neuron

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
        Connect this neuron to random neurons in the network.
        :param network: list of neurons that this neuron can connect to
        :return: None
        """
        # Create connections and assign proper connection strengths
        self.neuron_connections = network
        self.synaptic_weights = np.random.uniform(-1.0, 1.0, len(self.neuron_connections))
        self.bias = np.random.uniform(-1.0, 1.0)
        self.synaptic_weights[self.synaptic_weights < 0] = 0  # Ensure non-negative weights
        if self.bias < 0: self.bias = 0  # Ensure non-negative weights

        # Normalize weights to sum up to 1
        total_weight = np.sum(self.synaptic_weights) + self.bias
        if total_weight != 0:
            self.synaptic_weights /= total_weight
            self.bias /= total_weight

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
            if self.time_sensitive_neuron:
                # Calculate sums, take averages, and update action potential
                self.action_potential -= self.decay
                self.action_potential += np.dot(self.inputs, self.synaptic_weights) + self.bias

                self.output = 0

                if self.action_potential >= 1:
                    self.output = 1
                    self.decay = 1
                else:
                    self.decay /= 1.5
            else:
                if np.dot(self.inputs, self.synaptic_weights) + self.bias >= 1:
                    self.output = 1

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

    def train(self, correct_output: bool, backpropagations=0, cycles=0):
        """
        Calculate changes based on the inputted value
        :param correct_output: self-explanatory, if true connections strengthen, if false then weaken
        :param backpropagations: how many neurons to back-propagate, higher improves network learning, but requires more compute
        :param cycles: reference to x time steps back, 0 is the latest time step, limited by neurons memory size
        :return: None
        """
        if not self.is_input_neuron and cycles < len(self.input_memory) and cycles < len(self.output_memory):
            # get input context
            input_context = self.input_memory[cycles]
            output_context = self.output_memory[cycles]

            for i in range(len(self.synaptic_weights)):
                # check if the neuron is connected or not
                if self.synaptic_weights[i] > 0:
                    # Increment memory by 1 and pass the signal to each connected neuron
                    if backpropagations > 1:
                        # Back-propagate to the other neurons
                        self.neuron_connections[i].train(correct_output * self.synaptic_weights[i], backpropagations - 1,
                                                         cycles + 1)

                # Calculate modifier based on network performance
                modifier = 1 if correct_output else -1

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
