import pickle
import numpy as np


class Neuron:
    def __init__(self, network_size: int, activation, weight_initialization, memory_slots: int,
                 is_input_neuron=False, connection_decay_rate=0.1, max_connection_strength=10.0,
                 initial_connection_strength=5.0, learning_rate=0.1):
        """
        Create a neuron a self-managing neuron. Using input neurons to provide input to a network is advised.
        :param network_size: the number of neurons that this neuron will be able to connect to
        :param activation: user defined function: 1 input and returns a number
        :param weight_initialization: user defined function, takes 1 input of num_inputs, returns np array of weights
        :param memory_slots: number of memory slots, higher improves training  for hindsight but increases ram usage
        :param connection_decay_rate: how connections should decay, connections with no strength are trimmed, WIP
        :param max_connection_strength: the max strength of any one connection, WIP
        :param initial_connection_strength: how strong each connection should start at, WIP
        :param learning_rate: how much to change the weights when training
        """
        # Save values of initialization
        self.connection_decay_rate = connection_decay_rate
        self.max_connection_strength = max_connection_strength
        self.initial_connection_strength = initial_connection_strength
        self.learning_rate = learning_rate
        self.activation = activation
        self.memory_slots = memory_slots
        self.synaptic_weights = weight_initialization(network_size)

        # Prep variables for use
        self.connection_strengths = np.full(network_size, initial_connection_strength, dtype=np.float64)
        self.neuron_connections = np.empty(network_size, dtype=object)
        self.inputs = np.zeros(network_size, dtype=np.float64)
        self.is_input_neuron = is_input_neuron
        self.network = None
        self.output = 0.5
        self.baseline_score = 0.0
        self.input_memory = np.array([], dtype=np.float64)
        self.output_memory = np.array([], dtype=np.float64)

    def initialize_connections(self, network):
        """
        Connect this neuron to random neurons in the network.
        :param network: np array of neurons
        :return: None
        """
        self.network = network
        # Create random connections and assign proper connection strengths
        self.neuron_connections = np.random.choice(network, size=len(self.inputs), replace=False)

    def prime(self, inputs=None):
        """
        Always call before firing the neurons in the network
        :param inputs: array of floats, gets inputs automatically if undefined based on the connected neurons.
        :return: None
        """
        # Get inputs
        if inputs is None:
            self.inputs = np.array([neuron.output for neuron in self.neuron_connections])
        else:
            self.inputs = np.array(inputs, dtype=np.float64)

        # Debugging print statements (remove in production)
        print(f"\n{self.input_memory}")
        print(self.inputs)

        # Shift all items in array to make room for new input
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

        :return: None, call neuron.output to get it manually
        """
        if self.is_input_neuron is False:
            # Calculate sums, take averages, and apply activation function
            self.output = self.activation(np.sum(np.dot(self.inputs, self.synaptic_weights)) / len(self.inputs))

            # remember output
            self.output_memory = np.insert(self.output_memory, 0, self.inputs)
            # forget old output
            if len(self.output_memory) > self.memory_slots:
                self.output_memory = self.output_memory[:self.memory_slots]
        else:
            self.output = self.inputs
        print(self.output)

    def train(self, reward: float, backpropagations, reference_output):
        """
        Calculate changes based on the inputted value
        :param reward: The reward of the specific neuron
        :param backpropagations: how many neurons to back-propagate, higher improves learning, but requires more compute
        :param reference_output: Used to provide context to the neuron(s)
        :return: None
        """
        # check for context and get it
        memory_index = np.where(self.output_memory == reference_output)
        if len(memory_index[0]) > 0:
            total_input = sum(self.input_memory[memory_index])
            for i in range(len(self.synaptic_weights)):
                # check if the calculations need performing
                if self.synaptic_weights[i] > 0:
                    # get context
                    reference = self.input_memory[memory_index, i]
                    # modify weights
                    self.synaptic_weights[i] += self.learning_rate * reward * reference
                    # Increment memory by 1 and pass the signal to each connected neuron
                    if not self.is_input_neuron and backpropagations > 1:
                        # Back-propagate to the other neurons
                        connection_reward = reward / (reference / total_input)
                        self.neuron_connections[i].train(connection_reward, backpropagations - 1, reference)
                # reconnecting, WIP, Works, currently is inefficient
                else:
                    unconnected_neurons = np.where(self.synaptic_weights == 0)
                    for connection_index in unconnected_neurons[0]:
                        if reward < 0 and np.random.uniform(0, 1) > 0.9:
                            self.synaptic_weights[connection_index] = np.random.uniform(0, 0.05)

            # normalize weights to add up to 1
            total_weight = sum(self.synaptic_weights)
            self.synaptic_weights = [weight / total_weight for weight in self.synaptic_weights]
