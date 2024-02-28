import concurrent.futures
import pickle
import numpy as np
from typing import List

"""
!!!IMPORTANT NOTE!!!
nearly all constants will be made a hyperparameter for beta release and full releases
unless explicitly stated or the value is 0/None or in most cases 1
"""


class InputNeuron:
    def __init__(self, neuron_number: int, position: tuple[float, float]):
        """
        Create an input neuron, will only provide input to rest of network, does not do any processing on the data.
        :param neuron_number: Unique identifier for this neuron, numbered in order of creation; start at 0.
        :param position: The initial position of the input neuron, will not move from this position.
        Note: Neuron number is not saved, but should still be counted. Start at 0 (0, 1, 2, 3...) like list indexes.
        """
        self.number = neuron_number
        self.position = position
        self.output = 0

    def fire(self, output: float):
        """
        Does not do any processing, simply sets its output to the input.
        :param output: The output this neuron should provide.
        :return: None; get the output from neuron.output.
        """
        return output


class ActiveNeuron:
    def __init__(self, neuron_number: int, position: tuple[float, float], learning_rate: float):
        """
        Create a hidden neuron.
        :param neuron_number: Unique identifier for this neuron, numbered in order of creation, start at 0.
        :param position: The starting position of the hidden neuron.
        :param learning_rate: How quickly to change the weights when training.
        """
        # Save values of initialization
        self.number = neuron_number
        self.learning_rate = learning_rate
        self.position = position

        # Prep variables for use
        self.memory_index = 0
        self.network = None
        self.synaptic_weights = None

    def get_distances(self):
        """
        Internal function that gets distances to every other neuron in the network.
        :return: Array of distances to each neuron in network
        """
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
        self.synaptic_weights = np.random.uniform(0, 1, len(self.network))
        self.synaptic_weights[distances > 1] = 0

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
        :return: Output
        """
        # calculate output
        return np.dot(state, self.synaptic_weights)

    def train(self, positions: List[tuple[float, float]], context: List[float], reward: float):
        """
        Use hebbian learning to train the weights in the network and move the neurons around.
        :param positions: Array filled with tuples which are the positions of every neuron in the network.
        :param context: An array of outputs from every neuron, averaged.
        :param reward: The reward for the actions, acts as a simple scalar for weight changes; positive or negative.
        :return: None.
        note: context_size + cycle may NOT be more than (but can be equal to) the number of memory slots.
        """
        distances = self.get_distances()

        # Update synaptic weights
        mask = (0 < distances) and (distances < 1) or (self.synaptic_weights > 0.01)  # Use bitwise AND for element-wise comparison
        self.synaptic_weights[mask] += self.learning_rate * context[mask] * context[self.number] * reward

        # Ensure non-negative weights
        self.synaptic_weights[self.synaptic_weights < 0] = 0

        # Set weight for self to 0
        self.synaptic_weights[self.number] = 0

        # Normalize weights to add up to 1
        self.synaptic_weights /= np.sum(self.synaptic_weights)

        # move neurons
        # Exclude near zero weights
        valid_indices = np.where(self.synaptic_weights > 0.01)[0]

        # Calculate vectors
        vectors = np.array(self.position) - positions[valid_indices]

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1)
        normalized_vectors = vectors / norms[:, np.newaxis]

        # Scale vectors
        scaled_vectors = normalized_vectors * self.synaptic_weights[:, np.newaxis]

        # Sum vectors
        total_scaled_vector = np.sum(scaled_vectors, axis=0)

        self.position += total_scaled_vector


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
        self.network = sorted(input_neurons + hidden_neurons + output_neurons, key=lambda x: x.number)

        # setup self.input_memory
        self.memory = np.zeros((memory_slots, len(self.network)))
        self.memory_index = 0
        self.network_state = np.zeros((len(self.network)))



    def propagate_input(self, inputs: List[float]) -> List[float]:
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: Array of shape *number of input neurons*.
        :return: Outputs of the network in an array.
        """
        outputs = []

        current_state = np.zeros(len(self.network))

        # Handle input neurons separately
        for output, neuron in zip(inputs, self.input_neurons):
            current_state[neuron.number] = neuron.fire(output)

        # compute update for each neuron in parallel
        def compute_prediction_batch(batch):
            for neuron in batch:
                current_state[neuron.number] = neuron.fire(self.network_state)

        # divide neurons into batches
        batch_size = 500 # Adjust as needed
        batches = [self.internal_neurons[i:i + batch_size] for i in range(0, len(self.internal_neurons), batch_size)]

        # perform calculations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(compute_prediction_batch, batches)

        # collect outputs, update memory, and update network state
        self.memory[self.memory_index] = current_state
        self.memory_index = (self.memory_index + 1) % len(self.memory)

        self.network_state = current_state

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

        # Actually get context and average each section
        context = np.sum(self.memory[start_index: start_index + context_size: -1], axis=0) / context_size

        # Get positions
        positions = np.array([neuron.position for neuron in self.network])

        # Train in parallel
        def compute_training_batch(batch):
            for neuron in batch:
                neuron.train(positions, context, reward)

        # divide neurons into batches
        batch_size = 500
        batches = [self.internal_neurons[i:i + batch_size] for i in range(0, len(self.internal_neurons), batch_size)]

        # perform calculations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(compute_training_batch, batches)

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
