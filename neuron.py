import pickle
import numpy as np


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def initialize_weights(num_inputs, method):
    """
    internal function, calling manually may break things
    :param num_inputs: num of inputs
    :param method: method of initialization
    :return: None
    """
    if method == "random":
        return np.random.randn(num_inputs)
    elif method == "xavier":
        return np.random.randn(num_inputs) * np.sqrt(1.0 / num_inputs)
    elif method == "he":
        return np.random.randn(num_inputs) * np.sqrt(2.0 / num_inputs)
    elif method == "lecun":
        return np.random.randn(num_inputs) * np.sqrt(1.0 / num_inputs)
    elif method == "orthogonal":
        shape = (num_inputs, num_inputs)
        _, s, vh = np.linalg.svd(np.random.randn(*shape))
        return vh[:num_inputs, :]
    elif method == "ones":
        return np.ones(num_inputs)
    else:
        raise ValueError(f"Unknown weight initialization method: {method}")


class Neuron:
    def __init__(self, num_inputs: int, activation, activation_derivative, weight_initialization, memory_slots: int,
                 is_input_neuron=False, connection_decay_rate=0.1, max_connection_strength=10.0,
                 initial_connection_strength=5.0, learning_rate=0.1):
        """
        Create a neuron a self-managing neuron. Using input neurons to provide input to a network is advised.
        :param num_inputs: number of inputs the neuron should take
        :param activation: user defined function: 1 input and returns a number
        :param activation_derivative: user defined function: 1 input which is a weight, should return
        :param weight_initialization: user defined function, takes 1 input of num_inputs, returns np array of weights
        :param memory_slots: number of memory slots, higher improves accuracy but increases computati
        :param connection_decay_rate: how connections should decay, connections with no strength are trimmed, WIP
        :param max_connection_strength: the max strength of any one connection, WIP
        :param initial_connection_strength: how strong each connection should start at, WIP
        :param learning_rate: how much to change the weights when training
        """
        # Save values of initialization
        self.num_inputs = num_inputs
        self.connection_decay_rate = connection_decay_rate
        self.max_connection_strength = max_connection_strength
        self.initial_connection_strength = initial_connection_strength
        self.learning_rate = learning_rate
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.memory_slots = memory_slots
        self.synaptic_weights = weight_initialization(num_inputs)

        # Prep variables for use
        self.connection_strengths = np.full(num_inputs, initial_connection_strength, dtype=np.float64)
        self.neuron_connections = np.empty(num_inputs, dtype=object)
        self.inputs = np.zeros(num_inputs, dtype=np.float64)
        self.is_input_neuron = is_input_neuron
        self.network = None
        self.output = 0.5
        self.baseline_score = 0.0
        self.input_memory = np.array([], dtype=np.float64)
        self.activation_input_memory = np.array([], dtype=np.float64)
        self.output_memory = np.array([], dtype=np.float64)

    def initialize_connections(self, network):
        """
        Connect this neuron to random neurons in the network.
        :param network: np array of neurons
        :return: None
        """
        self.network = network
        # Create random connections and assign proper connection strengths
        self.neuron_connections = np.random.choice(network, size=self.num_inputs, replace=False)

    def reset_connection(self, index):
        """
        remove a connection and form a new one
        :param index: the connection to reset
        :return: None
        """
        self.synaptic_weights[index] = 0.5
        self.neuron_connections[index] = np.random.choice(self.network)

    def update_connection_strengths(self, inputs):
        """
        WIP
        DO NOT USE
        :param inputs: array of inputs called on the last fire
        :return:
        """
        # Calculate the decay factor
        decay = self.connection_decay_rate * np.ones(self.num_inputs, dtype=np.float64)

        # Update connection strengths
        self.connection_strengths -= decay
        self.connection_strengths += inputs

        # Clip connection strengths
        self.connection_strengths = np.clip(self.connection_strengths, 0.0, self.max_connection_strength)

        # Find and trim connections with no strength
        zero_indices = self.connection_strengths == 0
        self.connection_strengths[zero_indices] = self.initial_connection_strength
        self.neuron_connections[zero_indices] = np.random.choice(self.network, size=np.count_nonzero(zero_indices))

    def prime(self, inputs=None):
        """
        always call before firing the neurons in the network
        :param inputs: array of floats, gets inputs automatically if undefined based on the connected neurons.
        :return:
        """
        # Get inputs
        if inputs is None:
            inputs = np.array([neuron.output for neuron in self.neuron_connections], dtype=np.float64)
        else:
            inputs = np.array(inputs, dtype=np.float64)

        self.inputs = inputs

        # remember inputs
        self.input_memory = np.insert(self.input_memory, 0, self.inputs)
        # forget old inputs
        if len(self.input_memory) > self.memory_slots:
            self.input_memory = self.input_memory[:self.memory_slots]

    def fire(self, update_connection_strengths=False):
        """
        Make a prediction with the neuron based on the input

        note: when a connection strength reaches zero the connection is trimmed and a new connection formed else where
        :param update_connection_strengths: weather or not to update the connection strengths automatically.
        :return: None, call neuron.output to get it manually
        """
        # Calculate sums and apply activation
        if self.is_input_neuron is False:
            weighted_sum = np.dot(self.inputs, self.synaptic_weights)
            weighted_sum = np.sum(weighted_sum)  # Limit the range of weighted sum
            activation = self.activation(weighted_sum)

            # Update connections
            if update_connection_strengths:
                self.update_connection_strengths(self.inputs)
            # Save output of the neuron
            self.output = activation

            # remember output
            self.output_memory = np.insert(self.output_memory, 0, self.inputs)
            # forget old output
            if len(self.output_memory) > self.memory_slots:
                self.output_memory = self.output_memory[:self.memory_slots]

            # remember activation input
            self.activation_input_memory = np.insert(self.activation_input_memory, 0, self.inputs)
            # forget old activation input
            if len(self.activation_input_memory) > self.memory_slots:
                self.activation_input_memory = self.activation_input_memory[:self.memory_slots]
        else:
            self.output = self.inputs

    def train(self, loss: float, backpropogations):
        """
        Calculate changes based on the inputted value
        :param loss: The loss of the specific neuron, also works as reinforcement with lower scores being better.
        :param backpropogations: how many neurons to back-propagate, higher improves learning, but requires more compute
        :return: None
        """
        for i in range(len(self.synaptic_weights)):
            # Gradient descent
            gradient = loss * self.activation_derivative(self.input_memory)
            self.synaptic_weights[i] += self.learning_rate * gradient

            # Increment memory by 1 and pass the signal to each connected neuron
            if not self.is_input_neuron:
                # Back-propagate to the other neurons
                self.neuron_connections[i].train(loss, backpropogations - 1)


# May be removed
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def minimize(self, param_grad_pairs):
        if self.m is None:
            self.m = [np.zeros_like(params) for params, grads in param_grad_pairs]
            self.v = [np.zeros_like(params) for params, grads in param_grad_pairs]

        self.t += 1

        for i, (params, grads) in enumerate(param_grad_pairs):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Apply the update to params individually
            params[:] -= update


class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers, output_neurons, learning_rate):
        """
        create a neural network using the Neuron class.
        :param input_neurons: expected format: np.array[InputNeuron(), InputNeuron(), etc.]
        :param hidden_layers: expected format: np.array[[Neuron(), Neuron()], [Neuron(), Neuron()], etc.]
        :param output_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        :param learning_rate: how much to update the weights when training or reinforcing
        """
        self.input_neurons = input_neurons
        self.hidden_layers = hidden_layers
        self.output_neurons = output_neurons
        concatenated_hidden_layers = np.concatenate([hidden_layer.ravel() for hidden_layer in hidden_layers])
        self.active_neurons = np.concatenate((concatenated_hidden_layers, output_neurons))
        self.network = np.concatenate((input_neurons, self.active_neurons))

        self.learning_rate = learning_rate
        self.neuron_importance = []
        # Create an instance of the AdamOptimizer
        self.adam_optimizer = AdamOptimizer(learning_rate=learning_rate)

    def propagate_input(self, inputs, update_connection_strength=True):
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: array of shape (number of input neurons,)
        :param update_connection_strength:
        :return: network outputs
        """
        outputs = []

        # Prime neurons
        for i, neuron in enumerate(self.input_neurons):
            neuron.prime(inputs[i])

        # Fire input neurons
        for neuron in self.input_neurons:
            neuron.fire(update_connection_strength)

        # Hidden neurons layer by layer
        for layer in self.hidden_layers:
            # Prime neurons in the layer
            for neuron in layer:
                neuron.prime()
            # Fire neurons in the layer
            for neuron in layer:
                neuron.fire(update_connection_strength)

        # Output neurons
        for neuron in self.output_neurons:
            # Prime output neuron
            neuron.prime()
        for neuron in self.output_neurons:
            # Fire output neuron
            neuron.fire(update_connection_strength)
            outputs.append(neuron.output)

        return outputs

    def reinforce(self, reward: float):
        """
        perform reinforcement on the entire network.
        :param reward: the score to rate network, between -1 and 1
        :param neuron_importance: Array of how important each output_neuron's output is, should add up to 1, defaults to equal distrubution
        :return: None
        """
        if neuron_importance is None:
            neuron_importance = np.full(len(self.output_neurons), 1 / len(self.output_neurons))
        for neuron in self.output_neurons:
            neuron.train(reward, neuron_importance)
        # trigger train functions on the output neurons with, and by extension the backpropogation

    def train(self, X_train, y_train, update_connections):
        """
        Train the network based on expected input and output

        Does not implement epochs, please only do one input and output.
        :param X_train: input to train on
        :param y_train: The expected output of the input
        :param update_connections: Should the connection strengths and by extension connections be updated
        :param neuron_importance: Array of how important each output_neuron's output is, should add up to 1, defaults to equal distrubution
        :return predicted output with X_train input
        """
        # make predictions
        output = self.propagate_input(X_train, update_connections)

        for i, neuron in enumerate(self.output_neurons):
            # Calculate Mean Squared Error (MSE) loss
            loss = np.mean(np.subtract(y_train[i], output[i]) ** 2)
            neuron.train(loss, output[i])
        return output
    # extra

    def save_model(self, file_path):
        """
        Save the created neural network as a pkl. resulting file contains all data needed to reconstruct the network.
        :param file_path: path to save file at
        :return: None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_path):
        """
        Load a pkl file that contains a neural network. File must contain all data needed to reconstruct the network.
        :param file_path: path to load file from
        :return: None
        """
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model


def leaky_relu(x):
    np.where(x < 0, x * 0.01, x)


def leaky_relu_derivative(x):
    return np.where(x < 0, 0.01, 1)


def random_weights(x):
    return np.random.rand(x)


def main():
    # XOR data
    X_train = np.array([[0.01, 0.01], [0.01, 1], [1, 0.01], [1, 1]])
    y_train = np.array([[0.01], [1], [1], [0.01]])

    # Create a neural network for XOR problem
    input_neurons = np.array(
        [Neuron(1, activation=leaky_relu, activation_derivative=leaky_relu_derivative,
                weight_initialization=random_weights, memory_slots=1, is_input_neuron=True) for _ in
         range(2)])
    hidden_layer = np.array([
        [Neuron(num_inputs=2, activation=leaky_relu, activation_derivative=leaky_relu_derivative,
                weight_initialization=random_weights, memory_slots=1) for _ in range(2)]])
    output_neurons = np.array(
        [Neuron(num_inputs=2, activation=leaky_relu, activation_derivative=leaky_relu_derivative,
                weight_initialization=random_weights, memory_slots=1)])

    # Initialize connections
    for neuron in hidden_layer[0]:
        neuron.initialize_connections(input_neurons)
    output_neurons[0].initialize_connections(hidden_layer[0])

    neural_network = NeuralNetwork(
        input_neurons=input_neurons,
        hidden_layers=hidden_layer,
        output_neurons=output_neurons,
        learning_rate=0.01
    )

    # Train the neural network on XOR data
    epochs = 10000
    for epoch in range(epochs):
        output = []
        for i in range(len(X_train)):
            output.append(neural_network.train(X_train[i], y_train[i], False))
        if epoch % 1000 == 0:
            print(
                f"\nEpoch {epoch}: Predictions - {[round(prediction[0], 3) for prediction in output]}, Expected - {y_train.flatten()}",
                end=' ')

    # Evaluate the trained network on XOR data
    predictions = neural_network.propagate_input(X_train)
    print("\nFinal Predictions:")
    for i in range(len(X_train)):
        print(f"Input: {X_train[i]}, Target: {y_train[i]}, Predicted: {predictions[i][0]}")

    # Save the trained model
    neural_network.save_model("xor_model.pkl")
    print("Model saved.")

    # Load the model and make predictions
    loaded_model = NeuralNetwork.load_model("xor_model.pkl")
    loaded_predictions = loaded_model.propagate_input(X_train)
    print("\nLoaded Model Predictions:")
    for i in range(len(X_train)):
        print(f"Input: {X_train[i]}, Target: {y_train[i]}, Predicted: {loaded_predictions[i][0]}")


if __name__ == "__main__":
    main()
