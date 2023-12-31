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
            self.output = self.activation(np.sum(np.dot(self.inputs, self.synaptic_weights))/len(self.inputs))

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
        self.outputs = []

    def propagate_input(self, inputs):
        """
        Provide inputs for the entire network and propagate them through the entire network.
        :param inputs: array of shape *number of input neurons*
        :return: network outputs
        """
        outputs = []

        # Prime neurons
        print("\ninputs")

        for i, neuron in enumerate(self.input_neurons):
            neuron.prime(inputs[i])
        # Fire input neurons
        for neuron in self.input_neurons:
            neuron.fire()

        # Hidden neurons layer by layer
        for layer in self.hidden_layers:
            print("\nhidden")

            # Prime neurons in the layer
            for neuron in layer:
                neuron.prime()
            # Fire neurons in the layer
            for neuron in layer:
                neuron.fire()

        # Output neurons
        print("\noutputs")

        for neuron in self.output_neurons:
            # Prime output neuron
            neuron.prime()
        for neuron in self.output_neurons:
            # Fire output neuron
            neuron.fire()
            outputs.append(neuron.output)
        self.outputs = outputs
        return outputs

    def reinforce(self, reward, backpropogations, reference_output=None):
        """
        Train the network based on expected input and output
        :param reward: array of shape *number of output neurons, rewards each output separately, values between -1 and 1
        :param backpropogations: How many neurons to backpropogate through, higher values result in better fine-tuning
        but an exponential increase in compute required. Low values on large networks will result in some neurons
        never training
        :param reference_output: Used for training in hindsight. Leaving blank will reward the last predicted output
        :return None
        """
        # defaults
        if reference_output is None:
            reference_output = self.outputs

        # train each output neuron with the parameters
        for i, neuron in enumerate(self.output_neurons):
            neuron.train(reward[i], backpropogations, reference_output[i])

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


def activation(x):
    return x ** 2 - 1


def weight_initialize(x):
    return np.random.uniform(0.4, 0.6, x)


def main():
    # XOR data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # Create a neural network for XOR problem
    input_neurons = np.array(
        [Neuron(1, activation=activation, weight_initialization=weight_initialize, memory_slots=3,
                is_input_neuron=True) for _ in range(2)])

    hidden_layers = np.array([
        [Neuron(2, activation=activation, weight_initialization=weight_initialize, memory_slots=3)
         for _ in range(2)]])

    output_neurons = np.array(
        [Neuron(2, activation=activation, weight_initialization=weight_initialize, memory_slots=3)])

    # Initialize connections
    for neuron in hidden_layers[0]:
        neuron.initialize_connections(input_neurons)
    output_neurons[0].initialize_connections(hidden_layers[0])

    neural_network = NeuralNetwork(
        input_neurons=input_neurons,
        hidden_layers=hidden_layers,
        output_neurons=output_neurons,
        learning_rate=0.01,
    )

    # Train the neural network on XOR data
    epochs = 10000
    for epoch in range(epochs):
        output = []
        for i in range(len(X_train)):
            result = (neural_network.propagate_input(X_train))
            reward = (result[0] - 0.5) * 2
            neural_network.reinforce(reward, 5)
            output.append(result)
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
