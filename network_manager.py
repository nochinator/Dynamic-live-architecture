from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List
import pickle
import neuron as n
from typing import List, Union
try:
    import pyopencl as cl
    opencl_available = True
except ImportError:
    opencl_available = False
import numpy as np


class NeuralNetwork:
    def __init__(self, input_neurons: List[n.Neuron], hidden_layers: List[List[n.Neuron]],
                 output_neurons: List[n.Neuron]):
        """
        create a neural network using the Neuron class.
        :param input_neurons: expected format: np.array[InputNeuron(), InputNeuron(), etc.]
        :param hidden_layers: expected format: np.array[[Neuron(), Neuron()], [Neuron(), Neuron()], etc.]
        :param output_neurons: expected format: np.array[Neuron(), Neuron(), etc.]
        """
        self.input_neurons = input_neurons
        self.hidden_layers = hidden_layers
        self.output_neurons = output_neurons

    # use open cl if its avalible
    if opencl_available:
        def propagate_input(self, inputs: List[float]) -> List[float]:
            """
            Provide inputs for the entire network and propagate them through the entire network.
            :param inputs: array of shape *number of input neurons*
            :return: network outputs
            """

            outputs = []

            # Set up OpenCL context and command queue
            platform = cl.get_platforms()[0]
            device = platform.get_devices()[0]
            context = cl.Context([device])
            queue = cl.CommandQueue(context)

            # Create buffers for inputs and outputs
            inputs_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=np.array(inputs, dtype=np.float32))
            outputs_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=np.array(outputs, dtype=np.float32).nbytes)

            # Prime input neurons
            for i, neuron in enumerate(self.input_neurons):
                neuron.prime(inputs[i])

            # Fire input neurons
            for neuron in self.input_neurons:
                neuron.fire()

            # Hidden neurons layer by layer
            for layer in self.hidden_layers:
                # Prime neurons in the layer
                for neuron in layer:
                    neuron.prime()

                # Fire neurons in the layer using OpenCL
                program_code = """
                    __kernel void fire(__global float* inputs, __global float* outputs) {
                        int id = get_global_id(0);
                        outputs[id] = tanh(inputs[id]);  // Adjust activation function as needed
                    }
                """
                program = cl.Program(context, program_code).build()
                layer_inputs = np.array([neuron.output for neuron in layer], dtype=np.float32)
                layer_inputs_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                                hostbuf=layer_inputs)
                program.fire(queue, layer_inputs.shape, None, layer_inputs_buffer, outputs_buffer).wait()

                # Update neuron outputs
                outputs = np.empty_like(layer_inputs)
                cl.enqueue_copy(queue, outputs, outputs_buffer).wait()
                for i, neuron in enumerate(layer):
                    neuron.output = outputs[i]

            # Output neurons
            for neuron in self.output_neurons:
                # Prime output neuron
                neuron.prime()

            # Fire output neurons using OpenCL
            program_code = """
                __kernel void fire(__global float* inputs, __global float* outputs) {
                    int id = get_global_id(0);
                    outputs[id] = tanh(inputs[id]);  // Adjust activation function as needed
                }
            """
            program = cl.Program(context, program_code).build()
            output_inputs = np.array([neuron.output for neuron in self.output_neurons], dtype=np.float32)
            output_inputs_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                             hostbuf=output_inputs)
            program.fire(queue, output_inputs.shape, None, output_inputs_buffer, outputs_buffer).wait()

            # Update neuron outputs
            cl.enqueue_copy(queue, outputs, outputs_buffer).wait()
            for i, neuron in enumerate(self.output_neurons):
                neuron.output = outputs[i]
                outputs.append(neuron.output)

    # otherwise use multi-core cpu processing as fallback
    else:
        def propagate_input(self, inputs: List[float]) -> List[float]:
            outputs = []

            with ThreadPoolExecutor() as priming_executor:
                # Prime input neurons using a thread pool
                priming_executor.map(lambda x, y: x.prime(y), self.input_neurons, inputs)

            with ThreadPoolExecutor() as firing_executor:
                # Fire input neurons using a thread pool
                firing_executor.map(lambda x: x.fire(), self.input_neurons)

            for layer in self.hidden_layers:
                with ProcessPoolExecutor() as priming_executor:
                    # Prime neurons in the layer using a process pool
                    priming_executor.map(lambda x: x.prime(), layer)

                with ProcessPoolExecutor() as firing_executor:
                    # Fire neurons in the layer using a process pool
                    firing_executor.map(lambda x: x.fire(), layer)

            with ThreadPoolExecutor() as priming_executor:
                # Prime output neurons using a thread pool
                priming_executor.map(lambda x: x.prime(), self.output_neurons)

            with ThreadPoolExecutor() as firing_executor:
                # Fire output neurons using a thread pool
                firing_executor.map(lambda x: x.fire(), self.output_neurons)
                outputs.extend(neuron.output for neuron in self.output_neurons)

            return outputs

    def reinforce(self, reward: List[float], backpropagations: int) -> None:
        """
        Train the network based on expected input and output
        :param reward: array with rewards for each output separately, values between -1 and 1
        :param backpropagations: How many neurons to backpropogate through, higher values result in better fine-tuning
        but an exponential increase in compute required. Low values on large networks will result in some neurons
        never training
        :return: None
        """
        # train each output neuron with the parameters
        for i, neuron in enumerate(self.output_neurons):
            neuron.train(reward[i], backpropagations)

    def save_model(self, file_path):
        """
        Save the created neural network as a pkl. resulting file contains all data needed to reconstruct the network.
        :param file_path: path to save model to
        :return: None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_path):
        """
        Load a pkl file that contains a neural network. File must contain all data needed to reconstruct the network.
        :param file_path: path to load model from
        :return: None
        """
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
