# Dynamic-Live Neural Networks

## IMPORTANT NOTE:
The code in release and beta are no updated, the code in alpha is in development

Dynamic-Live neural networks are a type of RNN, differing significantly in that the connections between each neuron are learnable. When a neuron is created, it is not initially connected to anything. After creating all the neurons we need, a function is called for each neuron, specifying a list of neurons from which it can receive input. The network then autonomously learns to form effective connections between every neuron.

## How does it work?

The main feature of this network is still in progress, with a focus on creating basic functionalities, including training functions. Below is an overview of the current training process.

### Input Propagation

This function is fundamental for making predictions in this network. While most neural networks implement this in some form, ours is encapsulated in a single function within our library.

#### Priming

This step involves the neuron fetching its inputs from each connected neuron. We can't retrieve inputs when firing neurons, as doing so could cause issues with the data that the neurons receive, depending on the firing order.

#### Firing

Each neuron in a layer undergoes firing, which entails taking the dot product of the weights stored in the neuron and the inputs obtained during the priming step. The output is determined by the user's chosen activation function (custom activations are supported). The process is repeated for every layer in sequence. It's important to note that layering is only used to dictate the firing order of neuron groups and is unrelated to the connections between neurons. After every neuron has fired, the output of each neuron in the output layer is collected into an array and returned.

### Backpropagation

Unlike traditional methods that calculate loss and changes for each layer, we collect memory data in the background during priming and firing steps, up to a user-defined limit. This data doesn't influence predictions but provides context for training, particularly in complex networks where a neuron's actions might not manifest until several cycles later. During training, the output over all output is fed into each neuron's training function as a parameter. The neuron examines the contextual output's index, compares it to the same index in another memory bank for internal values, and uses them in loss calculation (custom loss functions are in progress). Subsequently, we examine the derivative of the activation function and calculate gradients for updating weights. After a neuron makes these changes, it reviews every neuron from which it receives input, identifies the output it received, and passes the signal up the network. This process occurs for every neuron, potentially being computationally expensive for large networks. While this may seem complex, it theoretically allows for powerful memory-focused training.

### Connection Training

This function is a work in progress. Currently, we build networks manually as they can't train connections. Ongoing research is exploring how this will work. Keep in mind that this is a novel concept, and as far as I can find, it has not been approached in this way. If you have concepts, please open an issue and explain your idea.

## How do I download this?

Currently, we don't have a release package. When available, this text will link to the releases. If you wish to use the current code, download the file and include it in your Python project. You can then import it like any other package, using the file name as the import name.

## Getting involved

We welcome anyone to get involved in various areas:

- **Research:** Explore our research areas, detailed in the announcement in discussions.
- **Bug Finding:** Testing the system and finding bugs, especially on beta releases, is invaluable. If you find one, please create a new issue.
- **Coding:** Contribute by writing code. You can find an issue to work on or check milestones and contribute to the next one.
