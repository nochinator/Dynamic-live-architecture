# Dynamic-Live Neural Networks

Dynamic-Live neural networks are a type of RNN, differing significantly in that the connections between each neuron are learnable. When a neuron is created, it is not initially connected to anything. After creating all the neurons you want, you must call a function for each neuron specifying a list of neurons from which it can receive input. To train the network, call the reinforce function in the network manager script. From the rewards you give, it will train the strengths/weights of the connections.

## IMPORTANT NOTE:

This is the alpha branch! This code is extremly unstable and changes often! If you have problems when using this then we can't help you. This specific readme may be outdated compared to the code in the alpha branch

## How does it work?

Some of the main features of this framework are still a work in progress (WIP). Below is an overview of how the internals currently work.

### Input Propagation

A function similar to this is included in every neural network framework that I can find. The functions below are called on each layer of the network in order. Keep in mind that the order of layers only determines the order neurons are fired, which can help get results out in fewer cycles. It does not determine which neurons can connect where.

#### Priming

This step involves the neuron fetching its inputs from each connected neuron. We can't retrieve inputs when firing neurons, as doing so could cause issues with the data that the neurons receive, depending on the user's network architecture.

#### Firing

Each neuron in a layer undergoes firing, which entails taking the sum of the dot product of the weights stored in the neuron and the inputs obtained during the priming step. For an activation function, we square it; currently, it's not user-selectable, but this will likely change in the future. After every neuron has fired, the output of each neuron in the output layer is collected into an array and returned.

### Training

This is one of the biggest differences from other networks. We currently only support reinforcement, but this could change if someone can figure out how to implement another type of training in this framework.

#### Backpropagation

Unlike traditional reinforcement methods that give the reward to everything and everything makes updates, we use concepts of backpropagation for reinforcement. The process starts with collecting memory data in the background during priming and firing steps, up to a user-defined limit. This data doesn't influence predictions but provides context for training, particularly in complex networks where a neuron's actions might not manifest until several cycles later. When the training function is called on a neuron, it receives reference output. This is used to find what input to make changes based on. We do this to support recurrent network and similar so the neurons can train based on the proper input. The neuron runs your reward score (positive or negative) through a math function that updates the weights/connection strengths of the neuron. The weights are then normalized so they add up to 1. Finaly, for every neuron it takes input from, it calls this same function, distributing the reward score among them and giving them proper reference.

#### Connection Training

This is very much WIP and can change at any time! As of now, the neuron looks at every neuron it can connect to but is not connected to, and, if the reward was negative, has a 1/10 chance to reconnect.

## How do I download this?

Look on the right side panel for the section labeled releases. The item displayed there is the latest full release. As of now, these are beta releases.

## Getting involved

We welcome anyone to get involved in various areas:

- **Research**: Explore our research areas, detailed in the announcement in discussions.
- **Bug Finding**: Testing the system and finding bugs, especially on beta releases, is invaluable. If you find one, please create a new issue.
- **Wiki Writer**: Writing the wiki pages is invaluable. It lets other users know how the framework works and how to use it.
- **Coding**: Contribute by writing code. You can find an issue to work on or check milestones and contribute to the next one.
