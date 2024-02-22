# Dynamic-Live Neural Networks

Dynamic-Live neural networks are a type of RNN, differing significantly in that the connections between each neuron are learnable.
It works on a positional system.
Every neuron has a position, and it cannot connect to neurons that are not nearby.
As the weights change, the neurons will move around and organize themselves.

## IMPORTANT NOTE:

This is the ALPHA branch. The code will be very buggy, and likely won't even work properly.

## How does it work?

Some of the main features of this framework are still a work in progress (WIP). 
Below is an overview of how the internals currently work.

### Input Propagation

A function similar to this is included in every neural network framework that I can find. 
The two functions below are called on every neuron, first priming, then firing.

#### Priming

This step involves the neuron fetching its inputs from each connected neuron.
We can't retrieve inputs when firing neurons, 
as doing so could cause issues with the data that the neurons receive, depending on the network structure.

#### Firing

Each neuron in the network undergoes firing, which entails taking the dot product of the weights stored in the 
neuron and the inputs obtained during the priming step.
We do not use an activation function as non-linearity is introduced in the structure of the network.
After every neuron has fired, the output of each neuron in the output layer is collected into an array and returned.

### Training

This is one of the biggest differences from other networks. 
We currently only support reinforcement, but this could change if someone can figure out how to implement gradient 
decent training in this framework.

#### Meet the neurons

To understand how training works, you need to know the three types of neurons: Input, Hidden, and Output.

##### Input Neurons

Input neurons take a single value in, then set that value as the output. 
They serve as a compatibility layer, giving each input a position in the network.
They do not move around.

##### Hidden Neurons

The Main part of the network. 
These neurons will move around as they train, and perform dot product-ing to produce the output.
No activation function is used as non-linearity comes from the non-linear structure of the network.

##### Anchor Neurons

Works the same as the hidden neurons; however, it will not move itself, it will only move other neurons.
This gives the hidden neurons a base, then they build around the anchor neurons.
Anchor neurons are often used for the outputs, and, when implemented, will be used in the feed-forward portion of 
the network.

#### Hebbian Learning

To explain the learning mechanism, you must know what hebbian learning is, this can be described as "neurons that 
fire together, wire together."
A learning rate value is also applied.
The following equation is run for every connection.
This method is inspired by real, biological neurons.

#### Directing Training

Hebbian learning alone has no direction, it will only learn patterns in the data and will only give nonsensical 
output.
To fix this, we provide a reward value so the equation becomes `weight += learning rate * output of self * output of connected neuron * reward`
The reward value can be positive or negative, allowing for the strength of connections to strengthen or weaken based on the quality of its actions.

#### Normalizing Values

After the weights are adjusted, we normalize them so that all the weights add up to 1. 
This prevents massive or tiny weights and improves stability.

#### Connecting Neurons

So now we can train towards a desired value, but how do we know if two neurons should be connected or not? 
Well, we could just say that every neuron is connected, but then we would form too many connections, so we have a 
positioning system and only neurons that are nearby can connect.

#### Moving Neurons Around

Now that neurons can only connect to nearby neurons, it only makes sense that neurons can move around to form complex 
structures and adapt the structure of the network. 
To do this, we get the direction of every connection, and both of the neurons connected are pulled together. 
This is done for every neuron so after normalizing weights.

#### Addressing the delayed reaction problems

When a neuron gives an output, then it may not show until several cycles later. 
This is caused by the fact that there is no clear path through the network, and as such, input doesn't have an 
instant reaction.
This causes two problems. 

The first is that the network has a reaction delay.
It can't react to an input for several cycles.
This problem can be solved by having a feed forward network embedded in it with anchor neurons.
This embedded network will fire using the traditional method before the rest of the network fires.

The second problem is a lot harder to solve.
When we go to train the network, we don't know what input and output data to refer to.
To help solve this, we keep a memory log of all the inputs the neuron receives and all the outputs it gives.
But we still don't know what output/input to refer to, so we take an average of the past x predictions. 
This will result in slower training, but at least we can train it.

##### What about the backpropagation system?

The backpropagation system from the beta branch doesn't work with the new position system, at least not yet.
The entire thing was based on the idea that users defined which neurons could and could not be connected to, but the 
positioning system doesn't do this. 
We could do it based on which neurons are nearby, but that would be excessively slow due to all the recurrences.
If you can come up with a way to do this, then please open an issue explaining it.

## Why is this useful/important?

While it has not been tested yet, this could offer some unique advantages over traditional feedforward networks.
One of the most obvious advantages are capturing both spacial and temporal relationships. 
This is a result of the reaction delay problems, and the fact that data is represented in a simulated space.
Another more speculative advantage is thinking in a more human way, allowing better interaction with humans.

## How do I download this?

Look on the right side panel for the section labeled releases. 
The item displayed there is the latest full release. 
As of now, they are beta releases.

## Getting involved

We welcome anyone to get involved in various areas:

- **Research**: Explore our research areas, detailed in the announcement in discussions.
- **Bug Finding**: Testing the system and finding bugs, especially on beta releases, is invaluable. 
If you find one, please create a new issue.
- **Wiki Writer**: Writing the wiki pages is invaluable. 
- It lets other users know how the framework works and how to use it.
- **Coding**: Contribute by writing code. 
You can find an issue to work on or check milestones and contribute to the next one.

## License
You may use this work for any non-commercial purpose. 
If you make any kind of spin off, then you must use the same license and give credit. 
I will not allow commercial use as of now. 
This may change in the future with specific permission.
Specific licensing details are below.

Dynamic Live Architecture © 2023 by nochinator is licensed under CC BY-NC-SA 4.0 
