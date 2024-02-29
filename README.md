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
This used to be two functions, but now it's only one.
First, the input for the network is given to the input neurons.
Input neurons don't do any processing.
This is simply how data gets into to the network.
Each neuron in the network will receive a list of the outputs of every neuron in the network.
We multiply the outputs of the network by the weights stored in the neuron.
We do not use an activation function as non-linearity is introduced in the structure of the network.
After every neuron has fired, the output of each neuron in the output layer is collected into an array and returned.

### Training

This is one of the biggest differences from other networks. 
We currently only support reinforcement, but this could change if someone can figure out how to implement gradient 
decent training in this framework.

#### Hebbian Learning

To explain the learning mechanism, you must know what hebbian learning is, this can be described as "neurons that 
fire together, wire together."
It can be represented wit this equation: `weight += learning rate * output of self * output of connected neuron`
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
positioning system where only neurons that are nearby can connect.

#### Moving Neurons Around

Now that neurons can only connect to nearby neurons, it only makes sense that neurons can move around to form complex 
structures and adapt the structure of the network. 
To do this, we get the direction of every connection, and both of the neurons connected are pulled together. 
This is done for every neuron after normalizing weights.

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
To help solve this, we keep a memory log of the output of every neuron.
But we still don't know what output/input to refer to, so we take an average of the past x predictions starting at y, 
where x and y are hyperparameters. 
This will result in slower training, but at least we can train it towards a desired output.

##### What about the backpropagation system?

The backpropagation system from the beta branch doesn't work with the new position system, at least not yet.
The entire thing was based on the idea that users defined which neurons could and could not be connected to, but the 
positioning system doesn't do this. 
Even if we did make it work with the position system,
it would be extremely slow due to the recurrent nature of the network.
If you can come up with a way to do this efficiently, then please open an issue explaining it.

## Why is this useful/important?

While it has not been tested yet, this could offer some unique advantages over traditional feedforward networks.
One of the most obvious advantages is capturing both spacial and temporal relationships. 
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
