import neuron
import network_manager
import numpy as np
import pygame
import sys
import random
from pygame.locals import *


new_neuron = neuron.Neuron
learning_rate = 0.1  # create dynamic adjustment system

# Create network
input_neurons = [new_neuron(memory_slots=3, is_input_neuron=True) for _ in range(10)]

hidden_neurons = [new_neuron(memory_slots=3, learning_rate=learning_rate) for _ in range(50)]

output_neurons = [new_neuron(memory_slots=3, learning_rate=learning_rate)]

# Initialize connections
front_neurons = [input_neurons[0]] + hidden_neurons
for neuron in hidden_neurons:
    neuron.initialize_connections(front_neurons)

output_neurons[0].initialize_connections(hidden_neurons)

# create network
nn = network_manager.NeuralNetwork(input_neurons, hidden_neurons, output_neurons)


# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1, 10
FPS = 60
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)

# Initialize game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong AI")

# dot position. should start centered
dot = pygame.Rect(1, 1, 1, 1)
taget = pygame

# Game loop
clock = pygame.time.Clock()
while True:
    pygame.display.flip()
    # Capture the current frame
    frame = pygame.surfarray.array3d(pygame.display.get_surface())

    # Convert the frame to black and white
    bw_frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

    # Flatten the black and white image into a 1D array
    flattened_input = bw_frame.flatten()

    # Pass the flattened array as input to the neural network
    movement = nn.propagate_input(flattened_input)

    for event in pygame.event.get():
        if event.type == QUIT:
            nn.save_model("model")
            pygame.quit()
            sys.exit()

    # Drawing
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, ai_paddle)

    # Cap the frame rate
    clock.tick(FPS)
