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
input_neurons = [new_neuron(memory_slots=3, is_input_neuron=True) for _ in range(20000)]

hidden_neurons = [new_neuron(memory_slots=3, learning_rate=learning_rate) for _ in range(200)]

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
WIDTH, HEIGHT = 200, 100
BALL_RADIUS = 5
PADDLE_WIDTH, PADDLE_HEIGHT = 5, 25
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong AI")

# Paddle and ball initial positions
ai_paddle = pygame.Rect(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)

# Ball movement direction
ball_dx = 5
ball_dy = random.randint(-2, 2)

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

    ai_paddle.y += (movement[0] - 0.5) * 5

    # Ball movement
    ball.x += ball_dx
    ball.y += ball_dy

    # Ball collisions with walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_dy = -ball_dy

    # Ball collision with AI paddle and far wall
    if ball.colliderect(ai_paddle):
        ball_dx = -ball_dx
        # reward

    if ball.right >= WIDTH:
        ball_dx = -ball_dx

    # Ball passes AI paddle
    if ball.left <= 0:
        # Reset ball position
        ball.x = WIDTH // 2 - BALL_RADIUS
        ball.y = HEIGHT // 2 - BALL_RADIUS
        # Send ball towards the far wall
        ball_dx = -ball_dx

    # Drawing
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, ai_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)

    # Cap the frame rate
    clock.tick(FPS)
