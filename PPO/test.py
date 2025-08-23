import pygame
import random
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('pong_model_ppo.h5')

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()

# Initialize game variables
ball_x, ball_y = 300, 200
ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
paddle1_y, paddle2_y = 150, 150
score1, score2 = 0, 0

# Function to update game state
def update_game_state(action):
    global ball_x, ball_y, ball_dx, ball_dy, paddle1_y, paddle2_y, score1, score2
    # Update paddle1_y based on action
    if action == 0 and paddle1_y > 0:
        paddle1_y -= 5
    elif action == 2 and paddle1_y < 300:
        paddle1_y += 5
    # Update ball position
    ball_x += ball_dx
    ball_y += ball_dy
    # Ball collision with top and bottom walls
    if ball_y < 0 or ball_y > 390:
        ball_dy *= -1
    # Ball collision with paddles
    if ball_x < 20 and paddle1_y < ball_y < paddle1_y + 100:
        ball_dx *= -1
        score1 += 1
    elif ball_x > 580 and paddle2_y < ball_y < paddle2_y + 100:
        ball_dx *= -1
        score2 += 1
    # Ball out of bounds
    elif ball_x < 0 or ball_x > 600:
        ball_x, ball_y = 300, 200
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
        score1, score2 = 0, 0
    # Update paddle2_y based on ball position
    if ball_y < paddle2_y + 50 and paddle2_y > 0:
        paddle2_y -= 5
    elif ball_y > paddle2_y + 50 and paddle2_y < 300:
        paddle2_y += 5

# Function to draw game objects
def draw_game_objects():
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, paddle1_y, 10, 100))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(590, paddle2_y, 10, 100))
    pygame.draw.circle(screen, (255, 255, 255), (int(ball_x), int(ball_y)), 10)
    pygame.draw.line(screen, (255, 255, 255), (300, 0), (300, 400))
    font = pygame.font.SysFont(None, 30)
    score_text = font.render(str(score1) + " - " + str(score2), True, (255, 255, 255))
    screen.blit(score_text, (260, 10))
    pygame.display.flip()

# Initialize game parameters
evaluation_episodes = 50000
total_reward = 0
avg_rewards = []
correct_actions = 0
total_actions = 0
accuracies = []

# Evaluation loop
for episode in range(evaluation_episodes):
    print(episode)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Get current state and choose action using the trained model
    state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))
    action = np.argmax(model.predict(np.array(state).reshape(1, -1))[0])

    # Update game state based on chosen action
    update_game_state(action)

    # Calculate total reward
    total_reward += score1 - score2

    # Calculate accuracy
    if action == 0 and paddle1_y > 0:
        correct_actions += 1
    elif action == 2 and paddle1_y < 300:
        correct_actions += 1

    total_actions += 1

    # Draw game objects
    draw_game_objects()

    # Append the average reward for this episode
    avg_rewards.append(total_reward / (episode + 1))

    # Append the accuracy for this episode
    accuracy = correct_actions / total_actions if total_actions > 0 else 0
    accuracies.append(accuracy)

# Calculate average reward over all evaluation episodes
avg_reward = total_reward / evaluation_episodes
print(f"Average reward over {evaluation_episodes} evaluation episodes: {avg_reward}")

# Print average accuracy
avg_accuracy = correct_actions / total_actions if total_actions > 0 else 0
print(f"Average accuracy over {evaluation_episodes} evaluation episodes: {avg_accuracy}")

# Plotting the average reward graph
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
plt.title('Average Reward over Evaluation Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Reward')

# Plotting the accuracy graph
plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracies) + 1), accuracies, color='orange')
plt.title('Accuracy over Evaluation Episodes')
plt.xlabel('Episode')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Close the Pygame window
pygame.quit()
