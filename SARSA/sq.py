import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()

# Initialize game variables
ball_x, ball_y = 300, 200
ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
paddle1_y, paddle2_y = 150, 150
score1, score2 = 0, 0

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Q-table initialization
q_table = {}

# Function to choose an action using epsilon-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2])
    else:
        return np.argmax(q_table.get(state, [0, 0, 0]))

def update_q_values(state, action, reward, new_state, new_action):
    global q_table
    # Check if the current state is in the q_table
    if state not in q_table:
        q_table[state] = [random.uniform(0, 1) for _ in range(3)]
    
    # Check if the new state is in the q_table
    if new_state not in q_table:
        q_table[new_state] = [random.uniform(0, 1) for _ in range(3)]

    # Calculate SARSA update
    current_q = q_table[state][action]
    new_q = reward + 0.9 * q_table[new_state][new_action]
    q_table[state][action] = current_q + 0.1 * (new_q - current_q)

# update game state
def update_game_state(action):
    global ball_x, ball_y, ball_dx, ball_dy, paddle1_y, paddle2_y, score1, score2
    if action == 0 and paddle1_y > 0:
        paddle1_y -= 5
    elif action == 2 and paddle1_y < 300:
        paddle1_y += 5
    ball_x += ball_dx
    ball_y += ball_dy
    if ball_y < 0 or ball_y > 390:
        ball_dy *= -1
    if ball_x < 20 and paddle1_y < ball_y < paddle1_y + 100:
        ball_dx *= -1
        score1 += 1
    elif ball_x > 580 and paddle2_y < ball_y < paddle2_y + 100:
        ball_dx *= -1
        score2 += 1
    elif ball_x < 0 or ball_x > 600:
        ball_x, ball_y = 300, 200
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
        score1, score2 = 0, 0
    if ball_y < paddle2_y + 50 and paddle2_y > 0:
        paddle2_y -= 5
    elif ball_y > paddle2_y + 50 and paddle2_y < 300:
        paddle2_y += 5

# draw game objects
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

# Training loop
training_episodes = 50000
evaluate_after = 500
avg_rewards=[]
accuracies = []
correct_actions = 0
total_actions = 0

for episode in range(training_episodes + evaluate_after):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # get current state and choose action
    state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))
    action = choose_action(state)

    # update game state based on chosen action
    update_game_state(action)

    # get new state and choose new action
    new_state = (
        int(ball_x / 10) - int(paddle1_y / 10),
        int(ball_y / 10),
        int(paddle2_y / 10),
        int(ball_dx / abs(ball_dx)),
        int(ball_dy / abs(ball_dy))
    )
    new_action = choose_action(new_state)

    reward = score1 - score2
    update_q_values(state, action, reward, new_state, new_action)

    draw_game_objects()
    
    # limit game to 60 frames per second
    clock.tick(60)
    evaluation_episodes = 10
    if episode >= evaluate_after:
        if episode == evaluate_after:
            print("Evaluation:")
        # Evaluate the learned policy
        total_reward = 0
        for _ in range(evaluation_episodes):
            state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))
            action = choose_action(state)
            update_game_state(action)
            total_reward += score1 - score2

            # Check if the chosen action is correct
            if action == 0 and paddle1_y > 0:
                correct_actions += 1
            elif action == 2 and paddle1_y < 300:
                correct_actions += 1

            total_actions += 1
        
         # Calculate accuracy, ensuring not to divide by zero
        accuracy = correct_actions / total_actions if total_actions > 0 else 0
        accuracies.append(accuracy)    
        avg_reward = total_reward / evaluation_episodes
        
        print(f"Episode {episode + 1}: Average reward over {evaluation_episodes} evaluation episodes: {avg_reward}, Accuracy: {accuracy}")
        avg_rewards.append(avg_reward / (episode + 1))

# Plotting the average reward graph

plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
plt.title('Average Reward over Evaluation Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()

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
