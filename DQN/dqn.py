import pygame
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Function to create and compile the Q-network model
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=5, activation='relu'))
    model.add(Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# Function to preprocess state for the neural network
def preprocess_state(state):
    return np.array(state).reshape(1, -1)

# Initialize Q-network model
model = create_model()

pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()

# initialize Q table
q_table = {}
for i in range(-10, 11):
    for j in range(-10, 11):
        for k in range(-10, 11):
            for l in range(-10, 11):
                q_table[(i, j, k, l)] = [random.uniform(0, 1) for _ in range(3)]

# initialize game variables
ball_x, ball_y = 300, 200
ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
paddle1_y, paddle2_y = 150, 150
score1, score2 = 0, 0

# Function to preprocess state for the neural network
def preprocess_state(state):
    return np.array(state).reshape(1, -1)

# Function to create the Q-network model
def create_q_model():
    model = Sequential()
    model.add(Dense(64, input_dim=5, activation='relu'))
    model.add(Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# Train the Q-network model
def train_q_network(model, state, target):
    model.fit(preprocess_state(state), np.array([target]), epochs=1, verbose=0)

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

# run game loop
training_episodes = 50000
evaluation_episodes = 10
evaluate_after = 500

# DQN parameters
epsilon = 0.1
gamma = 0.9
learning_rate = 0.1

# Create the Q-network model
q_network_model = create_q_model()

for episode in range(training_episodes + evaluation_episodes):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # get current state and Q-values
    state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))
    q_values = q_table.get(state)
    if q_values is None:
        # initialize new state with random Q-values
        q_table[state] = [random.uniform(0, 1) for _ in range(3)]

    # choose action with epsilon-greedy policy
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1, 2])
    else:
        action = np.argmax(q_values)

    # update game state based on chosen action
    update_game_state(action)

    # draw game objects
    draw_game_objects()

    # get new state and update Q-value
    new_state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))
    new_q_values = q_table.get(new_state, [random.uniform(0, 1) for _ in range(3)])

    reward = score1 - score2
    target = reward + gamma * np.max(new_q_values)

    q_table[state][action] += learning_rate * (target - q_table[state][action])

    # Train the Q-network model
    train_q_network(q_network_model, state, target)

    # limit game to 60 frames per second
    clock.tick(60)

    if episode >= evaluate_after:
        if episode == evaluate_after:
            print("Evaluation:")
        # Evaluate the learned policy
        total_reward = 0
        for _ in range(evaluation_episodes):
            state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))
            q_values = q_table.get(state)
            if q_values is None:
                q_table[state] = [random.uniform(0, 1) for _ in range(3)]
            action = np.argmax(q_values)
            update_game_state(action)
            total_reward += score1 - score2

        avg_reward = total_reward / evaluation_episodes
        print(f"Episode {episode + 1}: Average reward over {evaluation_episodes} evaluation episodes: {avg_reward}")

# Save the model after training
q_network_model.save('pong_model_dqn.h5')
print("Model saved after training.")

# Close the Pygame window
pygame.quit()
