import pygame
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Function to create and compile the PPO network model
def create_ppo_model():
    model = Sequential()
    model.add(Dense(64, input_dim=5, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    return model

# Function to preprocess state for the neural network
def preprocess_state(state):
    return np.array(state).reshape(1, -1)

# Function to choose an action using the PPO policy
def choose_action(model, state):
    action_prob = model.predict(preprocess_state(state))[0]
    action = np.random.choice([0, 1, 2], p=action_prob)
    return action

# Function to train the PPO model
def train_ppo_model(model, states, actions, advantages):
    actions_one_hot = tf.one_hot(actions, depth=3)
    advantages = np.expand_dims(advantages, axis=-1)
    
    model.train_on_batch(states, [actions_one_hot, advantages])

# Initialize PPO model
ppo_model = create_ppo_model()

pygame.init()
screen = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()

# initialize game variables
ball_x, ball_y = 300, 200
ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
paddle1_y, paddle2_y = 150, 150
score1, score2 = 0, 0

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

def calculate_advantages(rewards, gamma):
    advantages = []
    adv = 0
    for r in reversed(rewards):
        adv = adv * gamma + r
        advantages.append(adv)
    advantages.reverse()
    advantages = np.array(advantages)
    advantages -= np.mean(advantages)
    advantages /= np.std(advantages)
    return advantages

# run game loop
training_episodes = 1000
evaluation_episodes = 10
evaluate_after = 500
gamma = 0.95

for episode in range(training_episodes + evaluate_after):
    states, actions, rewards = [], [], []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # get current state
    state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))

    # choose action based on the PPO policy
    action = choose_action(ppo_model, state)

    # update game state based on chosen action
    update_game_state(action)

    # draw game objects
    draw_game_objects()

    # get new state
    new_state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))

    # calculate reward
    reward = score1 - score2

    # store state, action, and reward for training
    states.append(preprocess_state(state))
    actions.append(action)
    rewards.append(reward)

    # limit game to 60 frames per second
    clock.tick(60)

    if episode >= evaluate_after:
        if episode == evaluate_after:
            print("Evaluation:")
        # Train the PPO model
        advantages = calculate_advantages(rewards, gamma)
        train_ppo_model(ppo_model, np.vstack(states), np.array(actions), advantages)

        # Evaluate the learned policy
        total_reward = 0
        for _ in range(evaluation_episodes):
            state = (int(ball_x / 10) - int(paddle1_y / 10), int(ball_y / 10), int(paddle2_y / 10), int(ball_dx / abs(ball_dx)), int(ball_dy / abs(ball_dy)))
            action = choose_action(ppo_model, state)
            update_game_state(action)
            total_reward += score1 - score2

        avg_reward = total_reward / evaluation_episodes
        print(f"Episode {episode + 1}: Average reward over {evaluation_episodes} evaluation episodes: {avg_reward}")

# Save the model after training
ppo_model.save('pong_model_ppo.h5')
print("Model saved after training.")

# Close the Pygame window
pygame.quit()
