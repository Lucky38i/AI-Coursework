from DQN_Agent_Images import Agent, stack_frames
import numpy as np
import gym
import tensorflow as tf
import os
from collections import deque

# File Location
model_file = "data/models/dqn_Breakout_Modified_model.h5"

# Model Hyperparameters
state_size = [179, 144, 4]

# Preprocessing Hyperparamters
stack_size = 4

# Training Hyperparameters
lr = 0.00001
n_games = 500
gamma = 0.99
epsilon = 1
decay = 1e-5
epsilon_end = 0.01
batch_size = 64


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('Breakout-v0')
    stacked_frames = deque([np.zeros((179, 144), dtype=np.int) for i in range(4)], maxlen=4)
    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=state_size,
                  n_actions=env.action_space.n, mem_size=1000, batch_size=batch_size, epsilon_end=epsilon_end,
                  model_file=model_file, epsilon_dec=decay)

    if os.path.isfile(model_file):
        print("Loading existing Model")
        agent.load_model()

    scores = []
    eps_history = []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        observation, stacked_frame = stack_frames(stacked_frames, observation, True)
        while not done:
            # env.render()
            # Training developed from https://www.youtube.com/watch?v=SMZfgeHFFcA
            # with modifications to for frame stacking
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            next_observation, stacked_frame = stack_frames(stacked_frames, next_observation, False)
            score += reward
            agent.store_transition(observation, action, reward, next_observation, done)
            observation = next_observation
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score,
              'average_score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

        if i == 50:    # Saves after 50 episodes
            agent.save_model()
