import numpy as np
import random
import math
import gym
import os

# parameters
stateSize = 16
actionSize = 4
gamma = 0.9
episodes = 40000


class QLearningAgent:
    def __init__(self):
        self.Q = np.full((stateSize, actionSize), 0.0)
        self.time = 0
        self.a = 0

    def alpha(self):
        return max(0.1, min(1.0, 1.0 - math.log10(self.time / 25.0)))

    def epsilon(self):
        return max(0.1, min(1.0, 1.0 - math.log10(self.time / 25.0)))

    def action(self, state):
        self.time += 1
        if random.random() < self.epsilon():
            self.a = random.randrange(0, actionSize)
        else:
            self.a = np.argmax(self.Q[state, :])
        return self.a

    def reward(self, s, r, ps):
        self.Q[s, self.a] = (self.Q[s, self.a] + self.alpha() * (r + gamma *
                                                                 np.max(self.Q[ps, :])
                                                                 - self.Q[s, self.a]))


env = gym.make('FrozenLake-v0')
ql = QLearningAgent()
wins = 0
for i_episode in range(episodes):
    state = env.reset()
    for t in range(200):
        newState, reward, done, info = env.step(ql.action(state))
        ql.reward(state, reward, newState)
        state = newState
        if done:
            if reward > 0:
                wins += 1
            break
env.close()
print(wins, "wins in", episodes, "episodes")
