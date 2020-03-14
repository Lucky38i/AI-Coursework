from collections import deque

import numpy as np
from skimage import transform
from skimage.color import rgb2gray
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class ReplayBuffer:
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_center = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_center % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_center += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_center, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        next_state = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_state, terminal


def build_dqn(lr, n_actions, input_size):
    model = Sequential(
        [Conv2D(input_shape=input_size, filters=32, kernel_size=8, strides=4, padding='valid',
                kernel_initializer=GlorotNormal, activation='relu'),
         Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', kernel_initializer=GlorotNormal,
                activation='relu'),
         Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', kernel_initializer=GlorotNormal,
                activation='relu'),
         Flatten(),
         Dense(units=512, activation='elu', kernel_initializer=GlorotNormal),
         Dense(units=n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model


def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)
    cropped_frame = gray[31:, 8:-8]

    # Normalize pixel values
    normalized_frame = cropped_frame / 255.0

    preprocessed_frame = transform.resize(normalized_frame, [179, 144])
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((179, 144), dtype=np.int) for i in range(4)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


class Agent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=1000000, model_file='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.eps_dec = epsilon_dec
        self.batch_size = batch_size
        self.model_file = model_file
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_center < self.batch_size:
            return

        states, actions, rewards, next_state, done = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(next_state)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * done

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
