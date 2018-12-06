# DQN Agent for the MsPacman
# it uses Neural Network to approximate q function and replay memory & target q network

weight_path = "../AWS_models/first_aws_model-weights/first_aws_model--6000"

class TEST_DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = True
        self.load_model = True
        self.epsilon = 0 # no random moves 

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # create main model
        self.model = self.build_model()

        if self.load_model:
            self.model.load_weights(weight_path)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation=None, kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam())
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])
#         q_value = self.model.predict(state)
#         return np.argmax(q_value[0])


import sys
import gym
from matplotlib import pylab
from pylab import *
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers
import matplotlib.pyplot as plt


if __name__ == "__main__":
    EPISODES = 1000
    ALL_SCORES = np.zeros(EPISODES)

    env = gym.make('MsPacman-ram-v0')
    env = wrappers.Monitor(env, '/tmp/MsPacman-ram-experiment-1',force=True)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = TEST_DQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        with open('../Saved Scores/performance_test0.txt', 'a+') as file:
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            lives = 3
            while not done:
                dead = False
                while not dead:
                    if agent.render:
                        env.render()

                    # get action for the current state and go one step in environment
                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])

                    state = next_state
                    score += reward
                    dead = info['ale.lives']<lives
                    lives = info['ale.lives']
                    # if an action make the Pacman dead, then gives penalty of -100
                    reward = reward if not dead else -500

                if done:
                    scores.append(score)
                    episodes.append(e)
                    # pylab.plot(episodes, scores, 'b')
                    # pylab.savefig("./pacman.png")
                    print("episode:", e, "  score:", score)
                    file.write("episode:" + str(e) + "  score:" + str(score) + "\n")


            if e % 100 == 0 and e > 0:
                print('Average Score for {} Episodes so far: {}'.format(e, np.mean(ALL_SCORES[1:e:1])))

        ALL_SCORES[e] = score

    env.close()
    # plt.plot(ALL_SCORES)
    # plt.title("Random Agent: {} Episodes".format(EPISODES))
    # plt.show()

    print("-------------------------")
    print("Weight file: " + weight_path)
    print('Average Score for {} Episodes: {}'.format(EPISODES, np.mean(ALL_SCORES)))
