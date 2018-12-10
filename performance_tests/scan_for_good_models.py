# DQN Agent for the MsPacman
# it uses Neural Network to approximate q function and replay memory & target q network

# weight_path = "../AWS_models/first_aws_model-weights/first_aws_model--7000"

class TEST_DQNAgent:
    def __init__(self, state_size, action_size, weight_path):
        # if you want to see MsPacman learning, then change to True
        self.render = False
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
import pickle
import time

if __name__ == "__main__":

    weight_path_prefix = "../AWS_models/first_aws_model-cont/first_aws_modelcont--"
    curr_episode = 148000
    episode_step = 1000
    episode_end = 

    while curr_episode <= episode_end:

        NUM_EPISODES = 20
        NUM_ACTIONS = 9
        NUM_LIVES = 3

        ALL_SCORES = np.zeros(NUM_EPISODES) # track scores for every episode
        ALL_STEPS = np.zeros((NUM_EPISODES, NUM_LIVES)) # track steps for every episode
        ALL_ACTIONS = np.zeros((NUM_EPISODES, NUM_ACTIONS)) # stores aggregate actions

        env_orig = gym.make('MsPacman-ram-v0')
        env = wrappers.Monitor(env_orig, '/tmp/MsPacman-ram-experiment-1',force=True)
        # get size of state and action from environment
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        curr_weight_path = weight_path_prefix + str(curr_episode)
        print(curr_weight_path)
        agent = TEST_DQNAgent(state_size, action_size, curr_weight_path)

        i1 = 0
        i2 = 0
        i3 = 0

        for e in range(NUM_EPISODES):
            i1 += 1
            done = False
            score_in_episode = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            num_lives = 3
            while not done:
                i2 += 1
                # for every episode
                dead = False
                steps_in_life = 0 # number of steps taken in this life of this episode

                while not dead:
                    i3 += 1
                    # for every life
                    if agent.render:
                        env.render()

                    # get action for the current state and go one step in environment
                    action = agent.get_action(state)

                    # time.sleep(0.02)

                    ALL_ACTIONS[e][action] += 1
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])

                    state = next_state
                    score_in_episode += reward
                    steps_in_life += 1
                    dead = info['ale.lives']!=num_lives
                    num_lives = info['ale.lives']
                    lives = info['ale.lives']
                    # if an action make the Pacman dead, then gives penalty of -100

                ALL_STEPS[e][NUM_LIVES - num_lives - 1] = steps_in_life


                if done:
                    ALL_SCORES[e] = score_in_episode
                    # at the end of every life
                    # pylab.plot(episodes, scores, 'b')
                    # pylab.savefig("./pacman.png")
                    # print("episode:", e, "  score:", score_in_episode, "steps:", steps_in_life)

        with open("./scan.txt", "a") as logfile:
            logfile.write(curr_weight_path + "\n")
            logfile.write("\t\tAverage score: " + str(np.mean(ALL_SCORES)) + "\n")
            logfile.write("\t\tMaximum score: " + str(np.amax(ALL_SCORES)) + "\n")
            if np.mean(ALL_SCORES) > 700 or np.amax(ALL_SCORES) > 1500:
                logfile.write("\t\t^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

        curr_episode += episode_step
