# DQN Agent for the MsPacman
# it uses Neural Network to approximate q function and replay memory & target q network

weight_path = "../AWS_models/modified_rewards_2-weights/modified_rewards_2--4000"
>>>>>>> 62b4eb461cd0f00ac18c14d0abda2bb7772ec144

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
import pickle

if __name__ == "__main__":
    NUM_EPISODES = 1 #1000
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

    agent = TEST_DQNAgent(state_size, action_size)

    i1 = 0
    i2 = 0
    i3 = 0

<<<<<<< HEAD
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
            life = 0
            steps_in_life = 0 # number of steps taken in this life of this episode

            while not dead:
                i3 += 1
                # for every life
                if agent.render:
                    env.render()

                # get action for the current state and go one step in environment
                action = agent.get_action(state)
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

            if done:
                ALL_STEPS[e][life] = steps_in_life
                life += 1
                # at the end of every life
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./pacman.png")
                print("episode:", e, "  score:", score_in_episode)


        if e % 100 == 0 and e > 0:
            print('Average Score for {} Episodes so far: {}'.format(e, np.mean(ALL_SCORES[1:e:1])))

        ALL_SCORES[e] = score_in_episode

    env.close()
    env_orig.close()
    # plt.plot(ALL_SCORES)
    # plt.title("Random Agent: {} Episodes".format(EPISODES))
    # plt.show()

    results = {
        'scores'    :   ALL_SCORES,
        'steps'     :   ALL_STEPS,
        'actions'   :   ALL_ACTIONS
    }

    print(i1)
    print(i2)
    print(i3)

    with open(logfile, 'wb+') as f:
        pickle.dump(results, f)

    print("-------------------------")
    print("Weight file: " + weight_path)
    print('Average Score for {} Episodes: {}'.format(NUM_EPISODES, np.mean(ALL_SCORES)))
