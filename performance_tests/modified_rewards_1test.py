# DQN Agent for the MsPacman
# it uses Neural Network to approximate q function and replay memory & target q network

# weight_path = "../AWS_models/first_aws_model-weights/first_aws_model--6000"
# weight_path = "../AWS_models/modified_rewards_1-weights/modified_rewards_1--4000"

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
            print(weight_path)
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
import os

def main(weight_path, log_path, finished_files):
    split_path = weight_path.split("/")
    model_name = split_path[-1]

    if model_name not in finished_files:
        NUM_EPISODES = 1000
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

        logfile = os.path.join(log_path, "logfile_" + model_name + ".pickle")

        agent = TEST_DQNAgent(state_size, action_size, weight_path)

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
                    # at the end of every life
                    # pylab.plot(episodes, scores, 'b')
                    # pylab.savefig("./pacman.png")
                    print("episode:", e, "\tscore:", score_in_episode, "\tsteps:", steps_in_life)


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

        with open(logfile, 'wb+') as f:
            pickle.dump(results, f)

        print("-------------------------")
        print("Weight file: " + weight_path)
        print('Average Score for {} Episodes: {}'.format(NUM_EPISODES, np.mean(ALL_SCORES)))

        return model_name

    else:
        print("Model {} has already been run. Skipping.".format(model_name))


if __name__=="__main__":
    true_root = "/home/knavejack/Documents/School/2018-2019/CS4701/Pacman-Reinforcement-Learning/"
    models_dir = "AWS_models/"
    test_dir = "first_aws_model_cont"
    finished_path = "finished.txt"

    finished_files = set(open(finished_path, "r").read().split("\n"))

    with open(finished_path, 'a') as f:
        for root, dirs, files in os.walk(os.path.join(true_root, models_dir, test_dir)):
            for file in files:
                if "--" in file:
                    weight_path = os.path.join(root, file)
                    print(weight_path)
                    assert(os.path.exists(weight_path))

                    log_path = os.path.join(true_root, "performance_tests", "data", test_dir)
                    print(log_path)
                    assert(os.path.exists(log_path))
                    finished = main(weight_path, log_path, finished_files)

                    if finished:
                        f.write(finished+"\n")
                        f.flush()
