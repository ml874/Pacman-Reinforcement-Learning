import sys
import gym
import pylab
from matplotlib import pylab
from pylab import *
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers


# DQN Agent for the MsPacman
# it uses Neural Network to approximate q function and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.90
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999999
        self.epsilon_min = 0.05
        self.batch_size = 128
        self.train_start = 25000

        # create replay memory using deque
        self.memory = deque(maxlen=1000000)

        # create main model
        self.model = self.build_model()

        if self.load_model:
            self.model.load_weights("./Saved Weights/pacman.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):

        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))

        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0] #STATE
            action.append(mini_batch[i][1])    #ACTION
            reward.append(mini_batch[i][2])    #REWARD
            update_target[i] = mini_batch[i][3]#NEXT STATE
            done.append(mini_batch[i][4])      #DONE

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)



if __name__ == "__main__":

    env = gym.make('MsPacman-ram-v0')
    env = wrappers.Monitor(env, '/tmp/MsPacman-ram-experiment-1',force=True)

    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    # model meta data
    model_name = "first_aws_model"
    weights_path = "./saved-weights/" + model_name
    episodes_per_save = 1000
    thousands_of_episodes = 0
    curr_episode = 1

    while True:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])/256.0
        lives = 3
        while not done:
            dead = False
            while not dead:
                if agent.render:
                    env.render()

                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                score += reward

                dead = info['ale.lives'] != lives
                lives = info['ale.lives']

                reward = reward - 1 if not dead else -100  # if action make Pacman dead, then gives penalty of -100

                next_state = np.reshape(next_state, [1, state_size])/256.0

                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done)

                state = next_state

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./pacman.png")
                # print("episode:", e, "  score:", score, "  memory length:",
                      # len(agent.memory), "  epsilon:", agent.epsilon)

        # every time step do the training
        agent.train_model()

        if curr_episode % 1000 == 0:
            thousands_of_episodes += 1
            print("Completed: " + str(thousands_of_episodes) + " thousand episodes")

        # save the model
        if curr_episode % episodes_per_save == 0:
            curr_episode = 1
            agent.model.save_weights(weights_path + "--" + str(thousands_of_episodes) + "000")

        curr_episode += 1
