import sys
import gym
import pylab
from matplotlib import pylab
from pylab import *
import random
import numpy as np
import keras
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers
import tensorflow as tf
import time



def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

# ENV_NAME = "CartPole-v1"
# ENV_NAME= 'MountainCar-v0'
# ENV_NAME = "CartPole-v1"
ENV_NAME = 'MsPacman-ram-v0'

class DQNAgent:
    
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0025
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99999
        self.batch_size = 32
        self.train_start = 50000  

        # create replay memory using deque
        self.memory = deque(maxlen=750000)

        # create main model
        self.model = self.build_model()
        
        # keep track of # of frames trained on
        self.trained_frames = 0
        
        # keep track of # of frames trained on
        self.million_frames = 0
        
        # keep track of average q-value predictions
        self.q_val_predictions = []
    
    # approximate Q function using Neural Network: state is input and Q Value of each action is output
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_size,), activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
#         model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model
    
    
    # save sample <s,a,r,s'> to the replay memory
    def save_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
#         if self.epsilon > self.epsilon_min:
#             self.epsilon -= 1/1000000.0
#             self.epsilon *= self.epsilon_decay


    def get_action(self, state):
         if np.random.rand() <= self.epsilon:
            random_action = np.random.choice(self.action_size)
            return random_action
         else:
            q_values = self.model.predict(state)
            best_action = np.argmax(q_values[0])
            return best_action #, np.amax(q_values[0])
    
        
    def experience_replay(self, target_model):
        if len(self.memory) < self.train_start:
            return
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch)
        
        curr_state = np.stack(batch[:, 0], axis=1)[0] #STATE
        next_state = np.stack(batch[:, 3], axis=1)[0] #NEXT STATE
        action = batch[:, 1]                           #ACTION
        reward = batch[:, 2]                   #REWARD
        done = batch[:, 4]  #DEAD
        
        curr_state_Q_target = self.model.predict(curr_state)
        next_state_Q_target = target_model.predict(next_state)
        max_val_next_state_Q_target = np.amax(next_state_Q_target, axis=1)
        self.q_val_predictions.append(np.mean(max_val_next_state_Q_target))

#         print(next_state_Q_target)
#         print(max_val_next_state_Q_target)
        
        
        # VECTORIZE BELOW FOR LOOP ATTEMPT
#         done_ix = np.where(done == True)
#         not_done_ix = np.where(done == False)
        
#         done_reward = np.take(reward, done_ix)
#         not_done_reward = np.take(reward, not_done_ix) + self.discount_factor * np.take(max_val_next_state_Q_target, not_done_ix)
        
#         done_actions = np.take(actions, done_ix)
#         not_done_actions  np.take(actions, not_done_ix)
# #         print(done_reward, not_done_reward.shape)

#         np.put(a, [0, 2], [-44, -55])


        for i in range(self.batch_size):
            if done[i]:
                curr_state_Q_target[i][action[i]] = reward[i]
            else:
                curr_state_Q_target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(next_state_Q_target[i]))


        self.model.fit(curr_state, curr_state_Q_target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.trained_frames += self.batch_size
        self.million_frames += self.batch_size
        if self.epsilon > self.epsilon_min:
            self.epsilon -= 2/1000000.0


def main(plot_scores=True):

    # model meta data
    model_name = "second_aws_model"
    weights_path = "./saved-weights/" + model_name
    episodes_per_save = 1000
    thousands_of_episodes = 0
    curr_episode = 1

    print("Running first episode")


    episodes = []
    scores = []
    pos = [] # Mountaincar
    
    ma_scores = []
    
    
    env = gym.make(ENV_NAME)
#     env = wrappers.Monitor('/tmp/cartpole-experiment-0', force=True)

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = DQNAgent(observation_space, action_space)
    # Intiailize Target Model
    target_model = keras.models.clone_model(agent.model)
    target_model.set_weights(agent.model.get_weights())
    
    FRAMES = 0
    num_episodes = 0
    
#     print('HI')
# 
#     print(agent.trained_frames)
    while True:
        
        episodes.append(num_episodes)
        num_episodes += 1
        
        state = env.reset()
        state = np.reshape(state, [1, observation_space])/255.0
        score = 0
        lives = 3
        done = False
#         print('HI')

        # while game is not done
        while not done:
#             print('HI')

            dead = False
            while not dead:
                if agent.render:
                    env.render()
                action = agent.get_action(state)
                state_next, reward, done, info = env.step(action)
                score += reward
    #             print(state_next)
    #             reward = reward if not done else -reward

                # Check if dead: if current lives != lives
                dead = info['ale.lives'] != lives
                lives = info['ale.lives']


                if dead:
                    reward = -1.0                
                elif reward == 10.0:
                    reward = 0.75
                elif reward >= 100.0:
                    reward == 1.0
                else:
                    reward = -0.02                

    #             print(done, reward)
                state_next = np.reshape(state_next, [1, observation_space])/255.0
                agent.save_memory(state, action, reward, state_next, done)
                state = state_next
                if done:
                    print("Episode: {}, Frames Seen: {}, Frames Trained: {}, Score: {}, Memory Length: {}, Epsilon: {}".format(
                        num_episodes, FRAMES, agent.trained_frames, score, len(agent.memory), agent.epsilon))
                    pos.append(state_next[0][0])
    #                 print(pos)

                    break
    #                 score_logger.add_score(step, run)
                if agent.million_frames > 100000:
                    print('Update Target Model')
                    agent.million_frames = 0

                    """Returns a copy of a keras model."""
                    target_model = keras.models.clone_model(agent.model)
                    target_model.set_weights(agent.model.get_weights())


                agent.experience_replay(target_model)

                FRAMES += 1
    #         print(max_right)

        scores.append(score)

        if curr_episode % 50 == 0:
            print("Completed: " + str(curr_episode) + " episodes")
            sys.stdout.flush()

        # save the model
        if curr_episode % episodes_per_save == 0:
            open(model_name + "--" + str(curr_episode), 'a').close() # create file
            agent.model.save_weights(model_name  + "--" + str(curr_episode))
            print("saved weights successfully")

        curr_episode += 1

        


    #         break
        
        
        
        
        # SAVE MODEL
#         if num_episodes % 100 == 0:
#             timestr = time.strftime("%Y%m%d-%H%M%S")
#             agent.model.save_weights("./Saved Weights/Pacmanv3_{}.h5".format(timestr))
            
            
if __name__ == "__main__":
    main(plot_scores=False)