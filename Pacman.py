import argparse
import sys
import numpy as np

import gym
from gym import wrappers, logger


EPISODES = 50
ALL_SCORES = np.zeros(EPISODES)

env = gym.make("MsPacman-ram-v0")
env = wrappers.Monitor(env, '/tmp/MsPacman-ram-experiment-1',force=True)





for episode in range(EPISODES):
    env.reset()
    
    reward, info, done = None, None, None

    
    total_score = 0
    while done != True:
        # env.render()
        random_action = env.action_space.sample()
        state, reward, done, info = env.step(random_action)
        print(state)
        total_score += reward
    ALL_SCORES[episode] = total_score
    print("Total Score: {}".format(total_score))
    # print(state, reward, done, info)
    

print(ALL_SCORES)




env.close()