import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import wrappers, logger


EPISODES = 1000
ALL_SCORES = np.zeros(EPISODES)

env = gym.make("MsPacman-ram-v0")
env = wrappers.Monitor(env, '/tmp/MsPacman-ram-experiment-1',force=True)

for episode in range(EPISODES):
    with open('../Saved Scores/performance_testrandom.txt', 'a+') as file:

        env.reset()

        reward, info, done = None, None, None


        total_score = 0
        while done != True:
            # env.render() # call if want to watch gui window
            random_action = env.action_space.sample()
            state, reward, done, info = env.step(random_action)
            total_score += reward
        ALL_SCORES[episode] = total_score
        print("Total Score: {}".format(total_score))
        file.write("episode:" + str(episode) + "  score:" + str(total_score) + "\n")
        # print(state, reward, done, info)

env.close()
# plt.plot(ALL_SCORES)
# plt.title("Random Agent: {} Episodes".format(EPISODES))
# plt.show()

print("-------------------------")
print("RANDOM AGENT PERFORMANCE BASELINE")
print('Average Score for {} Episodes: {}'.format(EPISODES, np.mean(ALL_SCORES)))
