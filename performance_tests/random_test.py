import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

import gym
from gym import wrappers, logger


EPISODES = 500
ALL_SCORES = np.zeros(EPISODES)

env = gym.make("MsPacman-ram-v0")
env = wrappers.Monitor(env, '/tmp/MsPacman-ram-experiment-1',force=True)

for episode in range(EPISODES):
    env.reset()

    reward, info, done = None, None, None

    total_score = 0
    while done != True:
        env.render()
        random_action = env.action_space.sample()
        time.sleep(0.01)
        state, reward, done, info = env.step(random_action)
        total_score += reward
    ALL_SCORES[episode] = total_score
    print("Total Score: {}".format(total_score))
    # print(state, reward, done, info)

env.close()
plt.plot(ALL_SCORES)
plt.title("Random Agent: {} Episodes".format(EPISODES))
plt.show()

print("-------------------------")
print('Average Score for {} Episodes: {}'.format(EPISODES, np.mean(ALL_SCORES)))
