import argparse
import sys

import gym
from gym import wrappers, logger


env = gym.make("MsPacman-ram-v0")

state = env.reset()
reward, info, done = None, None, None
while done != True:
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    print(state, reward, done, info)