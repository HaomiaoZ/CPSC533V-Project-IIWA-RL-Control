import pybullet as p
import time
import pybullet_data
import numpy as np
from iiwa_env import IIWAEnv


env = IIWAEnv()

done = False
while not done:
    state, reward , done = env.step(np.array([1,1,1,1,1,1,1]))

env.close()
