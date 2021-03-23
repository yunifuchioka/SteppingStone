# temporary file used to debug env_locomotion_2d environment

import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)

from common.envs_utils import make_env

env = make_env("mocca_envs:Walker2DCustomEnv-v0", render=True)

obs = env.reset()

for iter in range(1000):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    #if done:
    #    obs = env.reset()