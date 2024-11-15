from droneRobot import DroneRobot

import numpy as np

from stable_baselines3 import SAC

import os

env = DroneRobot()
try:
    episodes = 10000
    timesteps = 1000*episodes
    model_path = os.path.join('Training','Saved Models','SAC_model')
    log_path = os.path.join('Training','Logs')

    model = SAC("MlpPolicy", env, verbose=1,device='cuda',tensorboard_log=log_path)
    model.learn(total_timesteps=timesteps, log_interval=4)
    model.save("sac_Drone",path=model_path)
except KeyboardInterrupt:
    model.save("sac_Drone",path=model_path)
    print(f"Model should be saved successfully at {model_path}")
