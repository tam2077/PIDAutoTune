# from droneRobot import DroneRobot

# import numpy as np
# import os

# from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# import gym

# env = DroneRobot()

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# model_path = os.path.join('Training','Saved Models','SAC_model')
# log_path = os.path.join('Training','Logs')
# episodes = 100000
# timesteps = 1000*episodes

# model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,device='cuda',tensorboard_log=log_path)
# #model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,device='cpu')
# model.learn(total_timesteps=timesteps, log_interval=10)
# model.save("td3_drone",path=model_path)

from droneRobot import DroneRobot

import numpy as np
import os

from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import gym

env = DroneRobot()

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
model_path = os.path.join('Training','Saved Models','td3_drone_model')
log_path = os.path.join('Training','Logs','td3_drone_logs')
episodes = 100000
timesteps = 1000*episodes

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device='cuda', tensorboard_log=log_path)
model.learn(total_timesteps=timesteps, log_interval=10)
model.save("td3_drone_model", path=model_path)