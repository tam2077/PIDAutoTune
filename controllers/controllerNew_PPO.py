from droneRobot import DroneRobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


env = DroneRobot()
env = DummyVecEnv([lambda: env])
try:
    log_path = os.path.join('Training', 'Logs')
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path,device='cpu',ent_coef=2,learning_rate=0.00001)
    ppo_path = os.path.join('Training','Saved Models','Drone_PPO_Model_1')

    episodes = 100000
    timesteps = 1000*episodes
    model.learn(total_timesteps=timesteps)
    model.save(ppo_path)
    print("Done training")
except KeyboardInterrupt:
    print("Interrupted, saving and exiting")
    model.save(ppo_path)
    print("Saved")
