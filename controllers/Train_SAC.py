# Import necessary libraries
from droneRobot import DroneRobot  # Import the DroneRobot class, which contains the drone environment
import numpy as np  # Import numpy for numerical operations, though it is not used in this snippet
from stable_baselines3 import SAC  # Import the SAC (Soft Actor-Critic) algorithm from Stable Baselines 3
import os  # Import os module for file and directory path management

# Define the path for saving the trained model and logging
model_path = os.path.join('Training', 'Saved Models', 'SAC_model')  # Path to save the trained model
log_path = os.path.join('Training', 'Logs')  # Path to save TensorBoard logs for monitoring the training process

# Initialize the environment (the drone control simulation)
env = DroneRobot()
env.target_location = [0, 0, 2] #X Y Z target location, go to this location. All reward is calculated based on this
env.debugMode = False  # Enable debug mode for testing your PID constants, defined in take_action() function
# Set the number of episodes to run and compute total timesteps based on the steps per episode
episodes = 1000  # Number of episodes to train the agent
timesteps = env.steps_per_episode * episodes  # Total timesteps for training, calculated based on steps per episode

# Start TensorBoard for monitoring the training progress (log path specified)
env.startTensorBoard(log_path)  # Initializes TensorBoard logging for visualization during training

# Create the SAC model using the MLP (Multi-Layer Perceptron) policy for continuous control tasks
model = SAC("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log=log_path)
# "MlpPolicy" specifies the use of a neural network-based policy
# "verbose=1" enables some logging details during training
# "device='cuda'" ensures the model uses the GPU if available for faster training
# "tensorboard_log" specifies the log directory for TensorBoard visualization

# Start training the model for the specified number of timesteps
model.learn(total_timesteps=timesteps, log_interval=4)  
# "total_timesteps" defines how long the model should be trained
# "log_interval=4" sets the frequency of logging to TensorBoard every 4 updates

# Save the trained model to the specified directory
model.save("sac_Drone", path=model_path)  
# Saves the trained model under the name "sac_Drone" in the defined path
