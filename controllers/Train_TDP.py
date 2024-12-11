# Import necessary libraries
from droneRobot import DroneRobot  # Import the custom DroneRobot class to define the environment for the drone
import numpy as np  # Import numpy for numerical operations, especially for creating the noise
import os, datetime  # Import os for file path handling and datetime for generating unique filenames
from stable_baselines3 import TD3  # Import the TD3 (Twin Delayed DDPG) algorithm from Stable Baselines 3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise  # Import Ornstein-Uhlenbeck noise for action noise

# Define the path for saving the trained model with a timestamp for uniqueness
model_path = os.path.join('Training', 'Saved Models', f'td3_drone_model{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
# Define the log path for storing TensorBoard logs
log_path = os.path.join('Training', 'Logs', 'td3_drone_logs')

# Initialize the environment (the drone control simulation)
env = DroneRobot()
env.target_location = [0, 0, 2] #X Y Z target location, go to this location. All reward is calculated based on this
env.debugMode = False  # Enable debug mode for testing your PID constants, defined in take_action() function

# Get the number of actions in the action space of the environment
n_actions = env.action_space.shape[-1]  # Number of actions, extracted from the environment's action space shape

# Initialize Ornstein-Uhlenbeck action noise for exploration during training
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
# This noise is used to add randomness to the actions, promoting exploration in the environment

# Start TensorBoard to log training progress and visualize it
env.startTensorBoard(log_path)

# Set the number of episodes to train the agent
episodes = 1000  # Number of episodes for training
timesteps = env.steps_per_episode * episodes  # Total number of timesteps, calculated by steps per episode and episodes

# Create the TD3 model with specific parameters
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device='cuda', tensorboard_log=log_path)
# "MlpPolicy" specifies the use of a Multi-Layer Perceptron-based policy for continuous action spaces
# action_noise adds noise to the actions for better exploration
# verbose=1 enables some logging details during training
# device='cuda' makes use of the GPU for faster training (change to 'cpu' if GPU is not available)
# tensorboard_log specifies where to store TensorBoard logs for visualization

# Start training the model for the specified number of timesteps
model.learn(total_timesteps=timesteps, log_interval=10)
# log_interval=10 specifies that logs will be output every 10 updates during the training process

# Save the trained model after completing the training
model.save("td3_drone_model", path=model_path)  
# Save the model with the filename "td3_drone_model" at the path defined earlier

