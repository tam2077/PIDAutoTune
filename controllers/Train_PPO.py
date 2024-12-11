# Import necessary libraries
from droneRobot import DroneRobot  # Import the DroneRobot class to define the custom environment for the drone
from stable_baselines3 import PPO  # Import PPO (Proximal Policy Optimization) from Stable Baselines 3 for RL
from stable_baselines3.common.vec_env import DummyVecEnv  # Import DummyVecEnv to handle vectorized environments
import os, datetime  # Import os for file path management and datetime for generating unique model names

# Define the path for saving the trained model, using the current timestamp to create a unique filename
ppo_path = os.path.join('Training', 'Saved Models', f'Drone_PPO_Model_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
# The file path will be generated with the current date and time to ensure the model file is unique
log_path = os.path.join('Training', 'Logs')  # Path to store the TensorBoard logs for monitoring training progress

# Set the number of episodes to train the agent
episodes = 100  # Number of episodes to train the agent (a single run of the environment)

# Initialize the environment (the drone control simulation)
env = DroneRobot()
env.debugMode = True  # Enable debug mode for additional information and debugging during training
env.startTensorBoard(log_path)  # Start TensorBoard to log training progress for visualization

# Calculate the total number of timesteps to train the agent
timesteps = env.steps_per_episode * episodes  # Total number of timesteps based on steps per episode and number of episodes

# Wrap the environment in a DummyVecEnv for compatibility with vectorized environments in Stable Baselines 3
env = DummyVecEnv([lambda: env])  # DummyVecEnv is used to handle a single environment in a vectorized format

# Initialize the PPO model with specific parameters
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, device='cpu', ent_coef=10, learning_rate=0.0001)
# 'MlpPolicy' specifies the use of a neural network-based policy (MLP)
# 'verbose=1' enables logging of training details during the process
# 'tensorboard_log' specifies where to store TensorBoard logs for training visualization
# 'device='cpu'' ensures that the model is trained on the CPU (change to 'cuda' for GPU)
# 'ent_coef=10' adjusts the entropy coefficient to control exploration (higher values promote more exploration)
# 'learning_rate=0.0001' sets the learning rate for the model optimization

# Start the training process for the specified number of timesteps
model.learn(total_timesteps=timesteps)  # Train the model for the calculated total timesteps

# Save the trained model to the specified path with a timestamp-based filename
model.save(ppo_path)  # Save the trained PPO model to the path defined earlier

# Print a message indicating that the training is complete
print("Done training")  # Display a message confirming the completion of the training process
