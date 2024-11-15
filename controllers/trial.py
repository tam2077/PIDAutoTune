import gym
from gym import spaces
import numpy as np
import sys
import math
sys.path.append(r"E:\Webots\lib\controller\python")
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv

from controller import Robot, Camera, DistanceSensor, GPS, Gyro, InertialUnit, Keyboard, Motor, Supervisor

class WebotsDroneEnv(gym.Env):
    def __init__(self):
        super(WebotsDroneEnv, self).__init__()

        # Define action and observation space
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -5, -0.261795, -0.261795, -0.261795, -1, -5, -5, -5, -5, -5, -5]),
            high=np.array([5, 5, 5, 0.261795, 0.261795, 0.261795, 1, 5, 5, 5, 5, 5, 5]),
            dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=np.array([-100, -100, -100, -100, -100, -100, -100, -100, -100]),
            high=np.array([100, 100, 100, 100, 100, 100, 100, 100, 100]),
            dtype=np.float64
        )

        # Initialize Webots environment
        self.robot = RobotSupervisorEnv()
        self.robot.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        
        # Initialize other required components (e.g., motors, sensors)
        self._init_motors()
        self._init_sensors()

    def _init_motors(self):
        # Initialize the motors
        self.motors = []
        for motor_name in ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]:
            motor = self.robot.getDevice(motor_name)
            motor.setPosition(float('inf'))
            self.motors.append(motor)

    def _init_sensors(self):
        # Initialize sensors like IMU, GPS, and Gyro
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(self.robot.timestep)
        
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.robot.timestep)
        
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.robot.timestep)

    def reset(self):
        # Reset the Webots environment and return the initial observation
        self.robot.simulationReset()
        self.robot.step(self.robot.timestep)
        
        return self.get_observations()

    def step(self, action):
        # Apply the action and step the environment forward
        self.apply_action(action)
        self.robot.step(self.robot.timestep)
        
        observation = self.get_observations()
        reward = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()
        
        return observation, reward, done, info

    def get_observations(self):
        # Get the current state of the drone
        position = self.gps.getValues()
        rotation = self.imu.getQuaternion()
        velocity = self.gyro.getValues()

        return np.concatenate([position, rotation, velocity])

    def apply_action(self, action):
        # Apply the action to the motors
        # Convert action into PID gains or motor speeds as needed
        self.motors[0].setVelocity(action[0])
        self.motors[1].setVelocity(action[1])
        self.motors[2].setVelocity(action[2])
        self.motors[3].setVelocity(action[3])
        # Add your action application logic here
        pass

    def get_reward(self, action):
        # Calculate the reward based on the current state and action
        target_location = [2, 2, 2]
        current_position = self.get_observations()[:3]
        distance = np.linalg.norm(np.array(current_position) - np.array(target_location))
        
        reward = -distance  # Example reward function
        return reward

    def is_done(self):
        # Determine if the episode is done
        position = self.get_observations()[:3]
        if np.any(np.abs(position) > 10):  # Example condition
            return True
        return False

    def render(self, mode='human'):
        # Optional: implement rendering if needed
        pass

    def close(self):
        # Clean up resources when closing the environment
        self.robot.simulationQuit(0)

env = WebotsDroneEnv()

# Example usage with a random policy
for episode in range(10):
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Sample random action
        observation, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
