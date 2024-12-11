
import sys
import os
import threading

sys.path.append(r"E:\Webots\lib\controller\python")
if(os.environ.get('WEBOTS_HOME') is not None):
    print("WEBOTS_HOME found")
    sys.path.append(os.environ.get('WEBOTS_HOME') + '//lib//controller//python')
else:
    print("WEBOTS_HOME is not defined, please set your environment variable correctly.")


import numpy as np
from controller import GPS, Gyro, InertialUnit, Motor, Supervisor

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from gym.spaces import Box

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv

import math
from math import sin, cos
from warnings import warn
import random


class DroneRobot(RobotSupervisorEnv):

    def __init__(self):
        """
        /*
        * @brief Initializes the DroneRobot environment and sets default parameters.
        *
        * This constructor initializes the simulation environment, sets up the observation and action spaces,
        * and defines key parameters such as PID gains, motor power, and episode configurations.
        * Additionally, it configures the random seed for reproducibility and sets up the simulation timestep.
        *
        * @details
        * - Initializes debug mode, random seed, and task status.
        * - Configures observation and action spaces using Gym's `Box`.
        * - Sets up PID gains with random initial values and motor power to zero.
        * - Defines simulation parameters such as timestep and maximum steps per episode.
        * - Maintains error tracking for the PID controller.
        *
        * @note The method assumes that the superclass `RobotSupervisorEnv` is correctly implemented.
        */
        """
        super().__init__()
        self.debugMode = False  # Set to True to enable debug mode
        self.randomSeed = 42
        random.seed(self.randomSeed)  # Set the random seed for reproducibility
        self.saveOK = False  # Set to True when the task is solved
        self.location_bound = 5
        self.angle_bound = 90
        self.max_velocity = 150 #Max velocity
    
        self.avg_target_score = -20 #set this to define when the task is considered solved
        self.previous_velocity = 0

        self.target_location = [0,0,2]

        
        self.past_errors = {"x": 0, "y": 0, "z": 0}
        self.integral_errors = {"x": 0, "y": 0, "z": 0}
        self.past_yaw_error = 0
        
        self.last_ep_score = 0

        self.didOnce = False
        self.actionCount = 0
        # Get the Webots simulation timestep
        self.timestep = self.getBasicTimeStep()
        
        self.dt = 1/200 # Run at 200 Hz
        self.cur_step_count = 0
        self.steps_per_episode = 3000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
 
        self.yaw_gain = 0 #Yaw gain, Yaw is set to move the drone in the general direction of the goal
        #Init values to random numbers
        self.pid_gains = {
            'X_P': random.uniform(0,100),
            'X_I': random.uniform(0,100),
            'X_D': random.uniform(0,100),
            
            'Y_P': random.uniform(0,100),
            'Y_I': random.uniform(0,100),
            'Y_D': random.uniform(0,100),
            
            'Z_P': random.uniform(0,100),
            'Z_I': random.uniform(0,100),
            'Z_D': random.uniform(0,100),
        }
        self.motor_power = {
            'm1': 0,
            'm2': 0,
            'm3': 0,
            'm4': 0
        }


        """
        Drone Bot Spaces					
        Num	Observation	    	
        0	Drone X	        
        1	Drone Y	        		
        2	Drone Z	        		
        3	Rot X	    	
        4	Rot Y	    
        5	Rot Z	    	
        6	Rot W	        		
        7	Linear X	 		
        8	Linear Y	 		
        9	Linear Z	 		
        10	Angular X	 		
        11	Angular Y		
        12	Angular Z	    
        
        """
        #All bounds set to infinite
        self.observation_space = Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]),
                                     high=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete

        self.action_space = Box(low=np.array([0,0,0,0,0,0,0,0,0]),
                                high=np.array([100,100,100,100,100,100,100,100,100]),
                                dtype=np.float16)
        
        
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
    

    def update_motor_power(self, current_location):
        """
            /**
            * @brief Updates the motor power for the drone based on the PID controller and the current location.
            *
            * This method calculates the error between the drone's current location and the target location for
            * each axis (x, y, z) and adjusts the motor power accordingly using PID control. It also applies yaw
            * control to align the drone towards the target direction. For testing purposes, yaw gain can be set to 
            * zero.
            *
            * @param current_location A list containing the drone's current position (x, y, z) and yaw angle.
            *
            * @details
            * - Computes proportional (P), integral (I), and derivative (D) terms for PID control on each axis.
            * - Normalizes yaw angle errors to the range [-π, π].
            * - Updates motor power values for maintaining stability and reaching the target location.
            * - Clips motor power values to ensure they remain within the allowed range.
            *
            * @note Assumes the `current_location` input includes yaw as the third element.
            * @see self.pid_gains, self.motor_power
            */

        """
        # Compute error between target and current position
        error = {
            "x": self.target_location[0] - current_location[0],
            "y": self.target_location[1] - current_location[1],
            "z": self.target_location[2] - current_location[2],
        }

        # Compute the angle (yaw) to target location in the xy-plane (in radians)
        delta_x = self.target_location[0] - current_location[0]
        delta_y = self.target_location[1] - current_location[1]
        desired_yaw = math.atan2(delta_y, delta_x)

        # Calculate the drone's current yaw angle (orientation in the xy-plane)
        current_yaw = math.atan2(current_location[1], current_location[0])

        # Calculate the error in yaw (angle difference), normalized to [-pi, pi]
        error_yaw = desired_yaw - current_yaw
        error_yaw = (error_yaw + math.pi) % (2 * math.pi) - math.pi

        # Initialize motor power dictionary
        motor_power = {
            'm1': 0,
            'm2': 0,
            'm3': 0,
            'm4': 0
        }

        # PID calculations for each axis (x, y, z)
        for axis in ['x', 'y', 'z']:
            # Proportional term (P)
            P = self.pid_gains[f'{axis.upper()}_P'] * error[axis]
            
            # Integral term (I)
            self.integral_errors[axis] += error[axis] * self.dt
            I = self.pid_gains[f'{axis.upper()}_I'] * self.integral_errors[axis]
            
            # Derivative term (D)
            D = self.pid_gains[f'{axis.upper()}_D'] * (error[axis] - self.past_errors[axis]) / self.dt

            # Total PID output for this axis
            PID_output = P + I + D

            # Update motor power based on the axis
            if axis == "x":
                motor_power['m1'] -= PID_output  # Adjust for roll (x-axis)
                motor_power['m2'] += PID_output
                motor_power['m3'] += PID_output
                motor_power['m4'] -= PID_output
            elif axis == "y":
                motor_power['m1'] -= PID_output  # Adjust for pitch (y-axis)
                motor_power['m2'] -= PID_output
                motor_power['m3'] += PID_output
                motor_power['m4'] += PID_output
            elif axis == "z":
                motor_power['m1'] += PID_output  # Adjust for vertical control
                motor_power['m2'] += PID_output
                motor_power['m3'] += PID_output
                motor_power['m4'] += PID_output

        # Yaw control using Proportional control (P only)
        yaw_P = self.yaw_gain * error_yaw
        motor_power['m1'] += yaw_P  # Adjust for yaw
        motor_power['m2'] -= yaw_P
        motor_power['m3'] += yaw_P
        motor_power['m4'] -= yaw_P
        #print("Pre-clipped motor power:", motor_power)
        # Clip the values to ensure motors' power stays within reasonable limits
        motor_power = {key: np.clip(value, 0, self.max_velocity) for key, value in motor_power.items()}

        # Set motor power directly
        self.motor_power = motor_power

        # Update past errors for the next iteration
        for axis in ['x', 'y', 'z']:
            self.past_errors[axis] = error[axis]
        self.past_errors['yaw'] = error_yaw
        #print(self.motor_power)

    def axis_angle_to_euler(self,axis_angle):
        """
            /**
            * @brief Converts an axis-angle representation to Euler angles in degrees.
            *
            * This method takes a 4D vector representing an axis-angle (x, y, z, angle)
            * and converts it to the corresponding Euler angles (roll, pitch, yaw) in degrees.
            *
            * @param axis_angle A list or tuple containing the axis components (x, y, z) and the rotation angle.
            * @return A tuple (roll, pitch, yaw) representing the Euler angles in degrees.
            *
            * @details
            * - Normalizes the axis vector if its magnitude is non-zero.
            * - Converts the axis-angle to a quaternion and then to Euler angles.
            * - Handles edge cases to ensure angles are properly constrained.
            *
            * @note The input angle is expected in radians.
            */

        """
        x, y, z, angle = axis_angle
        
        # Normalize the axis vector
        axis_length = math.sqrt(x*x + y*y + z*z)
        if axis_length != 0:
            x /= axis_length
            y /= axis_length
            z /= axis_length
        
        # Convert angle to radians
        angle_rad = angle
        
        # Convert axis-angle to quaternion
        w = math.cos(angle_rad / 2)
        x = x * math.sin(angle_rad / 2)
        y = y * math.sin(angle_rad / 2)
        z = z * math.sin(angle_rad / 2)
        
        # Convert quaternion to Euler angles
        ysqr = y * y
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        roll = math.degrees(math.atan2(t0, t1))
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.degrees(math.asin(t2))
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        yaw = math.degrees(math.atan2(t3, t4))
        
        return roll, pitch, yaw

    def get_observations(self):
        """
            /**
            * @brief Retrieves the current state of the drone as observations.
            *
            * This method gathers the drone's current position, orientation, and velocity, and combines them
            * into a single observation vector.
            *
            * @return A numpy array containing the concatenated observations: 
            *         [position (x, y, z), rotation (roll, pitch, yaw), velocity (linear and angular)].
            *
            * @details
            * - The position is obtained from the `translation` field of the robot.
            * - The rotation is converted from axis-angle representation to Euler angles using `axis_angle_to_euler`.
            * - The velocity includes both linear and angular components.
            *
            * @note Assumes that the Webots simulation provides the required fields and velocity information.
            * @see axis_angle_to_euler
            */

        """

        posField = self.robot.getField("translation")
        rotField = self.robot.getField("rotation")
        drone_position = posField.getSFVec3f()

        drone_rotation = rotField.getSFRotation()
        drone_rotation = self.axis_angle_to_euler(drone_rotation)
        speed = self.robot.getVelocity()
        observation = np.concatenate([drone_position, drone_rotation, speed])
        
        return observation

    def get_default_observation(self):
        """
            /**
            * @brief Provides a default observation vector.
            *
            * This method returns a default observation consisting of zero values, which matches the shape
            * of the observation space.
            *
            * @return A list of zeros with a length equal to the number of dimensions in the observation space.
            *
            * @details
            * - The default observation is used as a placeholder or initialization value when no meaningful
            *   observation data is available.
            */

        """
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        """
            /**
            * @brief Computes the reward based on the drone's current state and actions.
            *
            * This method evaluates the drone's performance in the environment by calculating a composite reward
            * that combines various factors such as distance to the target, stability, efficiency, and task completion.
            *
            * @param action (Optional) The action taken by the drone, though not directly used in the current implementation.
            * @return A float value representing the computed reward for the current timestep.
            *
            * @details
            * - **Distance to Target:** A quadratic penalty based on the squared distance to the target location.
            * - **Stability Reward:** Penalizes abrupt changes in velocity.
            * - **Efficiency Reward:** Adds a small time penalty to encourage faster task completion.
            * - **Stay Penalty:** Penalizes the drone for staying near the origin after a certain number of steps.
            * - **Direction Reward:** Rewards the drone for moving in the direction of the target.
            * - **Goal Bonus:** Provides a significant reward for reaching the target location.
            * - **Time Penalty:** Encourages efficiency by applying a small penalty for each timestep.
            *
            * @note This function assumes observations include position and velocity as described in `get_observations`.
            */

        """
        observations = self.get_observations()
        
        # Distance to target
        distance_to_target = np.linalg.norm(observations[:3] - self.target_location)
        reward_distance = -distance_to_target**2  # Quadratic penalty

        # Stability reward (encourage smooth movement)
        velocity = observations[3:6]
        velocity_change = np.linalg.norm(velocity - self.previous_velocity)
        reward_stability = -0.1 * velocity_change

        # Efficiency reward (smaller time penalty)
        reward_efficiency = -0.01 * self.cur_step_count

        # Penalty for staying near the origin
        start_location = np.array([0, 0, 0])
        distance_from_start = np.linalg.norm(observations[:3] - start_location)
        reward_stay_penalty = -50 if distance_from_start < 0.1 and self.cur_step_count > 100 else 0

        # Direction reward (encourage moving toward the target)
        direction = (self.target_location - observations[:3]) / np.linalg.norm(self.target_location - observations[:3])
        reward_direction = max(0, np.dot(direction, velocity))  # Positive reward

        # Goal bonus (larger incentive for success)
        goal_bonus = 50 if distance_to_target < 0.1 else 0

        # Time penalty (encourage faster task completion)
        time_penalty = -0.01 * self.cur_step_count

        # Combine rewards
        reward = (
            reward_distance 
            + 0.1 * reward_stability 
            + reward_efficiency 
            + reward_stay_penalty 
            + reward_direction 
            + goal_bonus 
            + time_penalty
        )
        
        self.previous_velocity = velocity
        self.last_ep_score = reward
        
        return reward
    
    def is_done(self):    
        """
            /**
            * @brief Checks whether the current episode is complete.
            *
            * This method evaluates conditions to determine if the episode should terminate, such as exceeding
            * step limits, going out of bounds, or exceeding angular constraints.
            *
            * @return `True` if the episode is complete, `False` otherwise.
            *
            * @details
            * - **Step Count Trigger:** The episode ends if the current step count exceeds the maximum allowed steps.
            * - **Location Bound Trigger:** The episode terminates if the drone's position deviates beyond a predefined
            *   bound from the target location on any axis.
            * - **Angle Bound Trigger:** The episode terminates if any rotational angle (roll, pitch, yaw) exceeds
            *   the specified angular limit.
            *
            * @note Observations are fetched from `get_observations`, and positional and angular bounds are defined
            *       within the function.
            */

        """  
        if self.cur_step_count >= self.steps_per_episode:
            print("\033[93mStep count triggered\033[0m")
            #warn("Step count triggered")
            #print("Step count triggered")
            self.stepFlag = True
            return True
        
        x_fault = np.abs(self.get_observations()[0] - self.target_location[0])
        y_fault = np.abs(self.get_observations()[1] - self.target_location[1])
        z_fault = np.abs(self.get_observations()[2] - self.target_location[2])

        #print(x_fault,y_fault,z_fault)

        if (x_fault > self.location_bound) or (y_fault > self.location_bound) or (z_fault > self.location_bound):
            print("\033[93mLocation bound triggered\033[0m")
            #warn("Location bound triggered")
            #print("Location bound triggered")
            return True

        angles = np.abs(self.get_observations()[3:6])
        #print(f"Angles are {angles}")
        if (angles[0] > self.angle_bound) or (angles[1] > self.angle_bound) or (angles[2] > self.angle_bound):
            #warn("Angle bound triggered")
            print("\033[93mAngle bound triggered\033[0m")
            #print("Angle bound triggered")
            return True 
        else:
            return False
            

    def solved(self):
        """
            /**
            * @brief Determines if the task is considered "solved."
            *
            * The task is deemed solved if the mean score of the last 100 episodes exceeds
            * a predefined threshold (`avg_target_score`), indicating consistent performance.
            *
            * @return `True` if the task is solved, `False` otherwise.
            *
            * @details
            * - **Threshold Check:** The task is solved if the mean score of the last 100 episodes is greater than `avg_target_score`.
            * - If the condition is met, the `saveOK` flag is set to `True`.
            *
            * @note Requires the `episode_score_list` to contain at least 100 episodes.
            */

        """
        if len(self.episode_score_list) > 100:  # Last 100 trials
            mean_score = np.mean(self.episode_score_list[-100:])
            if mean_score > self.avg_target_score:  # Target a low average penalty
                self.saveOK = True
                return True
        return False

    def get_info(self):
        """
            /**
            * @brief Provides additional information about the environment.
            *
            * This method returns a dictionary containing environment-specific details.
            *
            * @return A dictionary with key-value pairs representing additional environment information.
            *
            * @details
            * - Currently, this method returns a placeholder dictionary `{"Dummy": "dummy"}`.
            * - Can be extended to include meaningful diagnostic or metadata information.
            */

        """

        return {"Dummy": "dummy"}
    def launchTensorBoard(self,log_path):
        """
            /**
            * @brief Launches TensorBoard for monitoring training progress.
            *
            * This method runs a system command to start TensorBoard using the specified log directory.
            *
            * @param log_path The path to the directory containing TensorBoard log files.
            *
            * @details
            * - Executes the command `tensorboard --logdir=<log_path>` to launch TensorBoard.
            * - Requires TensorBoard to be installed and accessible from the system's command line.
            *
            * @note Ensure the specified `log_path` is valid and contains log files for visualization.
            */
        """
        os.system('tensorboard --logdir=' + log_path)
        return
    def startTensorBoard(self,log_path):
        """
            /**
            * @brief Starts TensorBoard in a separate thread.
            *
            * This method creates and starts a new thread to run TensorBoard, ensuring that it does not block the main process.
            *
            * @param log_path The path to the directory containing TensorBoard log files.
            *
            * @details
            * - Internally calls `launchTensorBoard` in a separate thread.
            * - Useful for starting TensorBoard asynchronously during training or simulation.
            *
            * @note Ensure the `log_path` is valid and TensorBoard is installed.
            */
        """
        t = threading.Thread(target=self.launchTensorBoard, args=([log_path]))
        t.start()
        return

    def render(self, mode='human'):
        """
            /**
            * @brief Renders the environment.
            *
            * This method is a placeholder for rendering the environment in various modes.
            *
            * @param mode The rendering mode (default is `"human"`).
            *
            * @details
            * - Currently, this method is not implemented.
            * - Can be extended to provide visualizations or other forms of rendering.
            */
        """
        pass
        ## THIS IS AN OVERRIDE FUNCTION ###
        # The actions must be applied once per episode instead of once per step since this is adjusting PID values
    def step(self, action):
        """    /**
     * @brief Executes a single simulation step, applying an action and updating the environment.
     *
     * This method steps the controller, applies the given action to the robot, and returns the resulting
     * state of the environment, including observations, reward, termination status, and additional info.
     *
     * @param action The action to be applied to the robot, defined by the use case (e.g., integer for discrete actions).
     * @return A tuple `(observations, reward, done, info)`:
     *         - `observations`: Current state of the environment.
     *         - `reward`: Reward received for the action taken.
     *         - `done`: Boolean indicating whether the episode is complete.
     *         - `info`: Additional diagnostic information.
     *
     * @details
     * - Increments the step count and initializes motor devices if not already done.
     * - Applies the action using `apply_action` and updates motor power with `update_motor_power`.
     * - Fetches observations, computes the reward, and checks if the episode is done.
     * - Handles integration with the Webots supervisor timestep.
     *
     * @note Assumes valid initialization of motors and sensors (e.g., GPS, IMU, gyro).
     */
        """
        self.cur_step_count += 1
        ## Needs to be defined in step to work properly
        self.m1_motor = self.getDevice('m1_motor') # type: Motor
        self.m1_motor.setPosition(float('inf'))
        
        self.m2_motor = self.getDevice('m2_motor') # type: Motor
        self.m2_motor.setPosition(float('inf'))
        
        self.m3_motor = self.getDevice('m3_motor') # type: Motor
        self.m3_motor.setPosition(float('inf'))
        
        self.m4_motor = self.getDevice('m4_motor') # type: Motor
        self.m4_motor.setPosition(float('inf'))

        self.imu = self.getDevice('inertial_unit') # type: InertialUnit
        self.imu.enable(self.timestep)

        self.gps = self.getDevice('gps') # type: GPS
        self.gps.enable(self.timestep)

        self.gyro = self.getDevice('gyro') # type: Gyro
        self.gyro.enable(self.timestep)

        if not self.didOnce:
            self.apply_action(action)
            self.didOnce = True
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()
        #get measurements
        observation = self.get_observations()
        current_location = [
            observation[0],
            observation[1],
            observation[2],
        ]
        self.update_motor_power(current_location)
        self.m1_motor.setVelocity(-self.motor_power['m1'])
        self.m2_motor.setVelocity(self.motor_power['m2'])
        self.m3_motor.setVelocity(-self.motor_power['m3'])
        self.m4_motor.setVelocity(self.motor_power['m4'])
        


        return (
            observation,
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    ## Guess what, another override function. Only change is to add and reset a flag
    def reset(self):
        """
            /**
            * @brief Resets the environment to its initial state.
            *
            * This method restores the simulation to its default configuration by resetting motor power,
            * step counters, error terms, and the Webots simulation state.
            *
            * @return A default observation, represented as a zero-filled numpy array matching the observation space shape.
            *
            * @details
            * - Resets motor power values to zero and clears PID error terms.
            * - Resets the Webots simulation state and physics using `simulationReset` and `simulationResetPhysics`.
            * - Ensures compatibility with Webots versions >R2020b but can be overridden for earlier versions.
            * - Calls `super().step` to integrate the reset state into the simulation timestep.
            *
            * @note This method is backward-compatible, allowing older supervisor implementations to be migrated.
            */
        """

        self.motor_power = {
            'm1': 0,
            'm2': 0,
            'm3': 0,
            'm4': 0
        }
        self.cur_step_count = 0
        self.didOnce = False
        self.past_errors = {"x": 0, "y": 0, "z": 0}
        self.integral_errors = {"x": 0, "y": 0, "z": 0}
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep)
        #return self.get_default_observation()
        return np.zeros(self.observation_space.shape)
         

    def apply_action(self, action):
        """
            /**
            * @brief Applies the specified action to the drone by updating PID controller gains.
            *
            * This method sets the PID gains based on the provided action. If debug mode is active,
            * the action is ignored, and default PID gains are used instead.
            *
            * @param action A list or array containing PID gain values for X, Y, and Z axes in the order:
            *               [X_P, X_I, X_D, Y_P, Y_I, Y_D, Z_P, Z_I, Z_D].
            *
            * @details
            * - In debug mode, fixed PID gains are applied, and the action parameter is ignored.
            * - Outside debug mode, the PID gains are updated directly from the action input.
            * - Resets the last episode score to zero after applying the action.
            *
            * @note Ensure that the action parameter contains exactly 9 elements representing the PID gains.
            */

        """

        if self.debugMode == True:
            print("Debug mode is active, ignoring model actions")
            self.pid_gains = {
            'X_P': 10,
            'X_I': 10,
            'X_D': 0.5,
            
            'Y_P': 10,
            'Y_I': 10,
            'Y_D': 0.5,
            
            'Z_P': 10,
            'Z_I': 10,
            'Z_D': 0.5,
        }
            
        else:
            print(f"Action is {action}")
            self.pid_gains = {
            'X_P': action[0],
            'X_I': action[1],
            'X_D': action[2],
            
            'Y_P': action[3],
            'Y_I': action[4],
            'Y_D': action[5],
            
            'Z_P': action[6],
            'Z_I': action[7],
            'Z_D': action[8],
        }
        print(f"last episodes score {self.last_ep_score}")
        self.last_ep_score = 0
