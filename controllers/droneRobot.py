import sys,time, numpy as np
sys.path.append(r"E:\Webots\lib\controller\python")
from controller import Robot, Camera, DistanceSensor, GPS, Gyro, InertialUnit, Keyboard, Motor, Supervisor

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np
import math
from math import sin, cos
from warnings import warn




class DroneRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.FLYING_ALTITUDE = 1.0
        self.altitude_integrator = 0
        self.max_velocity = 600

        self.position_weight = 1
        self.rotation_weight = 1
        self.velocity_weight = 1

        self.target_location = [0,0,10] #Target location in X Y Z.

        self.past_altitude_error = 0
        self.past_pitch_error = 0
        self.past_roll_error = 0
        self.past_yaw_rate_error = 0
        self.past_vx_error = 0
        self.past_vy_error = 0

        self.past_x_global = 0
        self.past_y_global = 0
        self.past_time = 0
        
        self.height_desired = self.target_location[2]
        self.last_ep_score = 0

        self.didOnce = False
        self.actionCount = 0
        self.timestep = self.getBasicTimeStep()

        self.cur_step_count = 0
        self.steps_per_episode = 1000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved


        #Init values to zero
        self.gains_pid = {
            'kp_att_y': 0,
            'kd_att_y': 0,
            
            'kp_att_rp': 0,
            'kd_att_rp': 0,
            
            'kp_vel_xy': 0,
            'kd_vel_xy': 0,

            'kp_z': 0,
            'ki_z': 0,
            'kd_z': 0
        }
        self.motor_power = {
            'm1': 0,
            'm2': 0,
            'm3': 0,
            'm4': 0
        }


        """
        Drone Bot Spaces					
        Num	Observation	    Min 	Max 		
        0	Drone X	        -5	    5		
        1	Drone Y	        -5  	5		
        2	Drone Z	        -5  	5		
        3	Rot X	    -0.261795	0.261795	#	
        4	Rot Y	    -0.261795	0.261795	# Wrong, axis angle format	
        5	Rot Z	    -0.261795	0.261795	#	
        6	Rot W	        -1	    1		
        7	Linear X	    -5  	5		
        8	Linear Y	    -5  	5		
        9	Linear Z	    -5  	5		
        10	Angular X	    -5  	5		
        11	Angular Y	    -5  	5		
        12	Angular Z	    -5  	5	    
        
        """
        self.observation_space = Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]),
                                     high=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete

        """
        Action space
        'kp_att_y': 1, 
        'kd_att_y': 0.5,
        
        'kp_att_rp': 0.5,
        'kd_att_rp': 0.1,
        
        'kp_vel_xy': 2,
        'kd_vel_xy': 0.5,

        'kp_z': 10,
        'ki_z': 5,
        'kd_z': 5

        count = 9

        """
        

        self.action_space = Box(low=np.array([0,0,0,0,0,0,0,0,0]),
                                high=np.array([10,10,10,10,10,10,10,10,10]),
                                dtype=np.float64)
        
        
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
    
        
               


    
    def constrain(self,value, min_val, max_val):
        return min(max_val, max(min_val, value))

    def pid_velocity_fixed_height_controller(self,actual_state, desired_state, gains_pid, dt):
        control_commands = {
            'altitude': 0,
            'roll': 0,
            'pitch': 0,
            'yaw': 0
        }
        self.pid_horizontal_velocity_controller(actual_state, desired_state, gains_pid, dt)
        self.pid_fixed_height_controller(actual_state, desired_state, gains_pid, dt, control_commands)
        self.pid_attitude_controller(actual_state, desired_state, gains_pid, dt, control_commands)
        motor_commands = self.motor_mixing(control_commands)
        return motor_commands

    def pid_fixed_height_controller(self,actual_state, desired_state, gains_pid, dt, control_commands):
        #global past_time,past_altitude_error,past_pitch_error,past_roll_error,past_yaw_rate_error,past_vx_error,past_vy_error,altitude_integrator

        altitude_error = desired_state['altitude'] - actual_state['altitude']
        altitude_derivative_error = (altitude_error - self.past_altitude_error) / dt
        control_commands['altitude'] = (
            self.gains_pid['kp_z'] * self.constrain(altitude_error, -1, 1) +
            self.gains_pid['kd_z'] * altitude_derivative_error +
            gains_pid['ki_z']
        )
        self.altitude_integrator = altitude_error * dt
        control_commands['altitude'] = (
            gains_pid['kp_z'] * self.constrain(altitude_error, -1, 1) +
            gains_pid['kd_z'] * altitude_derivative_error +
            gains_pid['ki_z'] * self.altitude_integrator + 48
        )
        self.past_altitude_error = altitude_error

    def motor_mixing(self,control_commands):
        return {
            'm1': control_commands['altitude'] - control_commands['roll'] + control_commands['pitch'] + control_commands['yaw'],
            'm2': control_commands['altitude'] - control_commands['roll'] - control_commands['pitch'] - control_commands['yaw'],
            'm3': control_commands['altitude'] + control_commands['roll'] - control_commands['pitch'] + control_commands['yaw'],
            'm4': control_commands['altitude'] + control_commands['roll'] + control_commands['pitch'] - control_commands['yaw']
        }

    def pid_attitude_controller(self,actual_state, desired_state, gains_pid, dt, control_commands):
        #global past_time,past_altitude_error,past_pitch_error,past_roll_error,past_yaw_rate_error,past_vx_error,past_vy_error,altitude_integrator
        
        pitch_error = desired_state['pitch'] - actual_state['pitch']
        pitch_derivative_error = (pitch_error - self.past_pitch_error) / dt
        roll_error = desired_state['roll'] - actual_state['roll']
        roll_derivative_error = (roll_error - self.past_roll_error) / dt
        yaw_rate_error = desired_state['yaw_rate'] - actual_state['yaw_rate']

        control_commands['roll'] = gains_pid['kp_att_rp'] * self.constrain(roll_error, -1, 1) + gains_pid['kd_att_rp'] * roll_derivative_error
        control_commands['pitch'] = -gains_pid['kp_att_rp'] * self.constrain(pitch_error, -1, 1) - gains_pid['kd_att_rp'] * pitch_derivative_error
        control_commands['yaw'] = gains_pid['kp_att_y'] * self.constrain(yaw_rate_error, -1, 1)

        self.past_pitch_error = pitch_error
        self.past_roll_error = roll_error
        self.past_yaw_rate_error = yaw_rate_error

    def pid_horizontal_velocity_controller(self,actual_state, desired_state, gains_pid, dt):
        #global past_time,past_altitude_error,past_pitch_error,past_roll_error,past_yaw_rate_error,past_vx_error,past_vy_error,altitude_integrator

        vx_error = desired_state['vx'] - actual_state['vx']
        vx_derivative = (vx_error - self.past_vx_error) / dt
        vy_error = desired_state['vy'] - actual_state['vy']
        vy_derivative = (vy_error - self.past_vy_error) / dt

        pitch_command = gains_pid['kp_vel_xy'] * self.constrain(vx_error, -1, 1) + gains_pid['kd_vel_xy'] * vx_derivative
        roll_command = -gains_pid['kp_vel_xy'] * self.constrain(vy_error, -1, 1) - gains_pid['kd_vel_xy'] * vy_derivative

        desired_state['pitch'] = pitch_command
        desired_state['roll'] = roll_command

        self.past_vx_error = vx_error
        self.past_vy_error = vy_error


    def axis_angle_to_euler(self,axis_angle):
        """Convert an axis-angle representation to Euler angles (in degrees)."""
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

        posField = self.robot.getField("translation")
        rotField = self.robot.getField("rotation")
        

        drone_position = posField.getSFVec3f()
        #drone_position = self.robot.getPosition()

        drone_rotation = rotField.getSFRotation()
        drone_rotation = self.axis_angle_to_euler(drone_rotation)

        speed = self.robot.getVelocity()
        #print(f"Drone Speed is {speed}")
        #print(f"Drone position {drone_position}")
        #print(f"position observation {np.round(drone_position,6)}")
        #print(f"rotation observation {drone_rotation}")
        #print(f"speed observation {np.round(speed,6)}")
        observation = np.concatenate([drone_position, drone_rotation, speed])
        
        return observation

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    # def get_reward(self, action=None):
    # # Calculate the position faults
    #     x_fault = self.get_observations()[0] - self.target_location[0]
    #     y_fault = self.get_observations()[1] - self.target_location[1]
    #     z_fault = self.get_observations()[2] - self.target_location[2]

    #     # Combine position faults into a single array
    #     position_fault = np.array([x_fault, y_fault, z_fault])
        
    #     # Calculate the rotation fault (assuming target angles are all zero)
    #     rotation_fault = np.abs(self.get_observations()[3:6])
        
    #     #Asymptotic reward for surviving longer, caps at 1000 
    #     #survival_reward = 1000 * (1 - np.exp(-0.005 * self.cur_step_count))
    #     survival_reward = 0
    #     #Asymptotic reward for velocity near target, velocity should approach zero when approaching target
    #     #rms_velocity = 
    #     #velocity_reward = 

    #     # Calculate the reward based on position and rotation faults
    #     reward = -self.position_weight * np.sum(np.square(position_fault)) \
    #              -self.rotation_weight * np.sum(np.square(rotation_fault)) \
    #              +survival_reward
        
    #     # Optional: Cap the reward at a minimum value to avoid very large penalties
    #     #reward = max(reward, -10)
        
    #     # Add the reward to the episode score
    #     self.last_ep_score += reward
        
    #     return reward

    def get_reward(self, action):
        # Get the observations
        observations = self.get_observations()
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(observations[:3] - self.target_location)
        
        # Reward for getting closer to target
        reward_distance = -distance_to_target
        
        # Penalize for unstable movements (e.g., large changes in velocity or acceleration)
        velocity = observations[3:6]
        reward_stability = -np.linalg.norm(velocity)
        
        # Reward for efficiency (e.g., reaching target quickly)
        reward_efficiency = -self.getTime()
        
        start_location = np.array([0, 0, 0])  # assuming the starting location is (0, 0, 0)
        current_location = observations[:3]
        distance_from_start = np.linalg.norm(current_location - start_location)
        if distance_from_start < 0.2:  # adjust this value to define "staying in place"
            reward_stay_penalty = -50  # adjust this value to define the penalty
        else:
            reward_stay_penalty = 0

        if np.linalg.norm(observations[:3] - self.target_location) < 0.1:
            goal_bonus = 10
        else:
            goal_bonus = 0
        
        time_penalty = -0.01 * self.cur_step_count

        # Combine rewards
        reward = reward_distance + 0.1 * reward_stability + 0.01 * reward_efficiency + reward_stay_penalty + goal_bonus + time_penalty
        self.last_ep_score = reward
        return reward

    def is_done(self):
              
        bound = 15
        angle_bound = 60
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

        if (x_fault > bound) or (y_fault > bound) or (z_fault > bound):
            print("\033[93mLocation bound triggered\033[0m")
            #warn("Location bound triggered")
            #print("Location bound triggered")
            return True

        angles = np.abs(self.get_observations()[3:6])
        #print(f"Angles are {angles}")
        if (angles[0] > angle_bound) or (angles[1] > angle_bound) or (angles[2] > angle_bound):
            #warn("Angle bound triggered")
            print("\033[93mAngle bound triggered\033[0m")
            #print("Angle bound triggered")
            #return True
            return True #Testing purposes
        else:
            return False
            

    def solved(self):
        # if len(self.episode_score_list) > 100:  # Over 100 trials thus far
        #     if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
        #         return True

        #For now, doesnt end
        return False

    def get_info(self):

        return {"Dummy": "dummy"}

    def render(self, mode='human'):
        pass


        ## THIS IS AN OVERRIDE FUNCTION ###
        # The actions must be applied once per episode instead of once per step since this is adjusting PID values
    def step(self, action):
        """
        The basic step method that steps the controller,
        calls the method that applies the action on the robot
        and returns the (observations, reward, done, info) object.

        :param action: Whatever the use-case uses as an action, e.g.
            an integer representing discrete actions
        :type action: Defined by the implementation of handle_emitter
        :return: tuple, (observations, reward, done, info) as provided by the
            corresponding methods as implemented for the use-case
        """ 
        self.cur_step_count += 1

        # I have no fucking idea why this needs to be here. 
        ## DOES NOT WORK IF DEFINED IN INIT
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

        dt = self.getTime() - self.past_time

        #get measurements
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        altitude = self.gps.getValues()[2]
        x_global = self.gps.getValues()[0]
        if(dt == 0):
            dt = 0.001 # Random number to prevent division by zero
        vx_global = (x_global - self.past_x_global) / dt
        y_global = self.gps.getValues()[1]
        vy_global = (y_global - self.past_y_global) / dt

        # Get body fixed velocities
        actual_yaw = roll
        cosyaw = cos(actual_yaw)
        sinyaw = sin(actual_yaw)
        vx = vx_global * cosyaw + vy_global * sinyaw
        vy = -vx_global * sinyaw + vy_global * cosyaw
        
        desired_state = {
        'roll': 0,
        'pitch': 0,
        'vx': 0,
        'vy': 0,
        'yaw_rate': 0,
        'altitude': self.height_desired
        }
        
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0
        
        self.height_desired += height_diff_desired * dt

        desired_state['yaw_rate'] = yaw_desired

        # PID velocity controller with fixed height
        desired_state['vy'] = sideways_desired
        desired_state['vx'] = forward_desired
        desired_state['altitude'] = self.height_desired

        #PID Control Functions
        self.motor_power = self.pid_velocity_fixed_height_controller(
        actual_state={
            'roll': roll,
            'pitch': pitch,
            'yaw_rate': yaw_rate,
            'altitude': altitude,
            'vx': vx,
            'vy': vy
        },
        desired_state=desired_state,
        gains_pid=self.gains_pid,
        dt=dt
        )

        #print(f"motor power {self.motor_power}")
        
        
        #self.m1_motor.setVelocity
        

        self.m1_motor.setVelocity(np.clip(-self.motor_power['m1'],-self.max_velocity,self.max_velocity))
        self.m2_motor.setVelocity(np.clip(self.motor_power['m2'],-self.max_velocity,self.max_velocity))
        self.m3_motor.setVelocity(np.clip(-self.motor_power['m3'],-self.max_velocity,self.max_velocity))
        self.m4_motor.setVelocity(np.clip(self.motor_power['m4'],-self.max_velocity,self.max_velocity))
        # Save past time for next time step
        self.past_time = self.getTime()
        self.past_x_global = x_global
        self.past_y_global = y_global
        
        # var = np.round([
        # self.past_altitude_error,
        # self.past_pitch_error,
        # self.past_roll_error,
        # self.past_yaw_rate_error,
        # self.past_vx_error,
        # self.past_vy_error,
        # self.altitude_integrator
        # ])
        # #print(var)

        

        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    ## Guess what, another override function. Only change is to add and reset a flag
    def reset(self):
        """
        Used to reset the world to an initial state.

        Default, problem-agnostic, implementation of reset method,
        using Webots-provided methods.

        *Note that this works properly only with Webots versions >R2020b
        and must be overridden with a custom reset method when using
        earlier versions. It is backwards compatible due to the fact
        that the new reset method gets overridden by whatever the user
        has previously implemented, so an old supervisor can be migrated
        easily to use this class.

        :return: default observation provided by get_default_observation()
        """
        #Not sure about this, reset all global variables for now
        self.past_altitude_error = 0
        self.past_pitch_error = 0
        self.past_roll_error = 0
        self.past_yaw_rate_error = 0
        self.past_vx_error = 0
        self.past_vy_error = 0
        self.past_x_global = 0
        self.past_y_global = 0
        #self.past_time = self.getTime()

        self.motor_power = {
            'm1': 0,
            'm2': 0,
            'm3': 0,
            'm4': 0
        }
        self.cur_step_count = 0
        self.didOnce = False
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep)
        #return self.get_default_observation()
        return np.zeros(self.observation_space.shape)
         
    





    def apply_action(self, action):
        print(f"last episodes score {self.last_ep_score}")
        self.last_ep_score = 0
        print(f"Action is {action}")
        self.gains_pid = {
            'kp_att_y': action[0],
            'kd_att_y': action[1],
            
            'kp_att_rp': action[2],
            'kd_att_rp': action[3],
            
            'kp_vel_xy': action[4],
            'kd_vel_xy': action[5],

            'kp_z': action[6],
            'ki_z': action[7],
            'kd_z': action[8]
        }
