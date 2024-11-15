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




class droneRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        
        self.timestep = self.getBasicTimeStep()
        
        self.observation_space = Box(low=np.array([-5,-5,-5,-0.261795,-0.261795,-0.261795,-1,-5,-5,-5,-5,-5,-5]),
                                     high=np.array([5,5,5,0.261795,0.261795,0.261795,1,5,5,5,5,5,5]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete

        self.action_space = Box(low=np.array([-100,-100,-100,-100,-100,-100,-100,-100,-100]),
                                high=np.array([100,100,100,100,100,100,100,100,100]),
                                dtype=np.float64)
        
        
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
    
        #Init Motors

        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.target_location = [2,2,2] #Target location in X Y Z. Unused for now

    def step(self, action):
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()
        self.m1_motor = self.getDevice('m1_motor') # type: Motor
        self.m1_motor.setPosition(float('inf'))
        
        self.m2_motor = self.getDevice('m2_motor') # type: Motor
        self.m2_motor.setPosition(float('inf'))
        
        self.m3_motor = self.getDevice('m3_motor') # type: Motor
        self.m3_motor.setPosition(float('inf'))
        
        self.m4_motor = self.getDevice('m4_motor') # type: Motor
        self.m4_motor.setPosition(float('inf'))

        self.m1_motor.setVelocity(-100)
        self.m2_motor.setVelocity(100)
        self.m3_motor.setVelocity(-100)
        self.m4_motor.setVelocity(100)
        
        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )


    def get_observations(self):

        return [0,0,0]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        return 1

    def is_done(self):
    
        return False

    def solved(self):
        # if len(self.episode_score_list) > 100:  # Over 100 trials thus far
        #     if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
        #         return True

        #For now, doesnt end
        return False

    def get_info(self):
        return None

    def render(self, mode='human'):
        pass


        ## THIS IS AN OVERRIDE FUNCTION ###
        # The actions must be applied once per episode instead of once per step since this is adjusting PID values

    ## Guess what, another override function. Only change is to add and reset a flag
    def reset(self):
        self.didOnce = False
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep)
        return self.get_default_observation()





    def apply_action(self, action):
        print(f"Action is {action}")
        


            
        


env = droneRobot()
env.action_space.seed(42)   

#base observation check, should be all zero
#print(env.reset())
for episode in range(10):
    print(f"Episode no. {episode}")
    observation = env.reset()
    #chosenAction = env.action_space.sample()
    chosenAction = [1,0.5,0.5,0.1,2,0.5,10,5,5]
    for _ in range(100):
        observation, reward, terminated, truncated = env.step(chosenAction)
        #print(f"observation is {observation}")
        if terminated or truncated:
            print("trigger")
            observation= env.reset()

env.close()


# Implement the pid functions so it trys to fly to its target