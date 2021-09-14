import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART

import os

from src.envs.hexapod_dart.controllers.sin_controller import SinusoidController

# To record entire trajectory of a particular state/observation
# In this case - this will be the behvaiuoral descriptor
class DutyFactor(rd.Descriptor):
    def __init__(self,desc):
        rd.Descriptor.__init__(self,desc)
        self._states = []
        
    def __call__(self):
        if(self._simu.num_robots()>0):
            self._states.append(self._simu.robot(0).positions())

            
robot = rd.Robot("src/envs/hexapod_dart/robot_model/hexapod_v2.urdf", "hexapod", False)
robot.free_from_world([0,0,0.0,0,0,0.25]) # place robot slightly above the floor
robot.set_actuator_types('servo') # THIS IS IMPORTANT
#print("actuator types: ",robot.actuator_types()) # default was torque

def simulate(ctrl, sim_time, render=False):

    #print("ctrl in simulate function", ctrl)

    # clone robot
    grobot = robot.clone()

    #initialize controllers
    ctrl_freq = 1000
    sinusoid_controller = SinusoidController(ctrl, False, ctrl_freq)
    stationary_controller = SinusoidController(np.zeros(36), False, ctrl_freq)
     
    # add stationary controller first to stabilise robot from drop
    grobot.add_controller(stationary_controller, 1.)
    stationary_controller.configure()

    # Add controller to the robot   
    grobot.add_controller(sinusoid_controller, 1.)
    sinusoid_controller.configure() # to set the ctrl parameters
    sinusoid_controller.activate(False)
    
    # Create simulator object
    simu = rd.RobotDARTSimu(0.001)
    simu.set_collision_detector("bullet") # DART Collision Detector is the default but does not support cylinder shapes                                             
    #print("SIMULATION TIMESTEP",simu.timestep())

    # create graphics                                                          
    if render:
        print("INSIDE RENDER TRUE")
        graphics = rd.gui.Graphics(simu, rd.gui.GraphicsConfiguration())
        #graphics.look_at([0.,0.,5.],[0.,0.,0.], [0.,0.,1.]) # camera_pos, look_at, up_vector
        simu.set_graphics(graphics)
    
    # add the robot and the floor
    simu.add_robot(grobot)
    #floor = simu.add_checkerboard_floor(20., 0.1, 1., np.zeros((6,1)), "floor")
    simu.add_checkerboard_floor(20., 0.1, 1., np.zeros((6,1)), "floor")

    # set friction parameters - only works with latest release of robot_dart
    #mu = 0.7
    #grobot.set_friction_coeffs(mu)
    #floor.set_friction_coeffs(mu)
    
    simu.run(1.0)

    # Change control mode from stabalise to walking
    stationary_controller.activate(False)
    sinusoid_controller.activate(True)
    
    # run
    simu.run(sim_time)

    #print("DONE SIMU")
    final_pos = grobot.positions() # [rx,ry,rz,x,y,z,joint_pos]

    if render:
        print("Final position by simu: " ,final_pos)
    
    return final_pos 



def desired_angle(in_x,in_y):
    '''
    Compute desired yaw angle (rotationabotu z axis) of the robot given final position x-y
    '''
    x = in_x
    y = in_y
    
    B = np.sqrt((x/2)**2 + (y/2)**2)
    alpha = np.arctan2(y,x)
    A = B/np.cos(alpha)
    beta = np.arctan2(y,x-A)

    if x < 0:
        beta = beta - np.pi

    # if angles out of 360 range, bring it back in
    while beta < -np.pi:
        beta = beta + 2*np.pi
    while beta > np.pi:
        beta = beta - 2*np.pi

    return beta

def angle_dist(a,b):

    dist = b-a

    # if the angles out of the 360 range, bring it back in
    while dist < -np.pi:
        dist = dist + 2*np.pi
    while dist > np.pi:
        dist = dist - 2*np.pi
        
    return dist 

def evaluate_solution(ctrl, render=False):

    #print("ctrl in evaluate solution", ctrl)
    sim_time = 5.0
    final_pos = simulate(ctrl, sim_time, render)

    #--------Compute BD (final x-y pos)-----------#
    x_pos = final_pos[3]
    y_pos = final_pos[4]

    # normalize BD
    offset = 1.5 # in this case the offset is the saem for both x and y descriptors
    fullmap_size = 3 # 8m for full map size 
    x_desc = (x_pos + offset)/fullmap_size
    y_desc = (y_pos + offset)/fullmap_size

    desc = [[x_desc,y_desc]]

    #------------Compute Fitness-------------------#
    beta = desired_angle(x_pos, y_pos)
    final_rot_z = final_pos[2] # final yaw angle of the robot 
    dist_metric = abs(angle_dist(beta, final_rot_z))

    fitness = -dist_metric

    #print("DONE evaluate_solution")

    obs_traj = None
    act_traj = None

    return fitness, desc, obs_traj, act_traj




    

