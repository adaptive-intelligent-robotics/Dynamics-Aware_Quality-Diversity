import RobotDART as rd
import dartpy

from pathlib import Path
import os 
import sys
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(this_file_path).parent))
import numpy as np
import math
from GPy.mappings import Constant
import GPy

import matplotlib.pyplot as plt
import argparse
import glob
import os
import matplotlib.pyplot as plt

from src.planning.a_star import A_star 
import json

import pandas as pd

from src.envs.hexapod_dart.controllers.sin_controller import SinusoidController

#optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("--archive_path", 
                    help='Path to MAP-elites archives', 
                    type=str)
arguments = parser.parse_args()

sim_time = 3.0

class HexapodMazeEnv():

    def __init__(self):

        self.urdf_path = "src/envs/hexapod_dart/robot_model/"
        # Create simulator object
        self.ctrl_freq = 100 # 100 Hz
        self.sim_freq = 1000.
        self.simu = rd.RobotDARTSimu(1/self.sim_freq) # simulatiion frequency
        self.simu.set_control_freq(self.ctrl_freq) # set control frequency of 100 Hz
        self.simu.set_collision_detector("bullet")

        #self.sinusoid_controller = SinusoidController(ctrl, False, ctrl_freq)
        self.stationary_controller = SinusoidController(np.zeros(36), False, self.ctrl_freq)
        self.sinusoid_controller = SinusoidController(np.zeros(36), False, self.ctrl_freq)
        render=True
        if render: 
            # Create graphics
            graphics = rd.gui.Graphics(self.simu, rd.gui.GraphicsConfiguration())
            self.simu.set_graphics(graphics)

        # Add robot and floor to the simulation
        self.simu.add_checkerboard_floor(20.0)        

        self.grobot = None
    
    def reset_controller(self, ctrl):
        #self.sinusoid_controller = SinusoidController(ctrl, False, self.ctrl_freq)
        return 0
    
    def init_hexapod(self):
        # Load robot urdf and intialize robot
        robot = rd.Robot(self.urdf_path+"hexapod_v2.urdf", "hexapod", False)
        robot.set_actuator_types('servo') # THIS IS IMPORTANT
        robot.set_position_enforced(True)
        robot.skeleton().setPosition(5, 0.15);
        #robot.free_from_world([0,0,0.0,0,0,0.15]) # BE CAREFUL - this function sets a new reference frame. getPositions will give you position w.r.t to this defined coord. goof id you wnat to get relative tranformation.
        #print("actuator types: ",robot.actuator_types()) # default was torque
        return robot

    def add_hexapod(self, init_pose):
        robot = self.init_hexapod()
        #robot.free_from_world(init_pose) # place robot according to start position - possible to use this but have to implement differently.
        self.grobot = robot.clone()
        for i in range(len(init_pose)):
            self.grobot.skeleton().setPosition(i, init_pose[i]);
        self.simu.add_robot(self.grobot)
        self.grobot.set_draw_axis("base_link")
        self.grobot.add_controller(self.stationary_controller, 1.)
        self.grobot.add_controller(self.sinusoid_controller, 1.)

    def add_box(self, box_dims, box_pose):        
        robot = rd.Robot.create_box(box_dims, box_pose, mass=50.0)
        self.simu.add_robot(robot)    

    def add_goal(self, pos):
        #xyz posiition of goal
        # have to change COLOR OF THE goal but dont know how atm - APi does not like it
        robot = rd.Robot.create_box([0.1, 0.1, 0.1], [0,0,0]+pos, mass=0.01, color=[0,1,0,1])
        robot.set_ghost()
        self.simu.add_robot(robot)    

    def add_boundary(self, boundary):
        #[xmin, xmax, ymin, ymax]
        #x portions of boundary - lines along/parllel tp x axis
        xbox_dim = [(boundary[1]-boundary[0]), 0.1, 0.2]
        xbox_pose1 = [0,0,0,(boundary[0]+boundary[1])/2.0, boundary[2]-0.05, 0.5]
        xbox_pose2 = [0,0,0,(boundary[0]+boundary[1])/2.0, boundary[3]+0.05, 0.5]
        #y portions of bouundary
        ybox_dim = [0.1, (boundary[3]-boundary[2]), 0.2]
        ybox_pose1 = [0,0,0,boundary[0]-0.05, (boundary[2]+boundary[3])/2.0, 0.5]
        ybox_pose2 = [0,0,0,boundary[1]+0.05, (boundary[2]+boundary[3])/2.0, 0.5]
        

        col = [0.3,0.3,0.3,1]
        x_bottom = rd.Robot.create_box(xbox_dim, xbox_pose1, mass=10.0, color=col)
        x_top = rd.Robot.create_box(xbox_dim, xbox_pose2, mass=10.0, color=col)
        y_left = rd.Robot.create_box(ybox_dim, ybox_pose1, mass=10.0, color=col)
        y_right = rd.Robot.create_box(ybox_dim, ybox_pose2, mass=10.0, color=col)

        self.simu.add_robot(x_bottom)
        self.simu.add_robot(x_top)
        self.simu.add_robot(y_left)
        self.simu.add_robot(y_right)    

    def run_controller(self, ctrl, sim_time=3.0):
        # ADD CONTROLLERS TO ROBOT ONLY AFTER ADDING ROBOT TO SIMULATION
        # add stationary controller first to stabilise robot from drop
        self.sinusoid_controller.set_ctrl(ctrl)
        #self.grobot.add_controller(self.stationary_controller, 1.)
        #self.stationary_controller.configure()
        #self.grobot.add_controller(self.sinusoid_controller, 1.)
        #self.sinusoid_controller.configure() # to set the ctrl parameters
        self.sinusoid_controller.activate(False)
        self.simu.run(0.5)

        # Change control mode from stabalise to walking
        self.stationary_controller.activate(False)
        self.sinusoid_controller.activate(True)

        # run
        self.simu.run(sim_time)

        # Change control mode from stabalise to walking
        self.stationary_controller.activate(True)
        self.sinusoid_controller.activate(False)
        self.simu.run(0.5)

        xy_pos = self.get_robot_xypos()
        return xy_pos
        
    def get_robot_xypos(self):
        pos = self.grobot.positions() # [rx,ry,rz,x,y,z,joint_pos]
        #pos = self.grobot.skeleton().getPositions() # [rx,ry,rz,x,y,z,joint_pos]  
        return np.array(pos[3:5])

    def get_robot_rotz(self):
        pos = self.grobot.positions() # [rx,ry,rz,x,y,z,joint_pos]
        return pos[2]
    
def add_obstacles(obstacles):
    for i, obstacle in enumerate(obstacles):
        box_dims = [(obstacle[1]-obstacle[0])/1.0, (obstacle[3]-obstacle[2])/1.0, 0.2]
        box_pose = [0,0,0,(obstacle[0]+obstacle[1])/2.0, (obstacle[2]+obstacle[3])/2.0, 0.5]
        env.add_box(box_dims, box_pose)
        
def add_goal(goal):
    pos = [goal[0], goal[1], 0.05]
    env.add_goal(pos)

def view_map(obstacles, boundary, start, goal):
    '''
    Format for input: 
    Rectangular obstacles [xmin, xmax, y min, ymax]
    obstacles = [[-4, -1, -3, 10], [2.5, 5.0,-8,5]] 
    boundary = [-10, 10, -10, 10] # [xmin, xmax, ymin, ymax]
    start, goal = [-7,0], [7,5]
    '''
    planner = A_star(obstacles, boundary, resolution=0.5)
    path = planner.plan(start, goal)
    print ("Planned path: ", path)
    all_nodes = planner.get_all_nodes()

    plt.plot([d[0] for d in planner.obstacles], [d[1] for d in planner.obstacles], 'Dk')
    plt.plot(start[0], start[1], '*b', alpha=1.0, label='start', markersize=12)
    plt.plot(goal[0], goal[1], '*r', alpha=1.0, label='goal', markersize=12)

    x = []
    y = []
    for node in all_nodes:
        x.append(node.x)
        y.append(node.y)
        plt.plot(node.x, node.y, '.b')
        plt.pause(0.000000001)

    plt.plot([d[0] for d in path], [d[1] for d in path], '-', linewidth=3)
    plt.pause(0.000000001)
    plt.show()

    
def load_archive(filename):
    # load in archive.dat file
    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma

    print("Archive file data shape: ", data.shape)
    # 41 columns fitness, bd1, bd2, bd_ground1, bdground2, genotype(36dim)
    genotype = data.iloc[:,-36:]
    fit = data.iloc[:,0]
    desc = data.iloc[:,1:3]

    genotype = genotype.to_numpy()
    fit = fit.to_numpy()
    desc = desc.to_numpy()

    return genotype, fit, desc

def convert_polar2desc(polar):
    r = polar[0]
    theta = polar[1]
    r_desc = (r/5.0) * 2.0 - 1.0
    theta_desc = theta/180.0
    return np.array([r_desc, theta_desc])

def convert_desc2polar(desc):
    r_desc = desc[0]
    theta_desc = desc[1]
    r = (r_desc + 1.0) * 0.5 * 5.0
    theta = theta_desc *180.0
    return np.array([r, theta])

def convert_polar2cart(polar):
    r = polar[0]
    theta = polar[1]
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def convert_cart2polar(cart):
    x = cart[0]
    y = cart[1]
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    return np.array([r,np.rad2deg(theta)])

def get_direction(robot_cm, robot_direction,  goal):
    cm2goal = goal - robot_cm
    cm2goal_polar = convert_cart2polar(cm2goal)

    theta = np.deg2rad(cm2goal_polar[1] - robot_direction)
    theta_reduced = np.arctan2(np.sin(theta),np.cos(theta))

    return np.rad2deg(theta_reduced)

def getFaceAngle():
    final_face_vec = [env_real.getRotation()[0], env_real.getRotation()[3]]
    theta = convert_cart2polar(final_face_vec)[1]
    return theta

def relative_coordinate(absolute_point, face_angle, cm):
    #World frame to robots frame
    #rm = np.array([[np.cos(np.deg2rad(face_angle)), np.sin(np.deg2rad(face_angle))],
    #                [-np.sin(np.deg2rad(face_angle)), np.cos(np.deg2rad(face_angle))]])

    rm = np.array([[np.cos(face_angle), np.sin(face_angle)],
                    [-np.sin(face_angle), np.cos(face_angle)]])

    point = absolute_point - cm
    return np.matmul(rm,point)


env = HexapodMazeEnv()    

def main():

    #################### SETUP ENVIRONMENT #####################
    
    # Environment settings - create obstacle course
    # Environment is set up by defining obstacles, boundary and start/goal
    '''
    # ENV TYPE 1
    obstacles = np.array([[-4, -1, -3, 10], [2.5, 5.0,-8,5]]) 
    boundary = [-9, 9, -9, 9] # [xmin, xmax, ymin, ymax]
    start, final_goal = [-7,0], np.array([7,5])
    #[[xmin, xmax, ymin, ymax], ...]
    '''
    
    # ENV - TYPE 2
    obstacles = np.array([[-6, -1, -1, 1.0], [-3, 1, 3, 4], [1, 1.5, -1.5, 8], 
                          [-6.5, 1, -1.5, -1.0], [-6.5, -6.0, -1, 8], [-6.5, 1.5, 8, 8.5]]) 
    boundary = [-9, 9, -9, 9]
    start = [0,0,]
    final_goal= np.array([-4+2, 2.3+3])
    

    obstacles_offset = obstacles + np.array([-0.8, 0.8, -0.8, 0.8])

    env.add_hexapod([0,0,0]+start+[0.15])
    add_obstacles(obstacles)
    add_goal(final_goal)
    #env.add_boundary(boundary)

    ################### SETUP MAP ##########################
    archive_filename = "src/planning/archive_example.dat"
    params, fit, desc = load_archive(archive_filename)
    desc = (desc*3) - 1.5 # rescale descriptors back
    
    ################# GET MAIN PATH TO GOAL ################
    bot_direction = env.get_robot_rotz()
    bot_pos = env.get_robot_xypos() #+ np.array(start)
    print("bot direction initial : ",np.rad2deg(bot_direction))
    print("bot pos initial : ",bot_pos)
        
    planner = A_star(obstacles_offset, boundary, resolution=0.05)
    main_path = planner.plan(bot_pos, final_goal) 
    sub_goals = main_path[0:len(main_path):20]
    print("Sub goals: ",sub_goals)
    sub_goals.append(final_goal.tolist())
    _ = sub_goals.pop(0) #Remove the initial position
    sub_goal = sub_goals.pop(0)

    sub_goal_reached = False
    while True:
        if sub_goal_reached == True:
            sub_goal_reached = False
            sub_goal = sub_goals.pop(0) if len(sub_goals) > 0 else final_goal
                
        bot_direction = env.get_robot_rotz()
        bot_pos = env.get_robot_xypos() #+ np.array(start)
        print("bot direction: ",np.rad2deg(bot_direction))
        print("bot pos: ",bot_pos)
        
        #plan path to subgoal
        planner = A_star(obstacles, boundary, resolution=0.05)
        path = planner.plan(bot_pos, sub_goal)
        goal = np.array(path[17]) if len(path) > 17 else sub_goal 
        relative_goal = relative_coordinate(goal, bot_direction, bot_pos)

        print("Relative goal: ", relative_goal)
        # select primitive from the repertoire with smallest difference to subgoal
        diff = np.power(desc-relative_goal,2).sum(axis=1)
        index = np.argmin(diff)
        print("Index of controller chosen: ", index)
        print("Local descriptor chosen:", desc[index])
        selected_params = params[index]
        

        cur_robot_xy = env.run_controller(selected_params)    
        #cur_robot_xy += np.array(start)
        if np.linalg.norm(cur_robot_xy - final_goal) < 0.3:
            print ("Goal reached")
            break
        elif np.linalg.norm(cur_robot_xy - np.array(sub_goal)) < 1.0:
            sub_goal_reached = True
            print("Sub goal reached")

        
    #ctrl = [1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0.5, 0.5, 0.25, 0.75, 0.5, 1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5, 1, 0.5, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5]
    #env.run_controller(ctrl)
    #env.run_controller(ctrl)
    #env.simu.run(15.0)

    # visualize map
    #view_map(obstacles, boundary, start, goal)
    
main()
