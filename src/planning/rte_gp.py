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

args = { 
        'directions': 40, # Desired mini_archive directions
        'ucb_const': 0.05, # Exploration vs exploitation of GP model
        'gp_opt_hp': False,
        'archive_prob': 0.0, # probability of selecting from mini_archive
        'blocked_legs': [],
        'gui': False,
        'visualization_speed': 2.0,
        'search_size': 300, #Pick top search_size number of controllers from the archives
        'kernel_l': 1.0,
        'kernel_var': 0.1, #Angle range to search for high performing behavior  
        'lateral_friction': 1.0,
        'archives_path': "./data/hexapod_prior",
        'mapelites_generations': 5000,
        'ablation': None} # "only_ucb", "only_learning"

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

def view_map(obstacles, boundary, start, goal, res=0.5):
    '''
    Format for input: 
    Rectangular obstacles [xmin, xmax, y min, ymax]
    obstacles = [[-4, -1, -3, 10], [2.5, 5.0,-8,5]] 
    boundary = [-10, 10, -10, 10] # [xmin, xmax, ymin, ymax]
    start, goal = [-7,0], [7,5]
    '''
    planner = A_star(obstacles, boundary, resolution=res)
    path = planner.plan(start, goal)
    print ("Planned path: ", path)
    all_nodes = planner.get_all_nodes()

    plt.plot([d[0] for d in planner.obstacles], [d[1] for d in planner.obstacles], 'Dk')
    plt.plot(start[0], start[1], '*b', alpha=1.0, label='start', markersize=5)
    plt.plot(goal[0], goal[1], '*r', alpha=1.0, label='goal', markersize=5)

    '''
    x = []
    y = []
    for node in all_nodes:
        x.append(node.x)
        y.append(node.y)
        plt.plot(node.x, node.y, '.b')
        plt.pause(0.000000001)
    '''

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

    #genotype = np.expand_dims(genotype.to_numpy(),axis=0)
    #fit = np.expand_dims(fit.to_numpy(), axis=0)
    #desc = np.expand_dims(desc.to_numpy(),axis=0)

    genotype = genotype.to_numpy()
    fit = fit.to_numpy()
    desc = desc.to_numpy()

    print("gen fit desc  data shape: ", genotype.shape, fit.shape, desc.shape)
    
    return genotype, fit, desc

class GP_model():    
    def __init__(self, inputs, outputs, mean_functions,
                 opt_restarts=3, opt_hyperparams=True, noise_var=1e-3):
        self.dim_in = len(inputs[0])
        self.dim_out = len(outputs[0])

        assert len(mean_functions) == self.dim_out
        assert inputs.ndim == 2 # action/commanded xy
        assert outputs.ndim == 2 # predicted relative xy

        self.m = [] # list of GP models - 2 GP models- one for each output dim
        for i in range(self.dim_out):
            kernel = GPy.kern.RBF(input_dim=len(inputs[0]),
                                  variance=args['kernel_var'],
                                  lengthscale=args['kernel_l'])
            self.m.append(GPy.models.GPRegression(inputs, outputs[:, i].reshape(-1, 1),
                                                  kernel=kernel,
                                                  mean_function=mean_functions[i],
                                                  normalizer=False,
                                                  noise_var=noise_var))

            # optimization to fit the model
            if opt_hyperparams:
                self.m[i].optimize_restarts(num_restarts=opt_restarts)
                
    def predict(self, inputs):
        #assert inputs.ndim == 2
        mean = []
        var = []
        for model in self.m:
            mu, sig = model.predict(inputs)
            mean.append(mu.flatten())
            var.append(sig.flatten())
        return np.array(mean).transpose() , np.array(var).transpose()

    def getLikelihood(self, inputs, point):
        assert inputs.ndim == 2
        assert point.shape == (self.dim_out,)
        log_likelihoods = []
        for i, model in enumerate(self.m):
            mu, var = model.predict(inputs)
            log_lik = -0.5*np.log(2*np.pi*var) - np.power(mu - point[i], 2)/(2.*var+1e-10)
            log_likelihoods.append(log_lik.flatten())
        
        total_lik = np.sum(log_likelihoods, axis=0)
        return np.exp(total_lik)

    
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

# Constant is a GPy class - returns a linear mapping.
# This class creates a mapping where we want each skill/descriptor to have its own  correspoding x and y
# This mapping takes in a particular descriptor in the repertoire
# Outputs the predicted x
# Likeise the next mapping takes in a particular descriptor and outputs the predicted y.
class Custom_mean_x(Constant):
    def __init__(self, input_dim, output_dim, dictionary):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dictionary = dictionary
        super(Custom_mean_x, self).__init__(input_dim=input_dim, output_dim=output_dim)

    def f(self, X):
        return  np.array([self.dictionary[tuple(x)] for x in X])

class Custom_mean_y(Constant):
    def __init__(self, input_dim, output_dim, dictionary):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dictionary = dictionary
        super(Custom_mean_y, self).__init__(input_dim=input_dim, output_dim=output_dim)

    def f(self, X):
        return  np.array([self.dictionary[tuple(x)] for x in X])


def disable_leg_param(param, blocked_legs):
    pp = param.copy()
    for leg in blocked_legs:
        assert leg < 6 and leg >= 0
        # disable first joint
        pp[6*leg] = 0   
        pp[6*leg+1] = 0   
        pp[6*leg+2] = 0  
        
        # disable 2nd joint
        pp[6*leg+3] = 0   
        pp[6*leg+4] = 0   
        pp[6*leg+5] = 0   
    return pp



    
def estimate_posteriors_ucb(current_parameter, current_obs, expected_obs, leg_blocks, frictions, prior_map_index, prior_map_trials, prior_map_means):
    ucb = np.zeros(len(prior_map_means))
    t = np.sum(prior_map_trials)
    posterior =  np.zeros(len(prior_map_means))
    #Compute UCBs
    for i in range(len(prior_map_means)):
        ucb[i] = args["ucb_const"] * np.sqrt(2*np.log(t)/prior_map_trials[i])
        
    #Update means
    perf = np.exp(-3.0 * np.linalg.norm(current_obs-expected_obs))
    prior_map_means[prior_map_index] = ((prior_map_trials[prior_map_index]-1)*prior_map_means[prior_map_index] + perf)/prior_map_trials[prior_map_index]

    posterior = (np.array(prior_map_means) + np.array(ucb))/(np.array(prior_map_means) + np.array(ucb)).sum()
    # ticks = tuple([str(d) for d in leg_blocks])
    # plot_map_ucb(prior_map_means, ucb, ticks)
    return posterior 

def gaussian_likelihood(means, vars, point):
    #TODO: Need to test
    log_likelihoods = []
    for i in range(means.shape[1]):
        var = vars.transpose()[i]
        mu = means.transpose()[i]
        log_lik = -0.5*np.log(2*np.pi*var) - np.power(mu - point[i], 2)/(2. * var + 1e-10)
        log_likelihoods.append(log_lik.flatten())
    
    total_lik = np.sum(log_likelihoods, axis=0)
    return np.exp(total_lik)



env = HexapodMazeEnv()    

def main():

    #### LOG DATA ####
    log_data = True
    model_desc_log = []
    
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
    final_goal= np.array([-4-1, 2.3+4])

    obstacles_offset = obstacles + np.array([-0.7, 0.7, -0.7, 0.7])

    env.add_hexapod([0,0,0]+start+[0.15])
    add_obstacles(obstacles)
    add_goal(final_goal)
    #env.add_boundary(boundary)

    ################### SETUP MAP/ARCHIVE ##########################
    archive_filename = "src/planning/archive_example.dat"
    params, fit, desc = load_archive(archive_filename)
    desc = (desc*3) - 1.5 # rescale descriptors back

    ################## APPLY ROBOT DAMAGE - IF NEEDED ##############
    #for i, pp in enumerate(params):
    #    params[i] = disable_leg_param(pp, [2]) # disable leg 3
    
    ############# SETUP GP AND MEAN FUNCTIONS ###################
    dictionary_x = dict()
    dictionary_y = dict()
    for i, dd in enumerate(desc):
        dictionary_x[tuple(dd)] = [dd[0]]
        dictionary_y[tuple(dd)] = [dd[1]]
    mf_x = Custom_mean_x(2, 1, dictionary_x)
    mf_y = Custom_mean_y(2, 1, dictionary_y)
    mean_function = [mf_x.copy(), mf_y.copy()]

    executed_controllers = []
    executed_descriptors = []
    observations = [] #outcomes
    
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

    first =True
    sub_goal_reached = False
    while True:
        if sub_goal_reached == True:
            sub_goal_reached = False
            sub_goal = sub_goals.pop(0) if len(sub_goals) > 0 else final_goal
                
        bot_direction = env.get_robot_rotz()
        bot_pos = env.get_robot_xypos() #+ np.array(start)
        print("bot direction: ",np.rad2deg(bot_direction))
        print("bot pos: ",bot_pos)
        
        # plan path to subgoal
        planner = A_star(obstacles, boundary, resolution=0.05)
        path = planner.plan(bot_pos, sub_goal)
        goal = np.array(path[17]) if len(path) > 17 else sub_goal 
        relative_goal = relative_coordinate(goal, bot_direction, bot_pos)
        print("Relative goal: ", relative_goal)
        
        # SELECTION OF PRIMITIVE
        # select primitive from the repertoire with smallest difference to subgoal
        if first: 
            diff = np.power(desc-relative_goal,2).sum(axis=1)
            index = np.argmin(diff)
            print("Index of controller chosen: ", index)
            print("Local descriptor chosen:", desc[index])
            selected_params = params[index]
            first = False
            model_desc_log.append(desc)
        else: 
            # SELECTION OF PRIMITIVE
            # GET GP PREDICTION OF REPERTOIRE
            model_desc = []
            for i in range(desc.shape[0]):    
                mu, var = gp_model.predict(desc[i].reshape((1,-1)))
                model_desc.append(mu[0])
            model_desc = np.array(model_desc)
            model_desc_log.append(model_desc)
            # model desc is updated descriptor model of x and y 
            #print("Model desc archive shape: ", model_desc.shape)
            diff = np.power(model_desc-relative_goal,2).sum(axis=1)
            #diff = np.power(desc-relative_goal,2).sum(axis=1) # no learning
            index = np.argmin(diff)
            print("Index of controller chosen: ", index)
            print("Local original descriptor chosen:", desc[index])
            print("Model descriptor chosen:", model_desc[index])
            selected_params = params[index]
        
        # RUN PRIMITIVE
        cur_robot_xy = env.run_controller(selected_params)    
        #cur_robot_xy += np.array(start)
        relative_xy = relative_coordinate(cur_robot_xy, bot_direction, bot_pos)
        print("Executed - relative coordinate: ", relative_xy)
        
        ############# LEARN/UPDATE GP MODEL #############
        executed_controllers.append(selected_params)
        executed_descriptors.append(desc[index])
        observations.append(relative_xy)
        print("Executed desc shape: ", np.array(executed_descriptors).shape)
        print("Observations shape: ", np.array(observations).shape)
        gp_model = GP_model(np.array(executed_descriptors), 
                            np.array(observations), 
                            mean_functions=mean_function, 
                            opt_hyperparams=False)
        
        ########### CHECK IF GOAL/SUB GOAL IS REACHED ##########
        if np.linalg.norm(cur_robot_xy - final_goal) < 0.3:
            print ("Goal reached")
            break
        elif np.linalg.norm(cur_robot_xy - np.array(sub_goal)) < 1.0:
            sub_goal_reached = True
            print("Sub goal reached")
        
    model_desc_log = np.array(model_desc_log)

    if log_data:
        print("Logging and saving data")
        filename="model_repertoire_log_no_damage.npz"
        np.savez(filename, model_desc_log=model_desc_log)
        print("END")
        
    # visualize map
    #view_map(obstacles, boundary, start, goal, res=0.05)
    

main()
