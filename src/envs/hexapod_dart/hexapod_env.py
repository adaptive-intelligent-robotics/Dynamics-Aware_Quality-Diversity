import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART
import matplotlib.pyplot as plt

import os
import torch
import src.torch.pytorch_util as ptu

from src.envs.hexapod_dart.controllers.sin_controller import SinusoidController
from src.envs.hexapod_dart.controllers.mb_sin_controller import MBSinusoidController

# To record entire trajectory of a particular state/observation
# In this case - this will be the behvaiuoral descriptor
class DutyFactor(rd.Descriptor):
    def __init__(self,desc):
        rd.Descriptor.__init__(self,desc)
        self._states = []
        
    def __call__(self):
        if(self._simu.num_robots()>0):
            self._states.append(self._simu.robot(0).positions())
            
            
class HexapodEnv:
    
    def __init__(self, dynamics_model,
                 render=False,
                 record_state_action=False,
                 ctrl_freq=100,):

        self.render = render
        self.record_state_action = record_state_action
        self.urdf_path = "src/envs/hexapod_dart/robot_model/"
        self.ctrl_freq = ctrl_freq # in Hz
        self.sim_dt = 0.001

        self.dynamics_model = dynamics_model

    def update_dynamics_model(dynamics_model):
        self.dynamics_model = dynamics_model
        return 1
        
    def init_robot(self):
        # Load robot urdf and intialize robot 
        robot = rd.Robot(self.urdf_path+"hexapod_v2.urdf", "hexapod", False)
        robot.free_from_world([0,0,0.0,0,0,0.15]) # place robot slightly above the floor
        robot.set_actuator_types('servo') # THIS IS IMPORTANT
        robot.set_position_enforced(True)
        #print("actuator types: ",robot.actuator_types()) # default was torque
        return robot
    
    def simulate(self, ctrl, sim_time, robot, render=None, video_name=None):

        if render is None:
            render = self.render
            
        # clone robot
        grobot = robot.clone()
            
        #initialize controllers
        ctrl_freq = self.ctrl_freq # 100 Hz
        sinusoid_controller = SinusoidController(ctrl, False, ctrl_freq)
        stationary_controller = SinusoidController(np.zeros(36), False, ctrl_freq)

        # Create simulator object
        simu = rd.RobotDARTSimu(self.sim_dt) # 1000 Hz simulation freq - 0.001 dt
        simu.set_control_freq(ctrl_freq) # set control frequency of 100 Hz
        simu.set_collision_detector("bullet") # DART Collision Detector is the default but does not support cylinder shapes                                             
        #print("SIMULATION TIMESTEP",simu.timestep())
        
        # create graphics                                                          
        if render:
            print("INSIDE RENDER TRUE")
            graphics = rd.gui.Graphics(simu, rd.gui.GraphicsConfiguration())
            #graphics.look_at([0.,0.,5.],[0.,0.,0.], [0.,0.,1.]) # camera_pos, look_at, up_vector
            simu.set_graphics(graphics)

            if video_name is not None: 
                camera = rd.sensor.Camera(simu, graphics.magnum_app(), graphics.width(), graphics.height(), 30)
                simu.add_sensor(camera)
                simu.run(0.01)
                
                camera.camera().record(True, True)
                camera.record_video(video_name)
                camera.look_at([0.1,1.4,1.3], [0.1,0,0.])

            
        # add the robot and the floor
        simu.add_robot(grobot)
        #floor = simu.add_checkerboard_floor(20., 0.1, 1., np.zeros((6,1)), "floor")
        simu.add_checkerboard_floor(20., 0.1, 1., np.zeros((6,1)), "floor")

        # ADD CONTROLLERS TO ROBOT ONLY AFTER ADDING ROBOT TO SIMULATION
        # add stationary controller first to stabilise robot from drop
        grobot.add_controller(stationary_controller, 1.)
        stationary_controller.configure()
        
        # Add controller to the robot   
        grobot.add_controller(sinusoid_controller, 1.)
        sinusoid_controller.configure() # to set the ctrl parameters
        sinusoid_controller.activate(False)

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

        grobot.reset()
        
        states_recorded = np.array(sinusoid_controller.states)
        actions_recorded = np.array(sinusoid_controller.actions)

        #print("States recorded: ", states_recorded.shape)
        #print("Action recorded: ", actions_recorded.shape)
        
        return final_pos, states_recorded, actions_recorded 


    def desired_angle(self, in_x,in_y, batch=False):
        '''
        Compute desired yaw angle (rotationabotu z axis) of the robot given final position x-y
        '''
        x = in_x
        y = in_y
            
        B = np.sqrt((x/2)**2 + (y/2)**2)
        alpha = np.arctan2(y,x)
        A = B/np.cos(alpha)
        beta = np.arctan2(y,x-A)

        if batch:
            for i in range(x.shape[0]):
                if x[i] < 0:
                    beta[i] = beta[i] - np.pi
                while beta[i] < -np.pi:
                    beta[i] = beta[i] + 2*np.pi
                while beta[i] > np.pi:
                    beta[i] = beta[i] - 2*np.pi
        else: 
            if x < 0:
                beta = beta - np.pi
            # if angles out of 360 range, bring it back in
            while beta < -np.pi:
                beta = beta + 2*np.pi
            while beta > np.pi:
                beta = beta - 2*np.pi

        return beta

    def angle_dist(self, a,b, batch=False):
        dist = b-a
        if batch:
            for i in range(a.shape[0]):
                while dist[i] < -np.pi:
                    dist[i] = dist[i] + 2*np.pi
                while dist[i] > np.pi:
                    dist[i] = dist[i] - 2*np.pi
        # if the angles out of the 360 range, bring it back in
        else:
            while dist < -np.pi:
                dist = dist + 2*np.pi
            while dist > np.pi:
                dist = dist - 2*np.pi
                
        return dist 
            
    def evaluate_solution(self, ctrl, render=False):

        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record = self.simulate(ctrl, sim_time, robot, render)
        
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
        beta = self.desired_angle(x_pos, y_pos)
        final_rot_z = final_pos[2] # final yaw angle of the robot 
        dist_metric = abs(self.angle_dist(beta, final_rot_z))
        
        fitness = -dist_metric

        if render:
            print("Desc from simulation", desc) 
        
        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None
    
        return fitness, desc, obs_traj, act_traj

    # for NN timestep model
    def simulate_model(self, ctrl, sim_time, mean, det):
        states_recorded = []
        actions_recorded = []

        #controller = MBSinusoidController(ctrl, False, self.ctrl_freq)
        controller = SinusoidController(ctrl, False, self.ctrl_freq)
        controller.configure()

        # initial state - everything zero except z position (have to make it correspond to what the model is trained on) 
        state = np.zeros(48)
        state[5] = -0.014 # robot com height when feet on ground is 0.136m 
        
        for t in np.arange(0.0, sim_time, 1/self.ctrl_freq):
            action = controller.commanded_jointpos(t)
            states_recorded.append(state)
            actions_recorded.append(action)        
            s = ptu.from_numpy(state)
            a = ptu.from_numpy(action)
            s = s.view(1,-1)
            a = a.view(1,-1)

            if det:
                # if deterministic dynamics model
                pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1))
            else:
                # if probalistic dynamics model - choose output mean or sample
                pred_delta_ns = self.dynamics_model.output_pred(torch.cat((s, a), dim=-1), mean=mean)
                
            #print(state.shape)
            #print(pred_delta_ns)            
            state = pred_delta_ns[0] + state # the [0] just seelect the row [1,state_dim]

        final_pos = state 
        states_recorded = np.array(states_recorded)
        actions_recorded = np.array(actions_recorded)

        return final_pos, states_recorded, actions_recorded
    
    def evaluate_solution_model(self, ctrl, mean=False, det=True):

        sim_time = 3.0
        final_pos, states_rec, actions_rec = self.simulate_model(ctrl, sim_time, mean, det)

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
        beta = self.desired_angle(x_pos, y_pos)
        final_rot_z = final_pos[2] # final yaw angle of the robot 
        dist_metric = abs(self.angle_dist(beta, final_rot_z))
        
        fitness = -dist_metric
        obs_traj = states_rec
        act_traj = actions_rec

        disagr = 0 
        return fitness, desc, obs_traj, act_traj, disagr
    
    def simulate_model_ensemble(self, ctrl, sim_time, mean, disagr):
        states_recorded = []
        actions_recorded = []
        model_disagr = []
        
        #controller = MBSinusoidController(ctrl, False, self.ctrl_freq)
        controller = SinusoidController(ctrl, False, self.ctrl_freq)
        controller.configure()
        # initial state
        # for this ensembles - the states need to be fed in the form of ensemble_size
        # state and actions fed in as [ensemble_size, dim_size]
        # ts expand and flatten takes care of the num particles/
        state = np.zeros(48)
        state[5] = -0.014 # robot com height when feet on ground is 0.136m 
        state = np.tile(state,(self.dynamics_model.ensemble_size, 1))
        
        for t in np.arange(0.0, sim_time, 1/self.ctrl_freq):
            action = controller.commanded_jointpos(t)
            states_recorded.append(state)
            actions_recorded.append(action)
            s = ptu.from_numpy(state)
            a = ptu.from_numpy(action)

            a = a.repeat(self.dynamics_model.ensemble_size,1)
            #print("s shape: ", state.shape)
            #print("a shape:", a.shape)
            
            # if probalistic dynamics model - choose output mean or sample
            if disagr:
                pred_delta_ns, _ = self.dynamics_model.sample_with_disagreement(torch.cat((self.dynamics_model._expand_to_ts_form(s), self.dynamics_model._expand_to_ts_form(a)), dim=-1))
                pred_delta_ns = ptu.get_numpy(pred_delta_ns)
                disagreement = self.compute_abs_disagreement(state, pred_delta_ns)
                #print("Disagreement: ", disagreement.shape)
                disagreement = ptu.get_numpy(disagreement) 
                #disagreement = ptu.get_numpy(disagreement[0,3]) 
                #disagreement = ptu.get_numpy(torch.mean(disagreement)) 
                model_disagr.append(disagreement)
                
            else:
                pred_delta_ns = self.dynamics_model.output_pred_ts_ensemble(s,a, mean=mean)
                
            #print("Samples: ", pred_delta_ns.shape)
            #print(state.shape)
            #print(pred_delta_ns.shape)            
            state = pred_delta_ns + state 

        final_pos = state[0] # for now just pick one model - but you have all models here
        states_recorded = np.array(states_recorded)
        actions_recorded = np.array(actions_recorded)
        model_disagr = np.array(model_disagr)
        
        return final_pos, states_recorded, actions_recorded, model_disagr

    
    def compute_abs_disagreement(self, cur_state, pred_delta_ns):
        '''
        Computes absolute state dsiagreement between models in the ensemble
        cur state is [4,48]
        pred delta ns [4,48]
        '''
        next_state = pred_delta_ns + cur_state
        next_state = ptu.from_numpy(next_state)
        mean = next_state

        sample=False
        if sample: 
            inds = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
            inds_b = torch.randint(0, mean.shape[0], next_state.shape[:1]) #[4]
            inds_b[inds == inds_b] = torch.fmod(inds_b[inds == inds_b] + 1, mean.shape[0]) 
        else:
            inds = torch.tensor(np.array([0,0,0,1,1,2]))
            inds_b = torch.tensor(np.array([1,2,3,2,3,3]))

        # Repeat for multiplication
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, mean.shape[1])
        inds_b = inds_b.unsqueeze(dim=-1).to(device=ptu.device)
        inds_b = inds_b.repeat(1, mean.shape[1])

        means_a = (inds == 0).float() * mean[0]
        means_b = (inds_b == 0).float() * mean[0]
        for i in range(1, mean.shape[0]):
            means_a += (inds == i).float() * mean[i]
            means_b += (inds_b == i).float() * mean[i]
            
        disagreements = torch.mean(torch.sqrt((means_a - means_b)**2), dim=-2, keepdim=True)
        #disagreements = torch.mean((means_a - means_b) ** 2, dim=-1, keepdim=True)

        return disagreements
    
    def evaluate_solution_model_ensemble(self, ctrl, mean=True, disagreement=True):
        torch.set_num_threads(1)
        sim_time = 3.0
        final_pos, states_rec, actions_rec, disagr = self.simulate_model_ensemble(ctrl, sim_time, mean, disagreement)

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
        beta = self.desired_angle(x_pos, y_pos)
        final_rot_z = final_pos[2] # final yaw angle of the robot 
        dist_metric = abs(self.angle_dist(beta, final_rot_z))
        
        fitness = -dist_metric

        obs_traj = states_rec
        act_traj = actions_rec

        #------------ Absolute disagreement --------------#
        # disagr is the abs disagreement trajectory for each dimension [300,1,48]
        # can also save the entire disagreement trajectory - but we will take final mean dis
        final_disagr = np.mean(disagr[-1,0,:])
        
        if disagreement:
            return fitness, desc, obs_traj, act_traj, final_disagr
        else: 
            return fitness, desc, obs_traj, act_traj


    def evaluate_solution_uni(self, ctrl, render=False, video_name=None):
        '''
        unidirectional task - multiple ways of walking forward
        BD - orientation of CoM w.r.t. threshold
        fitness - distance in the forward direction
        '''
        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record = self.simulate(ctrl, sim_time, robot, render, video_name)
        
        #--------Compute BD (orientation)-----------#
        orn_threshold = 0.005*np.pi
        com_orn_traj = s_record[:,0:3] # at ctrl freq
        rot_x_traj = com_orn_traj[0:-1:5,0]
        rot_y_traj = com_orn_traj[0:-1:5,1]
        rot_z_traj = com_orn_traj[0:-1:5,2]

        bd1 = np.mean(np.heaviside(rot_x_traj - orn_threshold, 0))
        bd2 = np.mean(np.heaviside(-rot_x_traj - orn_threshold, 0))
        bd3 = np.mean(np.heaviside(rot_y_traj - orn_threshold, 0))
        bd4 = np.mean(np.heaviside(-rot_y_traj - orn_threshold, 0))
        bd5 = np.mean(np.heaviside(rot_z_traj - orn_threshold, 0))
        bd6 = np.mean(np.heaviside(-rot_z_traj - orn_threshold, 0))

        desc = [[bd1, bd2, bd3, bd4, bd5, bd6]]
        
        #------------Compute Fitness-------------------#
        fitness = final_pos[3]

        if render:
            print("Desc from simulation", desc)
        
        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None
    
        return fitness, desc, obs_traj, act_traj

    def evaluate_solution_model_uni(self, ctrl, render=False, mean=False, det=True):
        '''
        unidirectional task - multiple ways of walking forward
        BD - orientation of CoM w.r.t. threshold
        fitness - distance in the forward direction
        '''
        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record = self.simulate_model(ctrl, sim_time, mean, det)
        
        #--------Compute BD (orientation)-----------#
        orn_threshold = 0.005*np.pi
        com_orn_traj = s_record[:,0:3] # at ctrl freq
        rot_x_traj = com_orn_traj[0:-1:5,0]
        rot_y_traj = com_orn_traj[0:-1:5,1]
        rot_z_traj = com_orn_traj[0:-1:5,2]
        
        bd1 = np.mean(np.heaviside(rot_x_traj - orn_threshold, 0))
        bd2 = np.mean(np.heaviside(-rot_x_traj - orn_threshold, 0))
        bd3 = np.mean(np.heaviside(rot_y_traj - orn_threshold, 0))
        bd4 = np.mean(np.heaviside(-rot_y_traj - orn_threshold, 0))
        bd5 = np.mean(np.heaviside(rot_z_traj - orn_threshold, 0))
        bd6 = np.mean(np.heaviside(-rot_z_traj - orn_threshold, 0))

        desc = [[bd1, bd2, bd3, bd4, bd5, bd6]]
        #print(desc)
        
        #------------Compute Fitness-------------------#
        fitness = final_pos[3]

        if render:
            print("Desc from simulation", desc) 
        
        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None
    
        return fitness, desc, obs_traj, act_traj

    
    def evaluate_solution_model_ensemble_uni(self, ctrl, render=False, mean=True, disagreement=True):
        '''
        unidirectional task - multiple ways of walking forward
        BD - orientation of CoM w.r.t. threshold
        fitness - distance in the forward direction
        '''
        torch.set_num_threads(1)
        robot = self.init_robot()
        sim_time = 3.0
        final_pos, s_record, a_record, disagr = self.simulate_model_ensemble(ctrl, sim_time, mean, disagreement)

        #print("s record shape: ", s_record.shape)
        #print("a record shape: ", a_record.shape)
        s_record = s_record[:,0,:]
        
        #--------Compute BD (orientation)-----------#
        orn_threshold = 0.005*np.pi
        com_orn_traj = s_record[:,0:3] # at ctrl freq
        rot_x_traj = com_orn_traj[0:-1:5,0]
        rot_y_traj = com_orn_traj[0:-1:5,1]
        rot_z_traj = com_orn_traj[0:-1:5,2]
        
        bd1 = np.mean(np.heaviside(rot_x_traj - orn_threshold, 0))
        bd2 = np.mean(np.heaviside(-rot_x_traj - orn_threshold, 0))
        bd3 = np.mean(np.heaviside(rot_y_traj - orn_threshold, 0))
        bd4 = np.mean(np.heaviside(-rot_y_traj - orn_threshold, 0))
        bd5 = np.mean(np.heaviside(rot_z_traj - orn_threshold, 0))
        bd6 = np.mean(np.heaviside(-rot_z_traj - orn_threshold, 0))

        desc = [[bd1, bd2, bd3, bd4, bd5, bd6]]
        #print(desc)
        
        #------------Compute Fitness-------------------#
        fitness = final_pos[3]

        #------------ Absolute disagreement --------------#
        # disagr is the abs disagreement trajectory for each dimension [300,1,48]
        # can also save the entire disagreement trajectory - but we will take final mean dis
        final_disagr = np.mean(disagr[-1,0,:])

        if self.record_state_action: 
            obs_traj = s_record
            act_traj = a_record
        else:
            obs_traj = None
            act_traj = None

        if disagreement:
            return fitness, desc, obs_traj, act_traj, final_disagr
        else: 
            return fitness, desc, obs_traj, act_traj














    

        
def plot_state_comparison(state_traj, state_traj_m):

    total_t = state_traj.shape[0]
    #total_t = 30
    
    for i in np.arange(3,4):
        traj_real = state_traj[:,i]
        traj_m = state_traj_m[:,i]
        plt.plot(np.arange(total_t), traj_real[:total_t], "-",label="Ground truth "+str(i))
        plt.plot(np.arange(total_t), traj_m[:total_t], '--', label="Dynamics Model "+str(i))

    return 1

if __name__ == "__main__":

    # initialize environment class
    from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
    from src.models.dynamics_models.deterministic_model import DeterministicDynModel

    variant = dict(
        mbrl_kwargs=dict(
            ensemble_size=4,
            layer_size=500,
            learning_rate=1e-3,
            batch_size=256,
        )
    )

    obs_dim = 48
    action_dim = 18
    M = variant['mbrl_kwargs']['layer_size']

    # initialize dynamics model
    prob_dynamics_model = ProbabilisticEnsemble(
        ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M]
    )

    # initialize dynamics model
    det_dynamics_model = DeterministicDynModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=500
    )

    model_path = "src/dynamics_model_analysis/trained_models/prob_ensemble_finalarch.pth"
    ptu.load_model(prob_dynamics_model, model_path)
    
    env = HexapodEnv(prob_dynamics_model, render=False, record_state_action=True)

    # test simulation with a random controller
    ctrl = np.random.uniform(0,1,size=36)

    fit, desc, obs_traj, act_traj = env.evaluate_solution(ctrl, render=False)
    fit_m, desc_m, obs_traj_m, act_traj_m = env.evaluate_solution_model_ensemble(ctrl, det=False, mean=True)
    print("Ground Truth: ", fit, desc)
    print("Probablistic Model: ", fit_m, desc_m)

    print(obs_traj.shape)
    print(obs_traj_m.shape)
    plot_state_comparison(obs_traj, obs_traj_m[:,0,:])
    plot_state_comparison(obs_traj, obs_traj_m[:,1,:])
    plot_state_comparison(obs_traj, obs_traj_m[:,2,:])
    plot_state_comparison(obs_traj, obs_traj_m[:,3,:])

    #plt.show()
