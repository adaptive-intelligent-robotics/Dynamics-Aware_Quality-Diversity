import os, inspect
import time

import pybullet
import pybullet_data as pd
from pybullet_utils import bullet_client as bc

import numpy as np
import math as m
import torch
import torch.nn as nn

from gym.utils import seeding

from src.envs.panda_bullet.utils import load_parameters
import src.torch.pytorch_util as ptu

#torch.set_num_threads(16)
#torch.set_num_interop_threads(16)
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

class pandaEnv:
    initial_positions = {
        'panda_joint1': 0.0, 'panda_joint2': -0.54, 'panda_joint3': 0.0,
        'panda_joint4': -2.6, 'panda_joint5': -0.30, 'panda_joint6': 2.0,
        'panda_joint7': 1.0, 'panda_finger_joint1': 0.00, 'panda_finger_joint2': 0.00,
    }

    def __init__(self, gui, use_IK=0,
                 base_position=(0.0, 0, 0.625), control_orientation=1, control_eu_or_quat=0,
                 joint_action_space=9, includeVelObs=False, controller_type=1, world_model_path=None):

        #import pybullet as op1
        #import pybullet_data as pd
        torch.set_num_threads(1) # each process can only use one thread
        #print("Number of threads torch is using: ", torch.get_num_threads())
        
        self.gui = gui

        if gui:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self._physics_client_id = self.p._client
        #print("client_id 123: ",  self.p._client)

        self.p.setAdditionalSearchPath(pd.getDataPath())

        self.p.resetDebugVisualizerCamera(2.1, 60, -40, [0.0, -0.0, -0.0], physicsClientId=self._physics_client_id) #(2.1, 90, -30, [0.0, -0.0, -0.0]) ori, 90 to be facing the robot flat
        self.p.resetSimulation(physicsClientId=self._physics_client_id)
        self.p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._physics_client_id)
        self.sim_timestep = 1.0/480. # 1.0/240 # 240Hz # higher better but more compexp
        self.p.setTimeStep(self.sim_timestep, physicsClientId=self._physics_client_id)
        self.p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)

        self.control_timestep = 0.1 # 10 Hz
        
        self._use_IK = use_IK # task space IK control or joint space control
        self._control_orientation = control_orientation # control orientation of end effector
        self._base_position = base_position # base position of the robot

        self.joint_action_space = joint_action_space # (7+2) 7 joints + 2 finger joints
        self._include_vel_obs = includeVelObs # decide to include end eff velocity as obs
        self._include_joint_pos_obs = True # decide to include joint pos and vel as obs 
        self._control_eu_or_quat = control_eu_or_quat # choose euler or quat for orntation

        self._workspace_lim = [[0.3, 0.65], [-0.3, 0.3], [0.60, 1.5]]
        self._eu_lim = [[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]]

        self.end_eff_idx = 11  # 8

        self._home_hand_pose = []

        self._num_dof = 7
        self._joint_name_to_ids = {}

        self.robot_id = None
        self.plane_id = None
        self.table_id = None

        self.seed()

        # for goals
        self.ee_goal_pos = None
        self.obj_goal_pos = None
            
        # Load robot and environment urdf
        self.reset() 
        
            
    def reset(self):

        # clear everything first and rest robot and objects
        # must reset or not you are just adding stuff in over each other
        self.p.resetSimulation(physicsClientId=self._physics_client_id)
        self.p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._physics_client_id)
        self.p.setTimeStep(self.sim_timestep, physicsClientId=self._physics_client_id)
        self.p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0, physicsClientId=self._physics_client_id)
        
        # Reset robot and all other objects in the environemtn
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.p.URDF_USE_INERTIA_FROM_FILE | self.p.URDF_USE_SELF_COLLISION
        
        # --- Load plane and table contained in pybullet_data --- #
        self.plane_id = self.p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
        self.table_id = self.p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], self.p.getQuaternionFromEuler([0,0,np.pi/2]), flags=flags, physicsClientId=self._physics_client_id) # table dim is 1.0m 1.5m,0.625m
        assert self.plane_id is not None, "Failed to load the plane (floor) model"
        assert self.table_id is not None, "Failed to load the table model"
        
        # sameple cube position from uniform distribution in world fram 
        x = 0.5 #np.random.uniform()*0.2 + 0.3
        y = 0.0 #np.random.uniform()*0.5 - 0.25
        z = 0.7
        cube_init_pos = [x, y, z]
        # --- Load other objects --- #
        #self.cube_id = self.p.loadURDF("cube_small.urdf", cube_init_pos, globalScaling=1.2, flags=flags, physicsClientId=self._physics_client_id)
        #self.cube_id = self.p.loadURDF("jenga/jenga.urdf", cube_init_pos, flags=flags, physicsClientId=self._physics_client_id)
        #self.block_id = p.loadURDF(os.path.join(pd.getDataPath(), "block.urdf"), [0.5, 0.1, 0.8], flags=flags, physicsClientId=self._physics_client_id)
        #self.tray_id = self.p.loadURDF("tray/tray.urdf", [0.5, 0., 0.63], physicsClientId=self._physics_client_id)
        #self.sphere_id = self.p.loadURDF("sphere_small.urdf",[0.5, 0.3, 1.0], physicsClientId=self._physics_client_id)
        self.cube_id = self.p.loadURDF("lego/lego.urdf", cube_init_pos, globalScaling=2, physicsClientId=self._physics_client_id)
        #self.mug_id = self.p.loadURDF("urdf/mug.urdf", cube_init_pos, globalScaling=2.5, physicsClientId=self._physics_client_id)
        
        #self.lego1_id = self.p.loadURDF("lego/lego.urdf", [0.5, -0.1, 0.85], globalScaling=2.0, physicsClientId=self._physics_client_id)
        #self.lego2_id = self.p.loadURDF("lego/lego.urdf", [0.5, 0.1, 0.85], globalScaling=2.0, physicsClientId=self._physics_client_id)

        # --- Load robot model --- #
        self.robot_id = self.p.loadURDF("franka_panda/panda.urdf",
                                   basePosition=self._base_position, useFixedBase=True, flags=flags,
                                   physicsClientId=self._physics_client_id)
        assert self.robot_id is not None, "Failed to load the panda model"

        '''
        # LINEAR AN ANGULAR DAMPING OPTIONS 
        for j in range(self.p.getNumJoints(self.robot_id)):
            self.p.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0)
        #self.p.changeDynamics(self.plane_id,-1)
        self.p.changeDynamics(self.table_id,-1, linearDamping=0, angularDamping=0)
        self.p.changeDynamics(self.cube_id,-1, linearDamping=0, angularDamping=0)
        '''
        
        # reset joints to home position
        num_joints = self.p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        idx = 0
        for i in range(num_joints):
            joint_info = self.p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if (joint_type == self.p.JOINT_REVOLUTE) or (joint_type == self.p.JOINT_PRISMATIC):
                assert joint_name in self.initial_positions.keys()

                self._joint_name_to_ids[joint_name] = i

                self.p.resetJointState(self.robot_id, i, self.initial_positions[joint_name], physicsClientId=self._physics_client_id)
                self.p.setJointMotorControl2(self.robot_id, i, self.p.POSITION_CONTROL,
                                        targetPosition=self.initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)
                idx += 1

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        if self._use_IK:
            self._home_hand_pose = [0.2, 0.0, 0.8,
                                    min(m.pi, max(-m.pi, m.pi)),
                                    min(m.pi, max(-m.pi, 0)),
                                    min(m.pi, max(-m.pi, 0))]

            self.apply_action(self._home_hand_pose)
            self.p.stepSimulation(physicsClientId=self._physics_client_id)

        # friction_anchor set in the panda urdf
        # contsraint to keep the fingers centered
        self.c = self.p.createConstraint(self.robot_id,
                                    9,
                                    self.robot_id,
                                    10,
                                    jointType=self.p.JOINT_GEAR,
                                    jointAxis=[1, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0], physicsClientId=self._physics_client_id)
        self.p.changeConstraint(self.c, gearRatio=-1, erp=0.1, maxForce=50, physicsClientId=self._physics_client_id)

        
        # get observation
        #obs = self.get_policy_obs()
    
        return 1

    
    def set_controller_params(self, params):
        # only works for nn - loads nn weights into a defined nn
        load_parameters(params, self.controller)
        #print("loading parameters", len(params))
        
    def setFriction(self, lateral_friction):
        self.p.changeDynamics(self.plane_id, linkIndex=-1, lateralFriction=lateral_friction, physicsClientId=self.physicsClient)
        
    def delete_simulated_robot(self):
        # Remove the robot from the simulation
        self.p.removeBody(self.robot_id, physicsClientId=self._physics_client_id)

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

        for joint_name in self._joint_name_to_ids.keys():
            jointInfo = self.p.getJointInfo(self.robot_id, self._joint_name_to_ids[joint_name], physicsClientId=self._physics_client_id)

            ll, ul = jointInfo[8:10]
            jr = ul - ll
            
            # For simplicity, assume resting state == initial state
            rp = self.initial_positions[joint_name]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)
            rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses

    def get_action_dim(self):
        """
        3 types of action modes: 
        1. Joint position control (7+2) 
        2. Task space control of end effector - x,y,z + orintation of endeff(euler or quaternion) 
        3. Task space control of end effector - just xyz
        """
        if not self._use_IK:
            print("DIRECT JOINT POSITION CONTROL: 7 + 2")
            return self.joint_action_space 

        if self._control_orientation and self._control_eu_or_quat == 0:
            return 6  # position x,y,z + roll/pitch/yaw of hand frame
        elif self._control_orientation and self._control_eu_or_quat == 1:
            return 7  # position x,y,z + quat of hand frame

        return 3  # position x,y,z

    def get_observation_dim(self):
        return len(self.get_observation())

    def get_workspace(self):
        return [i[:] for i in self._workspace_lim]

    def set_workspace(self, ws):
        self._workspace_lim = [i[:] for i in ws]

    def get_rotation_lim(self):
        return [i[:] for i in self._eu_lim]

    def set_rotation_lim(self, eu):
        self._eu_lim = [i[:] for i in eu]

    def get_observation(self):
        # Create observation state
        observation = []
        observation_lim = []

        # Get state of the end-effector link
        state = self.p.getLinkState(self.robot_id, self.end_eff_idx, computeLinkVelocity=1,
                               computeForwardKinematics=1, physicsClientId=self._physics_client_id)

        # POSITION AND ORIENTATION OF END EFFECTOR
        # in world frame!
        pos = state[0]
        orn = state[1]
        
        observation.extend(list(pos))
        observation_lim.extend(list(self._workspace_lim))
        if self._control_eu_or_quat == 0:
            euler = self.p.getEulerFromQuaternion(orn)
            observation.extend(list(euler))  # roll, pitch, yaw
            observation_lim.extend(self._eu_lim)
        else:
            observation.extend(list(orn)) # quaternion
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        # VELOCITY OF END EFFECTORY - OPTIONAL
        if self._include_vel_obs:
            # standardize by subtracting the mean and dividing by the std
            vel_std = [0.04, 0.07, 0.03]
            vel_mean = [0.0, 0.01, 0.0]
            vel_l = np.subtract(state[6], vel_mean)
            vel_l = np.divide(vel_l, vel_std)

            observation.extend(list(vel_l))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

        # JOINT POSITIONS and JOINT VELOCITIES
        if self._include_joint_pos_obs: 
            jointStates = self.p.getJointStates(self.robot_id, self._joint_name_to_ids.values(), physicsClientId=self._physics_client_id)
            #print(self._joint_name_to_ids.values())
            jointPoses = [x[0] for x in jointStates[:7]]
            jointVel = [x[1] for x in jointStates[:7]] # did not add to observation space yet
            #print(jointPoses)
            observation.extend(list(jointPoses))
            observation.extend(list(jointVel))
            
            observation_lim.extend([[self.ll[i], self.ul[i]] for i in range(0, len(self._joint_name_to_ids.values()))])

        #return observation, observation_lim
        return observation
    
    def get_obj_pose(self, obj_id):
        obj_states = self.p.getBasePositionAndOrientation(obj_id, physicsClientId=self._physics_client_id)
        pos = list(obj_states[0])
        orn = list(obj_states[1]) # quaternion

        if self._control_eu_or_quat == 0:
            orn = self.p.getEulerFromQuaternion(orn) # x,y,z rotation
        elif self._control_eu_or_quat == 1:
            orn = orn # quaternion

        # obj pose in world frame
        pose = np.array(pos+list(orn))
        #print("Pos orn: ",pos, orn)
        #print("Pose:", pose)
        return pose

    def get_policy_obs(self):

        observation=[]
        
        obs = self.get_observation()
        #print("observation end_eff pose: ", end_eff_pose)
        ee_pose = obs[0:6]
        joint_states = obs[6:] # joint positions and velocities
        observation.extend(ee_pose)
        #print("joint states length: ", len(joint_states))
        observation.extend(joint_states)
        #idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'],
        #               self._joint_name_to_ids['panda_finger_joint2']]
        #finger_state = self.p.getJointState(self.robot_id, idx_fingers[0], physicsClientId=self._physics_client_id)[0] # both fingers are tied to the same state
        #observation.extend([finger_state])
        
        cube_pose = self.get_obj_pose(self.cube_id) # xyz position
        observation.extend(cube_pose)
        
        # convert to robot frame
        observation = np.array(observation)
        #observation[0:3] = observation[0:3] - self._base_position # end eff position
        #observation[6:] = observation[7:] - self._base_position # cube position

        return observation
    
    def pre_grasp(self):
        self.apply_action_fingers([0.04, 0.04]) # open fingers fully

    def grasp(self, obj_id=None):
        self.apply_action_fingers([0.0, 0.0], obj_id)

    def apply_action_fingers(self, action, obj_id=None):
        # move finger joints in position control
        assert len(action) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))

        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        # use object id to check contact force and eventually stop the finger motion
        if obj_id is not None:
            _, forces = self.check_contact_fingertips(obj_id)
            # print("contact forces {}".format(forces))

            if forces[0] >= 20.0:
                action[0] = self.p.getJointState(self.robot_id, idx_fingers[0], physicsClientId=self._physics_client_id)[0]

            if forces[1] >= 20.0:
                action[1] = self.p.getJointState(self.robot_id, idx_fingers[1], physicsClientId=self._physics_client_id)[0]

        for i, idx in enumerate(idx_fingers):
            self.p.setJointMotorControl2(self.robot_id,
                                    idx,
                                    self.p.POSITION_CONTROL,
                                    targetPosition=action[i],
                                    force=10,
                                    maxVelocity=0.7,
                                    physicsClientId=self._physics_client_id)

    def apply_action(self, action, max_vel=-1, max_force=-1):

        if self._use_IK:
            # --- task space control using IK control --- #
            if not (len(action) == 3 or len(action) == 6 or len(action) == 7):
                raise AssertionError('number of action commands must be \n- 3: (dx,dy,dz)'
                                     '\n- 6: (dx,dy,dz,droll,dpitch,dyaw)'
                                     '\n- 7: (dx,dy,dz,qx,qy,qz,w)'
                                     '\ninstead it is: ', len(action))

            # --- Constraint end-effector pose inside the workspace --- #
            dx, dy, dz = action[:3]
            new_pos = [dx, dy,
                       min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

            # if orientation is not under control, keep it fixed
            if not self._control_orientation:
                new_quat_orn = self.p.getQuaternionFromEuler(self._home_hand_pose[3:6])

            # otherwise, if it is defined as euler angles
            elif len(action) == 6:
                droll, dpitch, dyaw = action[3:6]
                eu_orn = [min(2*m.pi, max(-m.pi, droll)),
                          min(m.pi, max(-m.pi, dpitch)),
                          min(m.pi, max(-m.pi, dyaw))]
                new_quat_orn = self.p.getQuaternionFromEuler(eu_orn)
                #print("HERE")
            # otherwise, if it is define as quaternion
            elif (len(action) == 7) and (self._control_eu_or_quat == 1):
                new_quat_orn = action[3:7]
            # otherwise, use current orientation
            else:
                new_quat_orn = self.p.getLinkState(self.robot_id, self.end_eff_idx, physicsClientId=self._physics_client_id)[5]

            # --- compute joint positions with IK --- #
            jointPoses = self.p.calculateInverseKinematics(self.robot_id, self.end_eff_idx, new_pos, new_quat_orn,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)

            if max_force == -1:
                max_force_app = np.array([87., 87. , 87. ,87.,12., 12., 12])
            else:
                max_force_app = np.ones(7)*max_force
                
            # --- set joint control --- #
            if max_vel == -1:
                self.p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                            jointIndices=self._joint_name_to_ids.values(),
                                            controlMode=self.p.POSITION_CONTROL,
                                            targetPositions=jointPoses,
                                            positionGains=[0.2] * len(jointPoses),
                                            velocityGains=[1.0] * len(jointPoses),
                                            physicsClientId=self._physics_client_id)
            else:
                for i in range(self._num_dof): # 7 joint positions only not fingers
                    self.p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                 jointIndex=i,
                                                 controlMode=self.p.POSITION_CONTROL,
                                                 targetPosition=jointPoses[i],
                                                 force=max_force_app[i],
                                                 maxVelocity=max_vel,
                                                 physicsClientId=self._physics_client_id)
                    
        else:
            # --- Direct Joint position control --- #
            assert len(action) == self.joint_action_space, ('number of motor commands differs from number of motor to control', len(action))

            joint_idxs = tuple(self._joint_name_to_ids.values())
            for i, val in enumerate(action):
                motor = joint_idxs[i]
                new_motor_pos = min(self.ul[i], max(self.ll[i], val))

                self.p.setJointMotorControl2(self.robot_id,
                                        motor,
                                        self.p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        positionGain=0.5, velocityGain=1.0,
                                        physicsClientId=self._physics_client_id)
                

    def interpolate_subaction(self, action):
        # interpolate between current location and the commanded subaction
        cur_ee = self.get_observation()[0:6]
        num_points = int(action[-1]*10) +1 # 10 is ctrl freq
    
        # interpolate x values to get x trajectory
        x_traj = np.linspace(cur_ee[0], action[0], num_points)
        y_traj = np.linspace(cur_ee[1], action[1], num_points)
        z_traj = np.linspace(cur_ee[2], action[2], num_points)
        #rotz_traj = np.linspace(cur_ee[5], action[3], num_points)
        
        traj = np.array([x_traj, y_traj, z_traj])
        #traj = np.array([x_traj, y_traj, z_traj, rotz_traj])
        
        return traj

    def rescale_subaction(self, action):
        """
        end effector control - subaction for pushing
        subactions input: (x,y,rotz)

        - actions are in robot frame
        - x and y world frame = xy robot frame (only change z)and T
        
        """
        if not (len(action) == 3):
            #raise AssertionError('number of action commands must be (dx,dy,dz, roll, pitch, yaw, finger_position)')
            raise AssertionError('number of subaction commands must be (dx,dy)')

        xy_action = np.array(action[0:2])
        #print("raw ctrl action: ", action)
        # actions scale - coming out from a 0 to 1 genotype
        #act_scale = np.array([0.35, 0.6, 2*np.pi])
        #act_offset = np.array([0.3, -0.3, -np.pi])
        act_scale = np.array([0.35, 0.6])
        act_offset = np.array([0.3, -0.3])

        xy_act = xy_action*act_scale + act_offset
        #print("scaled real action: ", xy_act) 
        # fix the rotation components and the z component
        z_offset = 0.015 # offset from the table
        z_world = self._base_position[-1] + z_offset

        # TIME
        min_runtime = 1.0
        max_runtime = 3.0
        runtime = action[-1]*(max_runtime-min_runtime) + min_runtime
        #print("Runtime raw: ", action[-1])
        #rescaled_subaction = np.array([action[0], action[1], z_world, action[2]])
        rescaled_subaction = np.array([xy_act[0], xy_act[1], z_world, runtime])
        
        return rescaled_subaction

    
    def apply_subaction(self, action):
        
        ee_action = np.concatenate((action[0:3],
                                    [m.pi,0.,0.0]),axis=None)
        #print("ee action: ", ee_action) 
        finger_action = 0.0 # close gripper for pushing task

        #print("Commanded ee action: ", ee_action)
        # apply actions - send commands
        self.apply_action(ee_action[0:6], max_vel=1.5, max_force=-1)
        self.apply_action_fingers([finger_action, finger_action]) 

        return ee_action

    
    def pushing_primitive(self, ctrl):
        '''
        primitive consists of multiple viapoints
        - each viapoint consists of an xyz position an orientation around z axis and duraation of timesteps (x,y,z,rotz,T)

        - for a pushing primitive - the z is kept constant so is not considered in the primitive (x,y,rotZ, T)

        ctrl is a list of primitives as such
        '''
        controlStep = 0 
        old_time = 0
        first = True

        obs_recording = []
        action_recording =[]
    
        subaction_size = 3 # x y size of 1 subaction in the primitive
        num_subactions = len(ctrl)/subaction_size # number of actions in primitive
        primitive = [ctrl[i:i + subaction_size] for i in range(0, len(ctrl), subaction_size)]
        #print("Primitive: ", primitive)
        for action in primitive: 
            
            controlStep = 0 
            old_time = 0
            first = True

            # get subaction trajectory
            action = self.rescale_subaction(action)
            traj = self.interpolate_subaction(action)

            runtime = action[-1]
            #print("Runtime: ", runtime)
            for i in range (int(runtime/self.sim_timestep)): 
                # Take actions only during the control frequency 
                if (i*self.sim_timestep - old_time > self.control_timestep) or first:   
                    # get observation
                    obs = self.get_policy_obs()
                    obs_recording.append(obs)
                    #print("Observation: ", obs.shape) 
                    
                    # send action to be clipped and then applied 
                    act = self.apply_subaction(traj[:,controlStep]) 
                    action_recording.append(act) # save action in buffer
                    #print("Action: ", act.shape)
                    
                    first = False
                    old_time = i*self.sim_timestep
                    controlStep += 1
                    
                self.p.stepSimulation(physicsClientId=self._physics_client_id) # step in sim

                if self.gui:
                    time.sleep(self.sim_timestep) 

        # convert recordings/replays to numpy array first
        obs_recording = np.array(obs_recording)
        action_recording = np.array(action_recording)

        #print(obs_recording.shape)
        #print(action_recording.shape)
        return obs_recording, action_recording
    
    def check_collision(self, obj_id):
        # check if there is any collision with an object (obj_id)
        contact_pts = self.p.getContactPoints(obj_id, self.robot_id, physicsClientId=self._physics_client_id)

        # check if the contact is on the fingertip(s)
        n_fingertips_contact, _ = self.check_contact_fingertips(obj_id)

        return (len(contact_pts) - n_fingertips_contact) > 0

    def check_contact_fingertips(self, obj_id):
        # check if there is any contact on the internal part of the fingers, to control if they are correctly touching an object
        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        p0 = self.p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[0], physicsClientId=self._physics_client_id)
        p1 = self.p.getContactPoints(obj_id, self.robot_id, linkIndexB=idx_fingers[1], physicsClientId=self._physics_client_id)

        p0_contact = 0
        p0_f = [0]
        if len(p0) > 0:
            # get cartesian position of the finger link frame in world coordinates
            w_pos_f0 = self.p.getLinkState(self.robot_id, idx_fingers[0], physicsClientId=self._physics_client_id)[4:6]
            f0_pos_w = self.p.invertTransform(w_pos_f0[0], w_pos_f0[1])

            for pp in p0:
                # compute relative position of the contact point wrt the finger link frame
                f0_pos_pp = self.p.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])

                # check if contact in the internal part of finger
                if f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p0_contact += 1
                    p0_f.append(pp[9])

        p0_f_mean = np.mean(p0_f)

        p1_contact = 0
        p1_f = [0]
        if len(p1) > 0:
            w_pos_f1 = self.p.getLinkState(self.robot_id, idx_fingers[1], physicsClientId=self._physics_client_id)[4:6]
            f1_pos_w = self.p.invertTransform(w_pos_f1[0], w_pos_f1[1])

            for pp in p1:
                # compute relative position of the contact point wrt the finger link frame
                f1_pos_pp = self.p.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])

                # check if contact in the internal part of finger
                if f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p1_contact += 1
                    p1_f.append(pp[9])

        p1_f_mean = np.mean(p0_f)

        return (p0_contact > 0) + (p1_contact > 0), (p0_f_mean, p1_f_mean)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def wrist_cam(self):
        pos, rot, _, _, _, _ = self.p.getLinkState(self.robot_id, linkIndex=self.end_eff_idx, computeForwardKinematics=True, physicsClientId=self._physics_client_id)
        rot_matrix = self.p.getMatrixFromQuaternion(rot)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # camera params
        height = 640
        width = 480
        fx, fy = 596.6278076171875, 596.6278076171875
        cx, cy = 311.98663330078125, 236.76170349121094
        near, far = 0.1, 10
        
        camera_vector = rot_matrix.dot((0, 0, 1))
        up_vector = rot_matrix.dot((0, -1, 0))
        
        camera_eye_pos = np.array(pos)
        camera_target_position = camera_eye_pos + 0.2 * camera_vector
        
        view_matrix = self.p.computeViewMatrix(camera_eye_pos, camera_target_position, up_vector)
        
        proj_matrix = (2.0 * fx / width, 0.0, 0.0, 0.0,
                       0.0, 2.0 * fy / height, 0.0, 0.0,
                       1.0 - 2.0 * cx / width, 2.0 * cy / height - 1.0, (far + near) / (near - far), -1.0,
                       0.0, 0.0, 2.0 * far * near / (near - far), 0.0)

        (_, _, px, depth, mask) = self.p.getCameraImage(width=width, height=height,
                                                   viewMatrix=view_matrix,
                                                   projectionMatrix=proj_matrix,
                                                   renderer=self.p.ER_BULLET_HARDWARE_OPENGL,
                                                   physicsClientId=self._physics_client_id)  # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def front_cam(self,mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])
        
        base_pos, _ = self.p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        base_pos = np.array(base_pos)
        base_pos[0] += 0.5 # to look at the center of table instead of looking at the robot base
        
        cam_dist = 1.3
        cam_yaw = 90 # rotation about z axis of 0,0,0 - in degrees!
        cam_pitch = -40
        RENDER_HEIGHT = 256
        RENDER_WIDTH = 256
        
        view_matrix = self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                          distance=cam_dist,
                                                          yaw=cam_yaw,
                                                          pitch=cam_pitch,
                                                          roll=0,
                                                          upAxisIndex=2)
        
        proj_matrix = self.p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                   nearVal=0.1, farVal=100.0)

        (_, _, px, depth, mask) = self.p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT,
                                                   viewMatrix=view_matrix,
                                                   projectionMatrix=proj_matrix,
                                                   renderer=self.p.ER_BULLET_HARDWARE_OPENGL,
                                                   physicsClientId=self._physics_client_id)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def create_marker(self, pos):
        
        obj_name = 'target'
        obj_args = {"type":"sphere",
                    "ghost":True,
                    "body":{
                        "baseMass":0,
                        "basePosition":[0.0, 0.0, 1.0]
                    },
                    "shape":{
                        "radius": 0.03
                    },
                    "visual":{
                        "specularColor":[0.0, 0.0, 0.0],
                        "rgbaColor":[1, 0, 0, 1]
                    }}
            
        targetbaseVisualShapeIndex = self.p.createVisualShape(self.p.GEOM_SPHERE, **obj_args['shape'], **obj_args['visual'], physicsClientId=self._physics_client_id)
        self.target = self.p.createMultiBody(
            baseVisualShapeIndex=targetbaseVisualShapeIndex,
            baseCollisionShapeIndex=-1, # no collision just a ghost
            baseMass=0,
            basePosition=pos,
            physicsClientId=self._physics_client_id)

        #self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1, physicsClientId=self._physics_client_id)

        return 0

    def evaluate_solution(self, ctrl):
        # random seed needs to be set when sampling anything in numpy
        # ESPECIALLY FOR MULTIPROCESSING - OR NOT IT DEFAULTS TO WALL CLOCK TIME
        # Which for parallel processing is the same numbers
        np.random.seed() # important
        
        # runtime in seconds
        controlStep = 0 
        old_time = 0
        first = True
        
        # resets the environment and rnn state - get initial obs and hidden state
        self.reset()
        #print("rnn state shape: ", hidden.shape)

        #--- INIT ENVS AND SETTLE ---#
        # give time for the objects to appear and initialize robot and env for 1s 
        for i in range (int(1.0/self.sim_timestep)): 
            self.p.stepSimulation(physicsClientId=self._physics_client_id) # step in sim
            if self.gui:
                time.sleep(self.sim_timestep) 

        # log initial object position        
        init_obj_pos = np.array(self.get_obj_pose(self.cube_id))[0:3]

        obs_recording, action_recording = self.pushing_primitive(ctrl)
        
        # convert recordings/replays to numpy array first
        obs_recording = np.array(obs_recording)
        action_recording = np.array(action_recording)

        '''
        # let everything settle for 1 s to get final descriptor
        for i in range (int(1.0/self.sim_timestep)):
            #self.apply_action(self._home_hand_pose)
            self.p.stepSimulation(physicsClientId=self._physics_client_id) # step in sim
            if self.gui:
                time.sleep(self.sim_timestep)
        '''

        cur_ee = self.get_observation()[0:6]
        cur_ee[2] = self._base_position[2] + 0.2
        #traj = self.interpolate_action(cur_ee[0:3].append(1.0))
        # let everything settle for 1 s to get final descriptor
        for i in range (int(1.0/self.sim_timestep)):
            self.apply_action(cur_ee, max_vel=0.8, max_force=-1)
            self.p.stepSimulation(physicsClientId=self._physics_client_id) # step in sim
            if self.gui:
                time.sleep(self.sim_timestep)

        final_obj_pos = np.array(self.get_obj_pose(self.cube_id))[0:3]

        # get descriptor
        obj_distance = final_obj_pos-init_obj_pos 
        # scale descriptor
        desc_x = obj_distance[0]
        desc_y = obj_distance[1]
        desc = [[desc_x, desc_y]]
        # get fitness - distance travelled by end effector - less distance travelled higher fitness
        subaction1 = ctrl[0:3]
        subaction2 = ctrl[3:]
        subaction1 = self.rescale_subaction(subaction1)
        subaction2 = self.rescale_subaction(subaction2)
        ee_distance_moved = np.linalg.norm(subaction2[0:3]-subaction1[0:3])

        obj_z_traj = obs_recording[:,-4]
        #print("Obj z traj: ", obj_z_traj)
        if np.amax(obj_z_traj)>(0.65+0.05) or np.amin(obj_z_traj)<0.65-0.05:
            fitness = "dead"
            print("Kill solution - real eval")
        else:
            fitness = -ee_distance_moved

        #print("obs recording last: ", obs_recording[-1,0])
        return fitness, desc, obs_recording, action_recording

    
    def debug_gui(self):
        ws = self._workspace_lim
        p1 = [ws[0][0], ws[1][0], ws[2][0]]  # xmin,ymin
        p2 = [ws[0][1], ws[1][0], ws[2][0]]  # xmax,ymin
        p3 = [ws[0][1], ws[1][1], ws[2][0]]  # xmax,ymax
        p4 = [ws[0][0], ws[1][1], ws[2][0]]  # xmin,ymax

        self.p.addUserDebugLine(p1, p2, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=10, physicsClientId=self._physics_client_id)
        self.p.addUserDebugLine(p2, p3, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=10, physicsClientId=self._physics_client_id)
        self.p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=10, physicsClientId=self._physics_client_id)
        self.p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=10, physicsClientId=self._physics_client_id)

        self.p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        self.p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        self.p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)

        self.p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        self.p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)
        self.p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.end_eff_idx, physicsClientId=self._physics_client_id)

        
class pandaPushingEnv:
    def __init__(self, dynamics_model, gui=False,):
        self.render = gui
        self.dynamics_model = dynamics_model

    def evaluate_solution(self, ctrl):
        env = pandaEnv(gui=self.render, use_IK=1)
        fit, desc, obs_traj, act_traj = env.evaluate_solution(ctrl)
        
        return fit, desc, obs_traj, act_traj

    def simulate_model(self, ctrl, env, mean, det):
        states_recorded = []
        actions_recorded = []
        # initilize controller

        # initial state
        state = env.get_policy_obs()
        
        subaction_size = 3 # x y size of 1 subaction in the primitive
        num_subactions = len(ctrl)/subaction_size # number of actions in primitive
        primitive = [ctrl[i:i + subaction_size] for i in range(0, len(ctrl), subaction_size)]
        #print("Primitive: ", primitive)
        for action in primitive:    
            controlStep = 0 
            old_time = 0
            first = True

            # get subaction trajectory
            action = env.rescale_subaction(action)
            traj = env.interpolate_subaction(action)

            runtime = action[-1]
            #print("Runtime: ", runtime)
            for i in range (int(runtime/env.sim_timestep)): 
                # Take actions only during the control frequency 
                if (i*env.sim_timestep - old_time > env.control_timestep) or first:   
                    # send action to be clipped and then applied 
                    act = traj[:3,controlStep] 
                    act = np.concatenate((act[0:3],
                                          [m.pi,0.,0.0]),axis=None)
                    #print("Model act shape", act.shape)
                    #print("State shape", state.shape)
                    
                    first = False
                    old_time = i*env.sim_timestep
                    controlStep += 1
        
                    states_recorded.append(state)
                    actions_recorded.append(act)        
                    s = ptu.from_numpy(state)
                    a = ptu.from_numpy(act)
                    s = s.view(1,-1)
                    a = a.view(1,-1)

                    #print(s.shape)
                    #print(a.shape)
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
        
        env = pandaEnv(gui=False, use_IK=1)
        final_pos, states_rec, actions_rec = self.simulate_model(ctrl, env, mean, det)

        init_obj_pose = states_rec[0,-6:]
        final_obj_pose = states_rec[-1,-6:]
        #print("Init obj pose: ",init_obj_pose)
        #print("Final obj pose: ",final_obj_pose)
        
        # get descriptor
        obj_distance = final_obj_pose[0:3]-init_obj_pose[0:3] 
        # scale descriptor
        desc_x = obj_distance[0]
        desc_y = obj_distance[1]
        desc = [[desc_x, desc_y]]
        # get fitness - distance travelled by end effector
        # - because less distance travelled higher fitness
        subaction1 = ctrl[0:3]
        subaction2 = ctrl[3:]
        subaction1 = env.rescale_subaction(subaction1)
        subaction2 = env.rescale_subaction(subaction2)
        ee_distance_moved = np.linalg.norm(subaction2[0:3]-subaction1[0:3])        

        #obj_z_traj = states_rec[:,-4]
        #print("Obj z traj: ", obj_z_traj)
        #if np.amax(obj_z_traj)>(0.65+0.15) or np.amin(obj_z_traj)<0.65-0.15:
        #    fitness = "dead"
        #    print("Kill solution")
        #else:
        fitness = -ee_distance_moved
            
        obs_traj = states_rec
        act_traj = actions_rec
        disagr = 0
        return fitness, desc, obs_traj, act_traj, disagr

    def simulate_model_ensemble(self, ctrl, env, mean, disagr):
        states_recorded = []
        actions_recorded = []
        model_disagr = []

        # initial state
        state = env.get_policy_obs()
        state = np.tile(state,(self.dynamics_model.ensemble_size, 1))
        
        subaction_size = 3 # x y size of 1 subaction in the primitive
        num_subactions = len(ctrl)/subaction_size # number of actions in primitive
        primitive = [ctrl[i:i + subaction_size] for i in range(0, len(ctrl), subaction_size)]
        #print("Primitive: ", primitive)
        for action in primitive:    
            controlStep = 0 
            old_time = 0
            first = True

            # get subaction trajectory
            action = env.rescale_subaction(action)
            traj = env.interpolate_subaction(action)

            runtime = action[-1]
            #print("Runtime: ", runtime)
            for i in range (int(runtime/env.sim_timestep)): 
                # Take actions only during the control frequency 
                if (i*env.sim_timestep - old_time > env.control_timestep) or first:   
                    # send action to be clipped and then applied 
                    act = traj[:3,controlStep] 
                    act = np.concatenate((act[0:3],
                                          [m.pi,0.,0.0]),axis=None)
                    #print("Model act shape", act.shape)
                    #print("State shape", state.shape)
                    
                    first = False
                    old_time = i*env.sim_timestep
                    controlStep += 1
        
                    states_recorded.append(state)
                    actions_recorded.append(act)        
                    s = ptu.from_numpy(state)
                    a = ptu.from_numpy(act)
                    a = a.repeat(self.dynamics_model.ensemble_size,1)
                    
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
                        
                    #print(state.shape)
                    #print(pred_delta_ns)            
                    state = pred_delta_ns + state 

        final_pos = state[0] # for now just pick one model but you have all the models here 
        states_recorded = np.array(states_recorded)
        actions_recorded = np.array(actions_recorded)
        model_disagr = np.array(model_disagr)
        
        return final_pos, states_recorded, actions_recorded, model_disagr
    
    def evaluate_solution_model_ensemble(self, ctrl, mean=True, disagreement=True):
        
        env = pandaEnv(gui=False, use_IK=1)
        final_pos, states_rec, actions_rec, disagr = self.simulate_model_ensemble(ctrl, env, mean, disagreement)

        #print(states_rec.shape)
        init_obj_pose = states_rec[0,0,-6:]
        final_obj_pose = states_rec[-1,0,-6:]
        
        # get descriptor
        obj_distance = final_obj_pose[0:3]-init_obj_pose[0:3] 
        # scale descriptor
        desc_x = obj_distance[0]
        desc_y = obj_distance[1]
        desc = [[desc_x, desc_y]]
        # get fitness - distance travelled by end effector
        # less distance travelled higher fitness
        subaction1 = ctrl[0:3]
        subaction2 = ctrl[3:]
        subaction1 = env.rescale_subaction(subaction1)
        subaction2 = env.rescale_subaction(subaction2)
        ee_distance_moved = np.linalg.norm(subaction2[0:3]-subaction1[0:3])
        
        fitness = -ee_distance_moved
        obs_traj = states_rec
        act_traj = actions_rec
        
        #------------ Absolute disagreement --------------#
        # disagr is the abs disagreement trajectory for each dimension [timesteps,1,state_dim]
        # can also save the entire disagreement trajectory - but we will take final mean dis
        #print("disagr shape: ", disagr.shape)
        final_disagr = np.mean(disagr[-1,0,:])
        
        if disagreement: 
            return fitness, desc, obs_traj, act_traj, final_disagr
        else: 
            return fitness, desc, obs_traj, act_traj
        
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
    
    
    
        
if __name__ == "__main__":

    # initilaise and test env class - quick test
    use_IK = 1
    env = pandaEnv(gui=True, use_IK=use_IK)

    #ctrl = [0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0]
    #ctrl = [0.2, 0.5, 0.05, 0.9, 0.5, 0.05]
    num_subactions = 2
    subaction_size = 3 # x and y 
    ctrl = np.random.rand(num_subactions*subaction_size)
    fit, desc, obs_traj, act_traj = env.evaluate_solution(ctrl)
    
    #env.evalaute_solution(ctrl)
    #env.evaluate_solution_model(ctrl)
