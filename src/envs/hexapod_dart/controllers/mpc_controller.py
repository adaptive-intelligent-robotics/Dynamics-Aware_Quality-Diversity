import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART

from src.optimizers.random_shooting.mppi import MPPIOptimizer


ARRAY_DIM = 100 #PERIOD integer timesteps of one motor period/cycle
NUM_JOINTS = 18 # hexapod wiht 3 DoF each 
NUM_LEGS = 6
NUM_JOINTS_PER_LEG = 3

class SinusoidController(rd.RobotControl):
    def __init__(self, ctrl, full_control):
        rd.RobotControl.__init__(self, ctrl, full_control)

        self.optimizer = MPPIOptimizer(
            self.horizon * self.plan_dim,
            self.planning_iters,
            self.num_rollouts,
            self.temperature,
            self.cost_function,
            polyak=self.polyak,
            filter_noise=self.filter_noise,
        )
        
    def __init__(self, ctrl, controllable_dofs):
        rd.RobotControl.__init__(self, ctrl, controllable_dofs)
        
        self.optimizer = MPPIOptimizer(
            self.horizon * self.plan_dim,
            self.planning_iters,
            self.num_rollouts,
            self.temperature,
            self.cost_function,
            polyak=self.polyak,
            filter_noise=self.filter_noise,
        )

    def configure(self):
        #  always configure after receving the controller - it is similar to setting params
        self.compute_sin_ctrl()
        self._active = True


    def calculate(self, t):
        '''
        final function that outputs commands - inherent to the parent class alrady, all final outpus must come from here
        can calculate the action/output at every timestep with the argument t
        '''
        
        target = self.get_action()
        current_pos = self.robot().positions(self._controllable_dofs)

        # applying some quick lowlevel feedback motor controller 
        gain = 1/(np.pi*0.001)# calc gain using dt as in ori implementation - dt is 0.001
        # this gain only works for the SERVO joint and does not work for torque joint
        err = target-current_pos
        cmd = gain*err

        return cmd


    def get_action(self, observation):
        '''
        rollout the different plans from current observation
        Pick best action seqeunce 
        Outputs only first action
        '''

        return action

    
    # TO-DO: This is NOT working at the moment!
    def clone(self):
        return MyController(self._ctrl, self._controllable_dofs)

