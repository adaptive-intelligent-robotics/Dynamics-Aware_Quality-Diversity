import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART


ARRAY_DIM = 100 #PERIOD integer timesteps of one motor period/cycle
NUM_JOINTS = 18 # hexapod wiht 3 DoF each 
NUM_LEGS = 6
NUM_JOINTS_PER_LEG = 3

# Create custom controller
# Sinusoid controller

class SinusoidController(rd.RobotControl):

    def __init__(self, ctrl, full_control, ctrl_freq):
        rd.RobotControl.__init__(self, ctrl, full_control)
        self._joint_signals = np.zeros((NUM_JOINTS,ARRAY_DIM)) # 18 joints, 100 timesteps in a period 
        self.ctrl_dt = 1/ctrl_freq
        self.states = []
        self.next_states = []
        self.actions = []
        
    def __init__(self, ctrl, controllable_dofs, ctrl_freq):
        rd.RobotControl.__init__(self, ctrl, controllable_dofs)
        self._joint_signals = np.zeros((NUM_JOINTS,ARRAY_DIM)) # 18 joints, 100 timesteps in a period 
        self.ctrl_dt = 1/ctrl_freq
        self.states = []
        self.actions = []
        
    def configure(self):
        # always configure after receving the controller - it is similar to setting params
        self.compute_sin_ctrl()
        self._active = True

    def set_ctrl(self, ctrl):
        self._ctrl = ctrl
        self.configure()
        
    def calculate(self, t):
        '''
        final function that outputs commands - inherent to the parent class alrady, all final outpus must come from here
        can calculate the action/output at every timestep with the argument t
        '''
        
        # states and actions
        pos = self.robot().positions() # 3+3 + 18
        vel = self.robot().velocities() # 3+3 + 18
        state = np.concatenate((pos,vel), axis=None)

        action = self.commanded_jointpos(t) # 18 - commanded joint positions
        #print("States at timestep: ", t , "is", state[5])
        
        # add and record states and actions 
        self.states.append(state)
        self.actions.append(action)

        # low level control
        current_jpos = pos[6:]        
        gain = 1/(np.pi*self.ctrl_dt)# caculate gain using dt as ori implementation - dt must be ctrl_freq
        # this gain only works for the SERVO joint and does not work for torque joint
        err = action-current_jpos
        cmd = gain*err
        #print("COMMAND: " , cmd)

        #print("Robot positions: ", self.robot().positions()[0]) # length 24
        # 18 joints, 3 cartesian position of com, 3 angular pos of com
        return cmd

    def commanded_jointpos(self, t):
        # commanded joint pos angles from the sinusoid control signal at each timestep, t

        # maximum joint angle from zero, can go likewise in -ve direction - to unnormalize the square wave
        top_joint_maxrange = np.pi/4
        bottom_joint_maxrange = np.pi/4
        
        joint_angles = np.zeros(NUM_JOINTS)

        # % over ARRAY_DIM to ensure that it is continous even after one period to keep outputting thhe correct timestep relative to the sinusoid
        timestep = int(np.floor(t*ARRAY_DIM)%ARRAY_DIM)
    
        for i in range(0,NUM_JOINTS,NUM_JOINTS_PER_LEG):
            joint_angles[i] = top_joint_maxrange*self._joint_signals[i,timestep]
        for j in range(1,NUM_JOINTS,NUM_JOINTS_PER_LEG):
            joint_angles[j] = bottom_joint_maxrange*self._joint_signals[j,timestep]
        for k in range(2,NUM_JOINTS,NUM_JOINTS_PER_LEG):
            joint_angles[k] = bottom_joint_maxrange*self._joint_signals[k,timestep]

        #print(joint_angles)
        return joint_angles
    
    def compute_sin_ctrl(self):

        cur_ctrl = self._ctrl
        
        self._joint_signals[0,:] = self.sinusoid_control_signal(cur_ctrl[0], cur_ctrl[1], cur_ctrl[2])        
        self._joint_signals[1,:] = self.sinusoid_control_signal(cur_ctrl[3], cur_ctrl[4], cur_ctrl[5])
        self._joint_signals[2,:] = self.sinusoid_control_signal(cur_ctrl[3], cur_ctrl[4], cur_ctrl[5])

        self._joint_signals[3,:] = self.sinusoid_control_signal(cur_ctrl[6], cur_ctrl[7], cur_ctrl[8])
        self._joint_signals[4,:] = self.sinusoid_control_signal(cur_ctrl[9], cur_ctrl[10], cur_ctrl[11])
        self._joint_signals[5,:] = self.sinusoid_control_signal(cur_ctrl[9], cur_ctrl[10], cur_ctrl[11])
        
        self._joint_signals[6,:] = self.sinusoid_control_signal(cur_ctrl[12], cur_ctrl[13], cur_ctrl[14])
        self._joint_signals[7,:] = self.sinusoid_control_signal(cur_ctrl[15], cur_ctrl[16], cur_ctrl[17])
        self._joint_signals[8,:] = self.sinusoid_control_signal(cur_ctrl[15], cur_ctrl[16], cur_ctrl[17])

        self._joint_signals[9,:] = self.sinusoid_control_signal(cur_ctrl[18], cur_ctrl[19], cur_ctrl[20])
        self._joint_signals[10,:] = self.sinusoid_control_signal(cur_ctrl[21], cur_ctrl[22], cur_ctrl[23])
        self._joint_signals[11,:] = self.sinusoid_control_signal(cur_ctrl[21], cur_ctrl[22], cur_ctrl[23])

        self._joint_signals[12,:] = self.sinusoid_control_signal(cur_ctrl[24], cur_ctrl[25], cur_ctrl[26])
        self._joint_signals[13,:] = self.sinusoid_control_signal(cur_ctrl[27], cur_ctrl[28], cur_ctrl[29])
        self._joint_signals[14,:] = self.sinusoid_control_signal(cur_ctrl[27], cur_ctrl[28], cur_ctrl[29])

        
        self._joint_signals[15,:] = self.sinusoid_control_signal(cur_ctrl[30], cur_ctrl[31], cur_ctrl[32])
        self._joint_signals[16,:] = self.sinusoid_control_signal(cur_ctrl[33], cur_ctrl[34], cur_ctrl[35])
        self._joint_signals[17,:] = self.sinusoid_control_signal(cur_ctrl[33], cur_ctrl[34], cur_ctrl[35])


                
    def sinusoid_control_signal(self, amplitude, phase, duty_cycle):            
        '''
        amplitude, phase and duty cycle all lie in range [0,1]
        applies sinusoid control signal for only ONE motor. this functino is applied to every motor
        returns sinusoid for one period (ARRAY_DIM)
        '''
        
        #building the square wave sinusoidal wave
        temp = np.zeros(ARRAY_DIM)
        uptime = int(np.floor(ARRAY_DIM*duty_cycle)) #unnormalize
        for count_i in range(uptime):
            temp[count_i] = amplitude
        for count_j in range(uptime, ARRAY_DIM):
            temp[count_j] = -amplitude

        # apply a filter to smooth the sqaure sine wave
        #TODO:
        command = np.zeros(ARRAY_DIM)
        kernel_size = int(ARRAY_DIM/10) # kernel windoow size for smoothing
        sigma = kernel_size/3

        kernel = np.zeros((2*kernel_size)+1)

        sum_v = 0.
        for i in range(len(kernel)):
            kernel[i] = np.exp(-(i-kernel_size)**2/(2*sigma**2)/(sigma*np.sqrt(np.pi)))
            sum_v = sum_v + kernel[i]

        for i in range(ARRAY_DIM):
            command[i] = 0
            for d in range(1,kernel_size):
                if (i-d) < 0:
                    command[i] = command[i] + temp[ARRAY_DIM+i-d]*kernel[kernel_size-d]
                else:
                    command[i] = command[i] + temp[i-d]*kernel[kernel_size-d]
            command[i] = temp[i]*kernel[kernel_size]

            for d in range(1, kernel_size):
                if (i+d) >= ARRAY_DIM:
                    command[i] = command[i] + temp[i+d-ARRAY_DIM]*kernel[kernel_size+d]
                else:
                    command[i] = command[i] + temp[i+d]*kernel[kernel_size+d]
            command[i] /= sum_v
            

        # apply phase shift - shift is applied by shifting the wave to the left
        final_command = np.zeros(ARRAY_DIM)
        current = 0 
        start = int(np.floor(ARRAY_DIM*phase))

        # shift the start of the sine wave by the phase
        for i in range(start,ARRAY_DIM):
            final_command[current] = command[i]
            current = current + 1

        # fill in the front part of the sine wave with the end of the shifted sine wave
        for j in range(start):
            final_command[current] = command[j]
            current = current + 1

        
        return final_command


    # TO-DO: This is NOT working at the moment!
    def clone(self):
        return MyController(self._ctrl, self._controllable_dofs)

