import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

import src.torch.pytorch_util as ptu

from src.envs.hexapod_dart.hexapod_env import HexapodEnv

from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from src.models.dynamics_models.deterministic_model import DeterministicDynModel

# Done below in the code
#from src.trainers.mbrl.mbrl_det import MBRLTrainer 
#from src.trainers.mbrl.mbrl import MBRLTrainer

from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer


def load_dataset(filename):

    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma

    # 41 columns
    # fitness, bd1, bd2, bd_ground1, bdground2, genotype(36dim)
    x = data.iloc[:,-36:]
    y = data.iloc[:,0:3]

    x = x.to_numpy()
    y = y.to_numpy()

    print("x dataset shape: ", x.shape)
    print("y dataset shape: ", y.shape)

    dataset = [x,y]

    return dataset


def add_sa_to_buffer(obs_dataset, act_dataset,  replay_buffer):

    N = obs_dataset.shape[0]

    s = obs_dataset[:, :-1]
    a = act_dataset[:, :-1]
    ns = obs_dataset[:, 1:]
    reward = 0
    done = 0
    info = {}

    #print(s.shape)
    
    for ep_i in range(N):
        for s_i in range(s.shape[1]):
            replay_buffer.add_sample(s[ep_i, s_i], a[ep_i, s_i], reward, done, ns[ep_i, s_i], info)

    print("Replay buffer size: ", replay_buffer._size)
    
    return 1


def plot_state_comparison(state_traj, state_traj_m):

    total_t = state_traj.shape[0]
    #total_t = 30
    
    #state_traj.shape[1]
    f, (ax1) = plt.subplots(1, 1)

    for i in np.arange(3,6):
        traj_real = state_traj[:,i]
        traj_m = state_traj_m[:,i]
        ax1.plot(np.arange(total_t), traj_real[:total_t], "-",label="Ground truth "+str(i))
        ax1.plot(np.arange(total_t), traj_m[:total_t], '--', label="Dynamics Model "+str(i))
    ax1.legend()

    '''
    for i in np.arange(9,12):
        traj_real = state_traj[:,i]
        traj_m = state_traj_m[:,i]
        ax2.plot(np.arange(total_t), traj_real, "-",label="Ground truth "+str(i))
        ax2.plot(np.arange(total_t), traj_m, '--', label="Dynamics Model "+str(i))
    ax2.legend()
    '''
    
    #plt.legend()
    plt.show()

    return 1


def gt_vs_model(dataset, env, n):

    data_x, data_y = dataset

    # sample n number of controllers for validation and comparison
    for i in range(n):

        ctrl_idx = np.random.randint(data_x.shape[0])
        ref_ctrl = data_x[ctrl_idx]
        ref_answer = data_y[ctrl_idx]

        # get model prediction for a reference controller to visualize
        fitness_m, desc_m, obs_traj_m, _ = env.evaluate_solution_model(ref_ctrl)
        fitness, desc, obs_traj, _ = env.evaluate_solution(ref_ctrl)
        #print(desc)

        #print("Ground truth: ", fitness, desc)
        #print("Reference answer from dataset: ", ref_answer)

        # plot BD's comaprison
        plt.scatter(x=desc_m[0][0] , y=desc_m[0][1] , c=fitness_m, cmap='viridis', s=30.0, marker='x', label='Model prediction '+str(i))
        plt.scatter(x=desc[0][0] , y=desc[0][1] , c=fitness, cmap='viridis', s=30.0, marker='o', label='Ground truth '+str(i))

        # plot xy trajectories postions - exact
        #plt.plot(obs_traj[:,3], obs_traj[:,4], "-",label="Ground truth "+str(i))
        #plt.plot(obs_traj_m[:,3], obs_traj_m[:,4], '--', label="Dynamics Model "+str(i))

    plt.xlim([0,1])
    plt.ylim([0,1])

    #plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()


def plot_prob_model_rollout(dataset, env, n):

    data_x, data_y = dataset
    ctrl_idx = np.random.randint(data_x.shape[0])
    ref_ctrl = data_x[ctrl_idx]
    ref_answer = data_y[ctrl_idx]

    data = pd.DataFrame()

    total_t = 300
    
    data_tmp = pd.DataFrame()
    # run the probablistic rollout on the saem genotype  for n times to see distrbution
    for i in range(n):    
        fitness_m, desc_m, obs_traj_m, _ = env.evaluate_solution_model(ref_ctrl)
        tmp = pd.DataFrame(obs_traj_m)
        tmp['run']=i # replication of the same variant
        tmp['timestep']=pd.DataFrame(np.arange(total_t))
        data_tmp=pd.concat([data_tmp, tmp], ignore_index=True)

    print(data_tmp)
    # distribution of probablisitic model over 100 sample rollouts
    sns.set(style="whitegrid")
    plt.figure()
    sns_plot = sns.lineplot(x="timestep", y=3,
                            data=data_tmp)
    sns_plot = sns.lineplot(x="timestep", y=4,
                            data=data_tmp)

    # mean of probablistic model
    fitness, desc, obs_traj_m, _ = env.evaluate_solution_model(ref_ctrl, mean=True)
    plt.plot(np.arange(total_t), obs_traj_m[:,3],'-.')
    plt.plot(np.arange(total_t), obs_traj_m[:,4],'-.')

    # ground truth
    fitness, desc, obs_traj, _ = env.evaluate_solution(ref_ctrl)
    plt.plot(np.arange(total_t), obs_traj[:,3],'--')
    plt.plot(np.arange(total_t), obs_traj[:,4],'--')

    print("Plotting.....")
    plt.show()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()
    
    obs_dim = 48
    action_dim = 18    

    # Deterministic = "det", Probablistic = "prob" 
    dynamics_model_type = "prob" #"det" #"prob"  

    # train = True (train) , train =False (test) 
    train = False
    
    if dynamics_model_type == "prob":

        from src.trainers.mbrl.mbrl import MBRLTrainer
        
        variant = dict(
            mbrl_kwargs=dict(
                ensemble_size=4,
                layer_size=500,
                learning_rate=1e-3,
                batch_size=512,
            )
        )
    
        M = variant['mbrl_kwargs']['layer_size']

        dynamics_model = ProbabilisticEnsemble(
            ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M]
        )

        dynamics_model_trainer = MBRLTrainer(
            ensemble=dynamics_model,
            **variant['mbrl_kwargs'],
        )

    elif dynamics_model_type == "det":
        
        from src.trainers.mbrl.mbrl_det import MBRLTrainer 

        dynamics_model = DeterministicDynModel(obs_dim=obs_dim,
                                               action_dim=action_dim,
                                               hidden_size=500)

        dynamics_model_trainer = MBRLTrainer(
            model=dynamics_model,
            batch_size=512,)
    

    # Initialize environment
    env = HexapodEnv(dynamics_model=dynamics_model,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=100)

    # Initialize replay buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=5000000,
        observation_dim=obs_dim,
        action_dim=action_dim,
        env_info_sizes=dict(),)

    
    if train: 
        ## TRAIN ##
        #filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/final_archive/state_action_data_vel_100hz.npz"
        filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/initial_archive/state_action_data_vel_100hz_initarch.npz"
        #filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/initial_all_evals/state_action_data_vel_100hz_initallevals.npz"

        data = np.load(filename)

        obs_dataset = data['obs'] #shape [num_epi, num_steps, obs_dim]
        act_dataset = data['act']
        
        print("Observation dataset shape: ",obs_dataset.shape)
        print("Action dataset shape: ",act_dataset.shape)
        
        # Add state action dataset to replay buffer
        print("Adding state action dataset to buffer")
        add_sa_to_buffer(obs_dataset, act_dataset, replay_buffer)
        
        # Train model on dataset
        print("Training model from buffer")
        dynamics_model_trainer.train_from_buffer(replay_buffer,
                                                 holdout_pct=0.1,
                                                 max_grad_steps=100000)

        ## SAVE MODEL ##
        dynamics_model_path = args.log_dir + "det_initarch.pth"
        ptu.save_model(dynamics_model, dynamics_model_path)

    else: 
        ## LOAD TRAINED MODEL ##
        #model_path = "src/dynamics_model_analysis/trained_models_test/det_trained_model.pth"
        #model_path = "src/dynamics_model_analysis/trained_models_test/prob_ensemble_highmaxvar_trained_model.pth"
        model_path = "src/dynamics_model_analysis/trained_models_test/prob_ensemble_lowmaxvar_trained_model.pth"

        # For the main evaluation analysis
        #Probabilistic
        #model_path = "src/dynamics_model_analysis/trained_models/det_trained_model.pth"
        #model_path = "src/dynamics_model_analysis/trained_models/det_trained_model.pth"
        #model_path = "src/dynamics_model_analysis/trained_models/det_trained_model.pth"

        # Deterministic
        #model_path = "src/dynamics_model_analysis/trained_models/det_finalarch.pth"
        #model_path = "src/dynamics_model_analysis/trained_models/det_initarch.pth"
        #model_path = "src/dynamics_model_analysis/trained_models/det_trained_model.pth"
        
        dynamics_model = ptu.load_model(dynamics_model, model_path)    
        
        
    ## VALIDATE ##
    filename_final = "src/dynamics_model_analysis/pi4_3s_100hz_data/final_archive/archive_1000100.dat"
    dataset_final = load_dataset(filename_final)

    data_x, data_y = dataset_final
    ctrl_idx = np.random.randint(data_x.shape[0])
    ref_ctrl = data_x[ctrl_idx]
    
    fit, desc, state_traj , _ = env.evaluate_solution(ref_ctrl)
    fit_m, desc_m, state_traj_m , _ = env.evaluate_solution_model(ref_ctrl, det=False)
    #fit_m, desc_m, state_traj_m , _ = env.evaluate_solution_model(ref_ctrl, det=False)

    print("Fitness comparison ---", "Ground truth:", fit, "Model: ", fit_m)
    print("Desc comparison ---", "Ground truth:", desc, "Model: ", desc_m)

    print("Dataset fitness and bd: ", data_y[ctrl_idx])

    print("length: ", state_traj.shape)
    print("length: ", state_traj_m.shape)

    
    #plot_state_comparison(state_traj, state_traj_m)

    #gt_vs_model(dataset_final, env, 10)
    
    plot_prob_model_rollout(dataset_final, env, 10)

