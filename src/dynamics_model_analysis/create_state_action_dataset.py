import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import src.torch.pytorch_util as ptu

from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble


def load_archive():

    #filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/initial_archive/archive_100100.dat"
    #filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/initial_all_evals/archive_100100.dat"
    #filename = "src/dynamics_model_analysis/pi4_3s_10hz_data/final_archive/archive_1000000.dat"
    filename = "src/dynamics_model_analysis/pi4_3s_10hz_data/initial_archive/archive_100000.dat"

    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
    
    #print(data.shape)
    #print(data)
    # 41 columns
    # fitness, bd1, bd2, bd_ground1, bdground2, genotype(36dim)
    
    gen = data.iloc[:,-36:]
    fit = data.iloc[:,0]
    desc = data.iloc[:,1:3]

    gen = gen.to_numpy()
    fit = fit.to_numpy()
    desc = desc.to_numpy()
    
    return gen, fit, desc

def get_state_action_dataset(gen_list, env):

    N = gen_list.shape[0] # number of solutions
    obs_dataset = []
    act_dataset = []
    
    for i in range(N):
        fitness, desc, obs_traj, act_traj = env.evaluate_solution(gen_list[i])
        obs_dataset.append(obs_traj)
        act_dataset.append(act_traj)

        if (i%50)==0:
            print("{}/{} episodes saved".format(i, N))
            
    obs_dataset = np.array(obs_dataset)
    act_dataset = np.array(act_dataset)

    print("Obs dataset shape: ",obs_dataset.shape) #finalshape [num_epi, num_steps, obs_dim]
    print("Act dataset shape: ",act_dataset.shape) #finalshape [num_epi, num_steps, act_dim]
    
    return obs_dataset, act_dataset

'''
def add_sa_to_buffer(self, obs_dataset, act_dataset,  replay_buffer):

    N = obs_dataset.shape[0]

    s = obs_dataset[:, :-1]
    a = act_dataset[:, :-1]
    ns = obs_dataset[:, 1:]
    reward = 0
    done = 0
    info = {}
        
    for ep_i in range(N):
        for s_i in range(len(s)):
            replay_buffer.add_sample(s[ep_i, s_i], a[ep_i, s_i], reward, done, ns[ep_i, s_i], info)
'''

            
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./", type=str)
    args = parser.parse_args()

    gen_arr, fit_arr, desc_arr = load_archive()
    
    variant = dict(
        mbrl_kwargs=dict(
            ensemble_size=4,
            layer_size=256,
            learning_rate=1e-3,
            batch_size=256,
        )
    )

    
    obs_dim = 48
    action_dim = 18
    M = variant['mbrl_kwargs']['layer_size']

    dynamics_model = ProbabilisticEnsemble(
        ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M]
    )

    env = HexapodEnv(dynamics_model=dynamics_model,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=10)

    print("Running simulation to collect state actions dataset from genotypes")
    obs_dataset, act_dataset = get_state_action_dataset(gen_arr, env)

    filename = args.log_dir + "/state_action_data_vel_10hz_initarch.npz"
    print("SAVING DATA IN: ", filename)
    np.savez_compressed(filename, obs=obs_dataset, act=act_dataset)

    '''
    #finalshape [num_epi, num_steps, obs_dim]
    #finalshape [num_epi, num_steps, act_dim]
    
    # to access and load the dataset
    data = np.load()
    obs_dataset = data['obs']
    act_dataset = data['act']
    '''
