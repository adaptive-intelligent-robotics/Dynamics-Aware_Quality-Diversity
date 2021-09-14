import numpy as np
import matplotlib.pyplot as plt
import src.torch.pytorch_util as ptu
from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
import pandas as pd

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

def plot_state_comparison(state_traj, state_traj_m, st_me, st_med):

    total_t = state_traj.shape[0]
    #total_t = 30

    #state_traj.shape[1]
    #f, (ax1) = plt.subplots(1, 1)

    for i in np.arange(2,3):
        traj_real = state_traj[:,i]
        traj_m = state_traj_m[:,i]
        traj_me = st_me[:,i]
        traj_med = st_med[:,i]
        plt.plot(np.arange(total_t), traj_real[:total_t], "-",label="Ground truth "+str(i))
        plt.plot(np.arange(total_t), traj_m[:total_t], '--', label="Prob Dyn Model - No ens "+str(i))
        plt.plot(np.arange(total_t), traj_me[:total_t], '-.', label="Prob Dyn Model - Ensemble "+str(i))
        plt.plot(np.arange(total_t), traj_med[:total_t], '-.', label="Prob Dyn Model - Ensemble Disagreement "+str(i))
    #ax1.legend()
    #plt.legend()
    
    return 1


def plot_disagreement_rollout(disagr_arr):

    plt.plot(disagr_arr)
    
    return 1


if __name__ == "__main__":

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

    model_path = "src/dynamics_model_analysis/trained_models/prob_ensemble_finalarch.pth"
    ptu.load_model(prob_dynamics_model, model_path)

    
    env = HexapodEnv(prob_dynamics_model, render=False, record_state_action=True)

    # test simulation with a random controller
    #ctrl = np.random.uniform(0,1,size=36)
    filename_final = "src/dynamics_model_analysis/pi4_3s_100hz_data/final_archive/archive_1000100.dat"
    dataset_final = load_dataset(filename_final)
    data_x, data_y = dataset_final
    ctrl_idx = np.random.randint(data_x.shape[0])
    ctrl = data_x[ctrl_idx]


    fit, desc, obs_traj, act_traj = env.evaluate_solution(ctrl, render=False)
    fit_m, desc_m, obs_traj_m, act_traj_m = env.evaluate_solution_model(ctrl, mean=True, det=False)
    fit_me, desc_me, obs_traj_me, act_traj_me = env.evaluate_solution_model_ensemble(ctrl, mean=True)
    fit_med, desc_med, obs_traj_med, act_traj_med, disagr = env.evaluate_solution_model_ensemble(ctrl, mean=True, disagreement=True)
    print("Ground Truth: ", fit, desc)
    print("Probablistic no ensemble: ", fit_m, desc_m)
    print("Probablistic Model Ensemble: ", fit_me, desc_me)
    print("Probablistic Model Ensemble disagreement: ", fit_med, desc_med)

    print("Disagreement shape: ", disagr.shape)
    disagr = disagr[:,:,2]
    plot_disagreement_rollout(disagr)
    
    plot_state_comparison(obs_traj, obs_traj_m, obs_traj_me[:,0,:], obs_traj_med[:,0,:])
    plot_state_comparison(obs_traj, obs_traj_m, obs_traj_me[:,1,:], obs_traj_med[:,1,:])
    plot_state_comparison(obs_traj, obs_traj_m, obs_traj_me[:,2,:], obs_traj_med[:,2,:])
    plot_state_comparison(obs_traj, obs_traj_m, obs_traj_me[:,3,:], obs_traj_med[:,3,:])

    
    plt.show()
