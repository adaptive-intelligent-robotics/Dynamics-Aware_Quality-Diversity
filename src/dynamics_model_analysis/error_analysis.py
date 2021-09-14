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
from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate

# Done below in the code
#from src.trainers.mbrl.mbrl_det import MBRLTrainer 
#from src.trainers.mbrl.mbrl import MBRLTrainer
#from src.trainers.qd.surrogate import SurrogateTrainer

from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer

# for mutation of the genotypes in the evals
from src.map_elites import common as cm
variation_operator = cm.variation

# params for mutation
params = \
    {
        # min/max of parameters
        "min": 0.0,
        "max": 1.0,
        
        #------------MUTATION PARAMS---------#        
        # mutation operator ["iso_dd", "polynomial", "sbx"]
        "mutation" : "iso_dd",
        
        # probability of mutating each number in the genotype - not used by iso_dd
        "mutation_prob": 0.2,    
        # param for 'polynomial' mutation for variation operator
        "eta_m": 10.0,
        
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,                
    }


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

def compute_bd_error(bd_gt, bd_m):
    bd_gt = np.array([bd_gt[0]])
    bd_m = np.array([bd_m[0]])
    err = bd_gt - bd_m

    # for BD - norm of the error
    bd_err = np.linalg.norm(err)
    return bd_err

def compute_batch_bd_error(bd_gt, bd_m):
    bd_gt = np.array(bd_gt)
    bd_m = np.array(bd_m)
    err = bd_gt - bd_m
    #print(err)
    
    # for BD - norm of the error
    bd_err = np.linalg.norm(err, axis=-1)
    #print(bd_err)
    
    return bd_err

def compute_fitness_error(fit_gt, fit_m):

    # can do batch or individual
    fit_gt = np.array(fit_gt)
    fit_m = np.array(fit_m)

    # use absolute value
    fit_err = np.abs(fit_gt-fit_m)
    return fit_err


def compute_batch_errors(fit_list_gt, bd_list_gt, fit_list_m, bd_list_m):
    # compute errors in batches
    fit_arr_gt = np.array(fit_list_gt)
    bd_arr_gt = np.array(bd_list_gt)
    fit_arr_m = np.array(fit_list_m)
    bd_arr_m = np.array(bd_list_m)

    fit_err_list = compute_fitness_error(fit_arr_gt, fit_arr_m)
    bd_err_list = compute_batch_bd_error(bd_arr_gt, bd_arr_m)
    return fit_err_list, bd_err_list

def get_mutated_inds(dataset, n):

    mut_gen_list = []
    data_x, data_y = dataset

    # sample n number of controllers for mutation for validation
    ctrl1_idx_list = np.random.randint(data_x.shape[0], size=n)
    ctrl2_idx_list = np.random.randint(data_x.shape[0], size=n)

    for i in range(n):    
        gen1 = data_x[ctrl1_idx_list[i]]
        gen2 = data_x[ctrl2_idx_list[i]]

        # mutated individual that we want to evaluate
        ref_ctrl = variation_operator(gen1, gen2, params)
        mut_gen_list.append(ref_ctrl)
    
    # list of mutated genotypes to evaluated
    return mut_gen_list

def error_analysis(to_evaluate_list, env, n, det, mean=False):

    # computes the BD error and fitness error of the model vs the ground truth eval
    plot=False    
    bd_error_list = [] # should end up being of length n
    fit_error_list = []
    for i in range(n):
        
        ref_ctrl = to_evaluate_list[i]
        # get model prediction for a reference controller to visualize
        fitness_m, desc_m, obs_traj_m, _ = env.evaluate_solution_model(ref_ctrl, det=det, mean=mean)
        fitness, desc, obs_traj, _ = env.evaluate_solution(ref_ctrl)

        bd_error = compute_bd_error(desc, desc_m)
        fitness_error = compute_fitness_error(fitness, fitness_m)

        bd_error_list.append(bd_error)
        fit_error_list.append(fitness_error)

        if plot:
            # plot BD's comaprison
            plt.scatter(x=desc_m[0][0] , y=desc_m[0][1] , c=fitness_m, cmap='viridis', s=30.0, marker='x', label='Model prediction '+str(i))
            plt.scatter(x=desc[0][0] , y=desc[0][1] , c=fitness, cmap='viridis', s=30.0, marker='o', label='Ground truth '+str(i))

            # plot xy trajectories postions - exact
            #plt.plot(obs_traj[:,3], obs_traj[:,4], "-",label="Ground truth "+str(i))
            #plt.plot(obs_traj_m[:,3], obs_traj_m[:,4], '--', label="Dynamics Model "+str(i))

    if plot:
        plt.xlim([0,1])
        plt.ylim([0,1])
        #plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.show()

    return fit_error_list, bd_error_list
    

def get_dynamics_model(dynamics_model_type):

    ## INIT MODEL ##
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


    return dynamics_model, dynamics_model_trainer


def get_surrogate_model():
    from src.trainers.qd.surrogate import SurrogateTrainer
    dim_x=36 # genotype dimnesion    
    model = DeterministicQDSurrogate(gen_dim=dim_x, bd_dim=2, hidden_size=64)
    model_trainer = SurrogateTrainer(model, batch_size=64)

    return model, model_trainer


def get_surrogate_model_pred(ctrl_list, model):
    fit_list = []
    bd_list = []
    for ctrl in ctrl_list: 
        with torch.no_grad():
            ctrl_torch = ptu.from_numpy(ctrl)
            ctrl_torch = ctrl_torch.view(1, -1)
            pred = model.output_pred(ctrl_torch)
            fit = pred[0,0]
            bd = pred[0,1:3]
            fit_list.append(fit)
            bd_list.append(bd)
            
    return fit_list, bd_list

def get_gt_batch_results(ctrl_list, env):
    fit_list = []
    bd_list = []
    for ctrl in ctrl_list:
        fitness, desc, obs_traj, _ = env.evaluate_solution(ctrl)
        fit_list.append(fitness)
        bd_list.append(desc[0])
            
    return fit_list, bd_list

def get_dynamics_model_batch_results(ctrl_list, env, det, mean=False):
    fit_list = []
    bd_list = []
    for ctrl in ctrl_list:
        fitness, desc, obs_traj, _ = env.evaluate_solution_model(ctrl, det=det, mean=mean)
        fit_list.append(fitness)
        bd_list.append(desc[0])
            
    return fit_list, bd_list

def get_dynamics_model_batch_results_sampling(ctrl_list, env, det, mean=False):
    fit_list = []
    bd_list = []
    for ctrl in ctrl_list:
        # Perfrom n sample rollouts to get a distribution of predictions for the same controller
        n=100
        fit_samples = []
        bd_samples = []
        for i in range(n):
            fitness, desc, obs_traj, _ = env.evaluate_solution_model(ctrl, det=det, mean=mean)
            fit_samples.append(fitness)
            bd_samples.append(desc[0])

        mean_fitness = np.mean(fit_samples)
        mean_desc = np.mean(np.array(bd_samples),axis=0)
        fit_list.append(mean_fitness)
        bd_list.append(mean_desc)
            
    return fit_list, bd_list



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()
    
    obs_dim = 48
    action_dim = 18    

    # Deterministic = "det", Probablistic = "prob" 
    #dynamics_model_type = "prob"   

    prob_dynamics_model, _ = get_dynamics_model("prob")
    model_path = "src/dynamics_model_analysis/trained_models/prob_ensemble_initarch.pth"
    prob_dynamics_model = ptu.load_model(prob_dynamics_model, model_path)    

    det_dynamics_model, _ = get_dynamics_model("det")
    model_path = "src/dynamics_model_analysis/trained_models/det_initarch.pth"
    det_dynamics_model = ptu.load_model(det_dynamics_model, model_path)    

    surrogate_model, _ = get_surrogate_model()
    model_path = "src/surrogate_model_analysis/trained_models/surrogate_nn_initarch.pth"
    surrogate_model = ptu.load_model(surrogate_model, model_path)    

    
    ## LOAD TRAINED MODEL ##
    # PROBABLISTIC MODELS
    #model_path = "src/dynamics_model_analysis/trained_models/prob_ensemble_initarch.pth"
    #model_path = "src/dynamics_model_analysis/trained_models/prob_ensemble_finalarch.pth"
    #model_path = "src/dynamics_model_analysis/trained_models/prob_ensemble_initallevals.pth"

    # DETERMINISTIC MODELS
    #model_path = "src/dynamics_model_analysis/trained_models/det_initarch.pth"
    #model_path = "src/dynamics_model_analysis/trained_models/det_finalarch.pth"
    #model_path = "src/dynamics_model_analysis/trained_models/det_initallevals.pth"
    
    #dynamics_model = ptu.load_model(dynamics_model, model_path)    
    
    
    # Initialize environments
    # surrogate model does not require a environment - just the controller
    prob_model_env = HexapodEnv(dynamics_model=prob_dynamics_model,
                                render=False,
                                record_state_action=True,
                                ctrl_freq=100)
    
    det_model_env = HexapodEnv(dynamics_model=det_dynamics_model,
                               render=False,
                               record_state_action=True,
                               ctrl_freq=100)
    
    ## PERFORM ERROR ANALYSIS ##
    # Performing error analysis (validation) on mutations of the training data

    # Load in archive or training data corresponding to trained models
    filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/final_archive/archive_1000100.dat"
    #filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/initial_archive/archive_100100.dat"
    dataset = load_dataset(filename)

    num_eval_datapts = 100
    data = pd.DataFrame()

    # get mutations and corresponding bd and fitness errors of the diff models
    
    # list of mutated indiviudals
    to_evaluate = get_mutated_inds(dataset, num_eval_datapts)

    ## GT BATCH ##
    print("Getting ground truth results")
    fit_list_gt, bd_list_gt = get_gt_batch_results(to_evaluate, det_model_env)

    ## SURROGATE MODEL ##
    print("Getting direct surrogate model results")
    fit_list_m, bd_list_m = get_surrogate_model_pred(to_evaluate, surrogate_model) 
    fit_error_list, bd_error_list = compute_batch_errors(fit_list_gt, bd_list_gt, fit_list_m, bd_list_m)
    data_tmp = pd.DataFrame()
    data_tmp["Fitness Error"] = fit_error_list
    data_tmp["BD Error"] = bd_error_list
    data_tmp["xlab"] = 1.0
    data_tmp["Model"] = "Surrogate Model"
    data = pd.concat([data, data_tmp], ignore_index=True)
    
    ## DETERMINISTIC DYNAMICS MODEL ##
    print("Getting deterministic dynamics model results")
    #fit_error_list, bd_error_list = error_analysis(to_evaluate, det_model_env, num_eval_datapts, det=True)
    fit_list_m, bd_list_m = get_dynamics_model_batch_results(to_evaluate, det_model_env, det=True)
    fit_error_list, bd_error_list = compute_batch_errors(fit_list_gt, bd_list_gt, fit_list_m, bd_list_m)
    data_tmp = pd.DataFrame()
    data_tmp["Fitness Error"] = fit_error_list
    data_tmp["BD Error"] = bd_error_list
    data_tmp["xlab"] = 2.0
    data_tmp["Model"] = "Det. Model"
    data = pd.concat([data, data_tmp], ignore_index=True)

    ## PROBABILISTIC DYNAMICS MODEL ##
    print("Getting probablistic dynamics model (sampling 100 rollouts per controller) results")
    #fit_error_list, bd_error_list = error_analysis(to_evaluate, prob_model_env, num_eval_datapts, det=False)
    fit_list_m, bd_list_m = get_dynamics_model_batch_results_sampling(to_evaluate, prob_model_env, det=False)
    fit_error_list, bd_error_list = compute_batch_errors(fit_list_gt, bd_list_gt, fit_list_m, bd_list_m)
    data_tmp = pd.DataFrame()
    data_tmp["Fitness Error"] = fit_error_list
    data_tmp["BD Error"] = bd_error_list
    data_tmp["xlab"] = 3.0
    data_tmp["Model"] = "Prob. Model"
    data = pd.concat([data, data_tmp], ignore_index=True)

    ## PROBABILISTIC DYNAMICS MODEL - MEAN ##
    print("Getting probabilistic dynamics model mean results")
    #fit_error_list, bd_error_list = error_analysis(to_evaluate, prob_model_env, num_eval_datapts, det=False, mean=True)
    fit_list_m, bd_list_m = get_dynamics_model_batch_results(to_evaluate, prob_model_env, det=False, mean=True)
    fit_error_list, bd_error_list = compute_batch_errors(fit_list_gt, bd_list_gt, fit_list_m, bd_list_m)
    data_tmp = pd.DataFrame()
    data_tmp["Fitness Error"] = fit_error_list
    data_tmp["BD Error"] = bd_error_list
    data_tmp["xlab"] = 4.0
    data_tmp["Model"] = "Prob. Model - Mean"
    data = pd.concat([data, data_tmp], ignore_index=True)

    print(data) 
    data.to_csv("error_analysis.csv")
    
    name="initarchmodel_testfinalarch"
    ## PLOT ERRORS ##
    ax1 = sns.violinplot(x="Model", y="BD Error", data=data, scale="area")
    plt.ylim(bottom=0)
    plt.savefig("bd_error_"+name+".svg")
    plt.clf()
    
    ax2 = sns.violinplot(x="Model", y="Fitness Error", data=data,scale="area") 
    plt.ylim(bottom=0)
    plt.savefig("fit_error_"+name+".svg")
    plt.clf()

    ax1 = sns.violinplot(x="Model", y="BD Error", data=data, scale="area", bw=0.1)
    plt.ylim(bottom=0)
    plt.savefig("bd_error_bw_"+name+".svg")
    plt.clf()
    
    ax2 = sns.violinplot(x="Model", y="Fitness Error", data=data,scale="area", bw=0.1) 
    plt.ylim(bottom=0)
    plt.savefig("fit_error_bw_"+name+".svg")
    plt.clf()

    ax3 = sns.boxplot(x="Model", y="BD Error", data=data)
    plt.ylim(bottom=0)
    plt.savefig("bd_error_boxplot_"+name+".svg")
    plt.clf()
    
    ax4 = sns.boxplot(x="Model", y="Fitness Error", data=data) 
    plt.ylim(bottom=0)
    plt.savefig("fit_error_boxplot_"+name+".svg")
    plt.clf()


    
    #ax.set_title("BD errors of models")
    #ax.set_ylabel("BD error")
    #ax.set_ylabel("Fitness error")
    #ax.set_xlabel("Models")
    
    
    #plt.show()

