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

# for fitness and novelty score
from src.map_elites import model_condition_utils
from src.map_elites import unstructured_container

import multiprocessing

# params for mutation
params = \
    {
        # min/max of parameters
        "min": 0.0,
        "max": 1.0,
        
        #------------MUTATION PARAMS---------#        
        # mutation operator ["iso_dd", "polynomial", "sbx"]
        "mutation" : "iso_dd",
        
        # probability of mutating each value in the genotype - not used in iso dd
        "mutation_prob": 0.2,
        # param for 'polynomial' mutation for variation operator
        "eta_m": 10.0,

        "parallel": True,
        
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,

        "nov_l": 0.005,
        "eps": 0.1, # usually 10%
        "k": 15,  # from novelty search

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

def get_archive(dataset):
    # arcchive is just a list of this species/objects
    archive = []

    genotype = dataset[0]
    fit_bd = dataset[1]

    fit = fit_bd[:,0]
    bd = fit_bd[:,1:]
    for i in range(genotype.shape[0]): 
        s = cm.Species(genotype[i], bd[i], bd[i], fit[i])
        archive.append(s)
    
    return archive

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


def addition_condition(s_list, archive, params):
    add_list = [] # list of solutions that were added
    discard_list = []
    label_list = []
    for s in s_list:
        success = unstructured_container.add_to_archive(s, archive, params)
        label_list.append(success)
        if success:
            add_list.append(s)
        else:
            discard_list.append(s) #not important for alogrithm but to collect stats
        
    return archive, add_list, discard_list, label_list


def model_addition_condition(s_list, archive, add_params):
    add_list = [] # list of solutions that were added
    discard_list = []
    add_list_idx = [] # list of solutions that were added
    discard_list_idx = []
    i = 0 # index counter
    for s in s_list:
        success = model_condition_utils.add_to_archive_2(s, archive, add_params)
        if success:
            add_list.append(s)
            add_list_idx.append(i)
        else:
            discard_list.append(s) #not important for alogrithm but to collect stats
            discard_list_idx.append(i) #not important for alogrithm but to collect stats
        i += 1
    return archive, add_list_idx, discard_list_idx


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
    s_list = []
    for ctrl in ctrl_list: 
        with torch.no_grad():
            ctrl_torch = ptu.from_numpy(ctrl)
            ctrl_torch = ctrl_torch.view(1, -1)
            pred = model.output_pred(ctrl_torch)
            fit = pred[0,0] # already in numpy format
            bd = pred[0,1:3]
            
            s = cm.Species(ctrl, bd, bd, fit)
            s_list.append(s)
            
    return s_list


def evaluate_(t):

    z, f = t
    fit, desc, obs_traj, _ = f(z) 
    
    desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    desc_ground = desc
    
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, desc_ground, fit)


def get_gt_batch_results(ctrl_list, env, params):

    '''
    #series 
    s_list = []
    for ctrl in ctrl_list:
        fitness, desc, obs_traj, _ = env.evaluate_solution(ctrl)
        s = cm.Species(ctrl, desc[0], desc[0], fitness)
        s_list.append(s)
     '''

    # parallel
    num_cores = 16
    pool = multiprocessing.Pool(num_cores)
    to_evaluate = []
    for ctrl in ctrl_list:
        to_evaluate += [(ctrl, env.evaluate_solution)]

    s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params)
    
    return s_list

def get_dynamics_model_batch_results(ctrl_list, env):
    s_list = []
    for ctrl in ctrl_list:
        fitness, desc, obs_traj, _ = env.evaluate_solution_model(ctrl, det=True)
        s = cm.Species(ctrl, desc[0], desc[0], fitness)
        s_list.append(s)
         
    return s_list

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

def get_model_gt_s_list(ctrl_list, env, params):
    print("Evaluating individuals using model")
    model_s_list = get_dynamics_model_batch_results(ctrl_list, env)
    print("Evaluating individuals via ground truth evaluations")
    gt_s_list = get_gt_batch_results(ctrl_list, env, params)
    return model_s_list, gt_s_list


def compute_fit_nov_score(archive, model_s_list):
    '''
    from a list of evaluated solutions already with bd and fitness score
    compute novelty score of each individual
    compute fitness score of each individual 
    '''
    k = 15
    tmp_archive = archive.copy()
    nov_score_list = []
    fit_score_list = []

    for s in model_s_list:
        # get the k nn of the individual we are evaluating
        neighbours_cur, _ = model_condition_utils.knn(s.desc, k, tmp_archive)
        # get novelty score
        nov_score_cur = model_condition_utils.get_novelty(s, neighbours_cur)
        # get fitness score 
        fit_score_cur = model_condition_utils.get_fitness_score(s, neighbours_cur)

        nov_score_list.append(nov_score_cur)
        fit_score_list.append(fit_score_cur)
    
    return nov_score_list, fit_score_list


def create_labels(archive, gt_s_list):
    '''
    create the gt labels (added or not addded) for each of individuals
    '''
    tmp_archive = archive.copy() 

    # 0 if not added, 1 if added
    tmp_archive, add_list, discard_list, label_list = addition_condition(gt_s_list, tmp_archive, params)
    
    return label_list


def auc_roc_analysis(archive, model_s_list, gt_s_list, params):
    '''
    takes in solution list (evaluated mutated individuals) by model and ground truth
    and thereshold value
    returns fpr and tpr for this particular threshold value of all solutions in this list
    '''
    real_archive = archive.copy() # for positives
    tmp_archive = archive.copy() # for negatives
    model_archive = archive.copy() 

    n_inds = len(model_s_list)

    # values of thresholds to choose from
    threshold_arr = np.array([0.2, 0.0, -0.2, -0.4, -1.])
    true_pos, false_pos, false_neg, true_neg = 0, 0, 0, 0 
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    for t_qua in threshold_arr:
        t_nov = 0.01
        k = 15
        add_params = [t_qua, t_nov, k]
        model_archive, add_list_model_idx, discard_list_model_idx = model_addition_condition(model_s_list, model_archive, add_params)
        
        # positives
        pos_s_list = [gt_s_list[i] for i in add_list_model_idx] # ground truth
        real_archive, add_list, discard_list, _ = addition_condition(pos_s_list, real_archive, params)
        true_pos = len(add_list)
        false_pos = len(discard_list)
        tp_list.append(true_pos)
        fp_list.append(false_pos)
        
        # negatives 
        neg_s_list = [gt_s_list[i] for i in discard_list_model_idx] # ground truth
        tmp_archive, add_list, discard_list, _ = addition_condition(neg_s_list, tmp_archive, params)
        false_neg = len(add_list)
        true_neg = len(discard_list)
        fn_list.append(false_neg)
        tn_list.append(true_neg)
        
    return tp_list, fp_list, tn_list, fn_list
    



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
    #filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/final_archive/archive_1000100.dat"
    filename = "src/dynamics_model_analysis/pi4_3s_100hz_data/initial_archive/archive_100100.dat"

    dataset = load_dataset(filename) # dataset is x-genotype annd y-fit and bd
    archive = get_archive(dataset)

    num_eval_datapts = 1000
    data = pd.DataFrame() # data to hold all the errors computed for plotting

    # list of mutated indiviudals
    to_evaluate = get_mutated_inds(dataset, num_eval_datapts)

    # evaluate fitness and bd using model and real/gt 
    model_s_list, gt_s_list = get_model_gt_s_list(to_evaluate, det_model_env, params)
    print("Computing direct nn model predictions")
    surrogate_s_list = get_surrogate_model_pred(to_evaluate, surrogate_model)

    # compute fit and nov score of each of the individuals based on model bd and fit
    nov_score_list, fit_score_list = compute_fit_nov_score(archive, model_s_list)
    sur_nov_score_list, sur_fit_score_list = compute_fit_nov_score(archive, surrogate_s_list)
    # compute labels based on gt real evals of each ind
    label_list = create_labels(archive, gt_s_list)

    data = pd.DataFrame()
    data["GT labels"] = label_list
    data["Fitness score"] = fit_score_list
    data["Novelty score"] = nov_score_list
    data["Fitness score direct"] = sur_fit_score_list
    data["Novelty score direct"] = sur_nov_score_list
    
    print("data shape: ", data.shape) 
    data.to_csv("aucroc_data_compare.csv")
    
    #threshold = [-0.2, 0, 0.2]
    #for t_qua in threshold: 
    #    data['>'+str(t_qua)] = data['Fitness score'].apply(lambda x: 1 if x>t_qua else 0) 

    #print("Performing auc roc analysis")
    #tp_list, fp_list, tn_list, fn_list = auc_roc_analysis(archive, model_s_list, gt_s_list, par#ams)
    #print(tp_list, fp_list, tn_list, fn_list)
    
    # plot
    #ax.set_title("BD errors of models")
    #ax.set_ylabel("BD error")
    #ax.set_ylabel("Fitness error")
    #ax.set_xlabel("Models")    
    
    #plt.show()

