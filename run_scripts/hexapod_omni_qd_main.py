import os, sys
import argparse
import numpy as np

from src.map_elites.qd import QD

from src.envs.hexapod_dart.hexapod_env import HexapodEnv

from src.models.dynamics_models.deterministic_model import DeterministicDynModel
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate

# added in get dynamics model section
#from src.trainers.mbrl.mbrl_det import MBRLTrainer
#from src.trainers.mbrl.mbrl import MBRLTrainer
#from src.trainers.qd.surrogate import SurrogateTrainer

import src.torch.pytorch_util as ptu

def get_dynamics_model(dynamics_model_type):
    obs_dim = 48
    action_dim = 18
    
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

        # ensemble somehow cant run in parallel evaluations
    elif dynamics_model_type == "det":
        from src.trainers.mbrl.mbrl_det import MBRLTrainer 
        dynamics_model = DeterministicDynModel(obs_dim=obs_dim,
                                               action_dim=action_dim,
                                               hidden_size=500)
        dynamics_model_trainer = MBRLTrainer(
            model=dynamics_model,
            batch_size=512,)


    return dynamics_model, dynamics_model_trainer

def main(args):

    px = \
    {
        # type of qd 'unstructured, grid, cvt'
        "type": args.qd_type,
        
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to parallelize
        "batch_size": args.b_size,
        # proportion of total number of niches to be filled before starting
        "random_init": 0.99, #0.005,  
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": args.dump_period,

        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": False,
        # min/max of genotype parameters - check mutation operators too
        "min": 0.0,
        "max": 1.0,
        
        #------------MUTATION PARAMS---------#
        # selector ["uniform", "random_search"]
        "selector" : args.selector,
        # mutation operator ["iso_dd", "polynomial", "sbx"]
        "mutation" : args.mutation,
    
        # probability of mutating each number in the genotype
        "mutation_prob": 0.2,

        # param for 'polynomial' mutation for variation operator
        "eta_m": 10.0,
        
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,

        #--------UNSTURCTURED ARCHIVE PARAMS----#
        # l value - should be smaller if you want more individuals in the archive
        # - solutions will be closer to each other if this value is smaller.
        "nov_l": 0.015,
        "eps": 0.1, # usually 10%
        "k": 15,  # from novelty search

        "log_time_stats": False, 
        # 0 for random emiiter
        # 1 for optimizing emitter
        # 2 for random walk emitter
        # 3 for model disagreement emitter
        "emitter_selection": 0,
        
    }
    
    dim_x = 36 #genotype size
    obs_dim = 48
    action_dim = 18
    
    # Deterministic = "det", Probablistic = "prob" 
    dynamics_model_type = "det" 
    dynamics_model, dynamics_model_trainer = get_dynamics_model(dynamics_model_type)

    if args.dynamics_model_path != None:
        print("Loading pretrained dynamics model from:", args.dynamics_model_path)
        dynamics_model = ptu.load_model(dynamics_model, args.dynamics_model_path)

    env = HexapodEnv(dynamics_model=dynamics_model,
                     render=False,
                     record_state_action=True,
                     ctrl_freq=100)
    
    f_real = env.evaluate_solution
    
    qd = QD(args.dim_map, dim_x,
            f_real,
            n_niches=args.n_niches,
            params=px, log_dir=args.log_dir)
    
    qd.compute(num_cores_set=args.num_cores, max_evals=args.max_evals)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #-----------------Type of QD---------------------#
    # options are 'cvt', 'grid' and 'unstructured'
    parser.add_argument("--qd_type", type=str, default="unstructured")
    
    #---------------CPU usage-------------------#
    parser.add_argument("--num_cores", type=int, default=8)
    
    #-----------Store results + analysis-----------#
    parser.add_argument("--log_dir", type=str)
    
    #-----------QD params for cvt or GRID---------------#
    # ONLY NEEDED FOR CVT OR GRID MAP ELITES - not needed for unstructured archive
    parser.add_argument("--dim_map", default=6, type=int) # Dim of behaviour descriptor
    parser.add_argument("--grid_shape", default=[100,100], type=list) # num discretizat
    parser.add_argument("--n_niches", default=5000, type=int)

    #----------population params--------#
    parser.add_argument("--b_size", default=200, type=int) # For parralellization - 
    parser.add_argument("--dump_period", default=5000, type=int) 
    parser.add_argument("--max_evals", default=1e6, type=int) # max number of evaluation
    parser.add_argument("--selector", default="uniform", type=str)
    parser.add_argument("--mutation", default="iso_dd", type=str)

    #---------load dynamics model pretrained weights------------#
    parser.add_argument("--dynamics_model_path", default=None, type=str)
    
    args = parser.parse_args()
    
    main(args)
