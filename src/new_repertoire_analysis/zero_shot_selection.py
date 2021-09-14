import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.torch.pytorch_util as ptu
from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.models.dynamics_models.deterministic_model import DeterministicDynModel

import glob
from multiprocessing import Pool


dynamics_model = DeterministicDynModel(48,18,500)
dynamics_model = ptu.load_model(dynamics_model, "hexapod_detdyn_archcond_trained.pth")
env = HexapodEnv(dynamics_model=dynamics_model, render=True, ctrl_freq=100)

def load_dataset(filename):
    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma                                                                                                       
    # 44 columns                                                                                                                                                                                
    # fitness, bd1-bd6, disagreement, genotype(36dim)
    
    print("data shape:", data.shape)
    #print(data.iloc[0,:])                                                                                                                                                                       
    x = data.iloc[:,-36:].to_numpy()
    fitness = data.iloc[:,0].to_numpy()
    bd = data.iloc[:,1:7].to_numpy()
    disagr = data.iloc[:,7].to_numpy()

    return x, fitness, bd, disagr


def get_zero_shot_results():

    base_dir_syn_arch = "results_hexapod_continual_probens/archives_100000"
    files_syn_arch = glob.glob(base_dir_syn_arch+'/*')
    print("Synthetic archive files: ",files_syn_arch)

    #filename_syn_arch = "results_hexapod_continual_probens/archives_100000/archive_model_gen_500_rep4.dat"

    save_fit_list = []
    i = 0
    for filename in files_syn_arch:
        # load the synthetic archive
        x, fitness, bd, disagr = load_dataset(filename)
        
        print(x.shape, fitness.shape, bd.shape, disagr.shape)
        
        # See distirbution of the disagreements
        
        
        # select individual/index with the lowest disagreement or lower than avergage disagreement that has highest fitness
        selected_idxs = np.argsort(disagr)[:20]  #np.argmax(fitness) #np.argmin(disagr)
        fitnesses = fitness[selected_idxs]
        selected_idx = np.argsort(fitnesses)[-1]

        video_filename = "zero_shot_rep_"+str(i)+".mp4"
        
        # execute solution in a real evaluation
        gt_fit, _, _, _ = env.evaluate_solution_uni(x[selected_idx], render=True, video_name=video_filename)
        #print("Expected_fitness: ", fitness[selected_idx])
        print("Ground truth fitness: ", gt_fit)

        save_fit_list.append(gt_fit)
        i+=1

    save_fit = np.array(save_fit_list)
    print("Average zero shot fitness performance: ", np.mean(save_fit), "+-", np.std(save_fit))
    print("Average zero shot fitness performance: ", np.percentile(save_fit, 50), "[", np.percentile(save_fit, 25), ",", np.percentile(save_fit, 75), "]")
    

def get_top_disagreement_results():

    base_dir_syn_arch = "results_hexapod_continual_probens/archives_100000"
    files_syn_arch = glob.glob(base_dir_syn_arch+'/*')
    print("Synthetic archive files: ",files_syn_arch)

    #filename_syn_arch = "results_hexapod_continual_probens/archives_100000/archive_model_gen_500_rep4.dat"

    best_fit_list = []
    avg_fit_list = []

    i = 0
    for filename in files_syn_arch:
        # load the synthetic archive
        x, fitness, bd, disagr = load_dataset(filename)
        
        print(x.shape, fitness.shape, bd.shape, disagr.shape)
        
        # See distirbution of the disagreements
        
        # select individual/index with the lowest disagreement or lower than avergage disagreement that has highest fitness
        selected_idxs = np.argsort(disagr)[:20]  #np.argmax(fitness) #np.argmin(disagr)

        x = x[selected_idxs]
        number_processes = 32
        with Pool(number_processes) as pool:
            list_fit, list_bd, _, _ = list(zip(*pool.map(env.evaluate_solution_uni, x)))

        fit_arr = np.array(list_fit)
        
        avg_fit_list.append(np.mean(fit_arr))
        best_fit_list.append(np.amax(fit_arr))

        # few shot save best skill video
        best_skill_idx = np.argmax(fit_arr)
        video_filename = "few_shot_rep_"+str(i)+".mp4"
        gt_fit, _, _, _ = env.evaluate_solution_uni(x[best_skill_idx], render=True, video_name=video_filename)
        print("GT FITNESS: ", gt_fit)
        i += 1
        

    avg_fit = np.array(avg_fit_list)
    best_fit = np.array(best_fit_list)
    
    print("Average fitness performance - top 20 disagr: ", np.mean(avg_fit), "+-", np.std(avg_fit))    
    print("Best fitness performance - top 20 disagr: ", np.mean(best_fit), "+-", np.std(best_fit))

    print("Average fitness performance - top 20 disagr: ", np.percentile(avg_fit, 50), "[", np.percentile(avg_fit, 25),  ",", np.percentile(avg_fit, 75), "]")    
    print("Best fitness performance - top 20 disagr: ", np.percentile(best_fit, 50), "[", np.percentile(best_fit, 25),  ",", np.percentile(best_fit, 75), "]")

    

if __name__ == "__main__":


    #main()
    get_zero_shot_results()
    #get_top_disagreement_results() # few shot
