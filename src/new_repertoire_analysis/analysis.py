import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.torch.pytorch_util as ptu
from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.models.dynamics_models.deterministic_model import DeterministicDynModel


from src.map_elites import common as cm
from src.map_elites import unstructured_container

import glob
from multiprocessing import Pool

dynamics_model = DeterministicDynModel(48,18,500)
dynamics_model = ptu.load_model(dynamics_model, "hexapod_detdyn_archcond_trained.pth")
env = HexapodEnv(dynamics_model=dynamics_model, render=False, ctrl_freq=100)

def load_dataset(filename):
    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma     
    # 44 columns                                                                              
    # fitness, bd1-bd6, disagreement, genotype(36dim)

    print("data shape:", data.shape)
    #print(data.iloc[0,:])
    x = data.iloc[:,-36:]
    y = data.iloc[:,0:7]

    x = x.to_numpy()
    y = y.to_numpy()

    print("x dataset shape: ", x.shape)
    print("y dataset shape: ", y.shape)

    dataset = [x,y]
    return dataset

def get_gt_results(filename):
    # get ground truth results of the synthetic archive by re-evaluating all genotypes found my syntehtic archive
    # Load in archive or training data corresponding to trained models
    #filename = "test_results_hexapod_uni_continual_3/archive_model_gen_980.dat"
    dataset = load_dataset(filename)
    
    x,y = dataset
    fit_gt = []
    bd_gt = []
    i=0

    number_processes = 32
    with Pool(number_processes) as pool:
        list_fit, list_bd, _, _ = list(zip(*pool.map(env.evaluate_solution_uni, x)))
        
    '''    
    for gen in x:
        if i%50 ==0:
            print(i)
        fit, bd, _, _ = env.evaluate_solution_uni(gen)
        fit_gt.append(fit)
        bd_gt.append(bd)
        i+=1
    '''
    print("List fit: ", len(list_fit))
    print("List bd: ", len(list_bd))
        
    x = np.array(x)
    fit_gt = np.array(list_fit) #fit_gt)
    bd_gt = np.squeeze(np.array(list_bd)) #bd_gt)

    #print("fit gt shape: ",fit_gt.shape)
    #print("bd gt shape: ",bd_gt.shape)
    #np.savez("gt_archive_gen_980.npz", fit=fit_gt, bd=bd_gt, gen=x)
    # return the ground turth
    return fit_gt, bd_gt, x


def compute_batch_bd_error(bd_gt, bd_m):
    bd_gt = np.array(bd_gt)
    bd_m = np.array(bd_m)
    #print("bd gt shape: ", bd_gt.shape)
    #print("bd m shape: ", bd_m.shape)
    err = np.abs(bd_gt - bd_m)
    print("Err shape: ", err.shape)
    #print("Err: ", err)
    
    # for BD - norm of the error across all the bd dimensions
    #bd_err = np.linalg.norm(err, axis=-1)
    bd_err = np.mean(err, axis=-1)
    print("BD Err shape: ", bd_err.shape)
    
    return bd_err

def compute_fitness_error(fit_gt, fit_m):
    # can do batch or individual
    fit_gt = np.array(fit_gt)
    fit_m = np.array(fit_m)

    # use absolute value
    fit_err = np.abs(fit_gt-fit_m)
    return fit_err


def create_ground_truth_repertoire(z, fit_gt, bd_gt):
    # direct add to the archive of all solutions in the model archive 
    print("Creating gt repertoire/archive - re-evaluated from synthetic archive - direct add")
    num_inds = fit_gt.shape[0]
    #print("fit gt: ",fit_gt[0])
    #print("bd gt: ", bd_gt)
    
    gt_archive = []
    for i in range(num_inds):
        s = cm.Species(z[i], bd_gt[i], fit_gt[i])
        gt_archive.append(s)
        
    #log_dir = "src/new_repertoire_analysis/"
    #params = {"type": "unstructured"}
    #cm.save_archive(gt_archive, "gt_980", params, log_dir)    

    return gt_archive


def create_ground_truth_real_repertoire(z, fit_gt, bd_gt):
    params = {"type": "unstructured",
              "nov_l": 0.2,
              "eps": 0.1, # usually 10%
              "k": 15,  # from novelty search
    }

    #  add to the archive according to the archive addition condition - 
    print("Creating gt repertoire/archive re-evaluated from synthetic archive -  addition condition")
    num_inds = fit_gt.shape[0]
    #print("fit gt: ",fit_gt[0])
    #print("bd gt: ", bd_gt)
    
    gt_archive = []
    for i in range(num_inds):
        s = cm.Species(z[i], bd_gt[i], fit_gt[i])
        gt_archive.append(s)

    final_archive = []    
    final_archive, add_list, _ = addition_condition(gt_archive, final_archive, params)
    print("Final archive size - addition condition: ", len(final_archive))

    #log_dir = "src/new_repertoire_analysis/"
    #cm.save_archive(final_archive, "gt_addcond_980", params, log_dir)    

    return final_archive


def addition_condition(s_list, archive, params):
    add_list = [] # list of solutions that were added
    discard_list = [] # list of solutions that were discarded
    for s in s_list:
        #if self.qd_type == "unstructured":
        success = unstructured_container.add_to_archive(s, archive, params)
        if success:
            add_list.append(s)
        else:
            discard_list.append(s) #not important for alogrithm but to collect stats
            
    return archive, add_list, discard_list
                                                                                                    

def get_baseline_results(base_dir_baseline_arch):

    best_fit_baseline = []
    avg_fit_baseline = []
    archive_size_baseline = []
    
    #######--- BASELINE ARCHIVES ---###########
    # load in baseline archives of task done using vanilla ME with no model - full evaluations
    files_baseline_arch = glob.glob(base_dir_baseline_arch+'/*')
    print("Baseline archive files: ",files_baseline_arch)
    
    for filename in files_baseline_arch:
        
        # load archive
        dataset = load_dataset(filename)
        x_base,y_base = dataset
        fit_baseline = y_base[:,0]
        bd_baseline = y_base[:,1:7]
              
        best_fit_baseline.append(np.amax(fit_baseline)) # compute best fitness
        avg_fit_baseline.append(np.mean(fit_baseline)) # compute avg fitness
        archive_size_baseline.append(fit_baseline.shape[0]) # compute archive size 

    print("#### Baseline Archives ####")
    print("Best fitness: ", np.mean(best_fit_baseline), "+-", np.std(best_fit_baseline))
    print("Avg fitness: ", np.mean(avg_fit_baseline), "+-", np.std(avg_fit_baseline))
    print("Archive size: ", np.mean(archive_size_baseline), "+-", np.std(archive_size_baseline))

    print("#### Baseline Archives - median and quartiles####")
    print("Best fitness: ", np.percentile(best_fit_baseline, 50), "[", np.percentile(best_fit_baseline, 25), ",", np.percentile(best_fit_baseline, 75), "]")
    print("Avg fitness: ", np.percentile(avg_fit_baseline, 50), "[", np.percentile(avg_fit_baseline, 25), ",", np.percentile(avg_fit_baseline, 75), "]")
    print("Archive size: ", np.percentile(archive_size_baseline, 50), "[", np.percentile(archive_size_baseline, 25), ",", np.percentile(archive_size_baseline, 75), "]")
    

def main(): 

    base_dir_syn_arch = "results_hexapod_continual_probens/archives_100000"
    base_dir_baseline_arch = "hpc_results/hexapod_uni_baseline/archives_100000"

    # DATA AND METRICS WE WANT AT THE END
    mean_bd_err = [] # Mean BD error per replication 
    mean_fit_err = [] # Mean Fitness error per replication

    # archive with direct add = replications
    best_fit_archive_direct = []
    avg_fit_archive_direct = [] 
    archive_size_archive_direct = [] 
    
    # archive with addition condition = replications
    best_fit_archive_condition = []
    avg_fit_archive_condition = [] 
    archive_size_archive_condition = [] 

    # baseline archives - replications
    best_fit_baseline = []
    avg_fit_baseline = [] 
    archive_size_baseline = [] 

    
    ######---- SYNTHETIC ARCHIVES ------ ######
    
    # load in syntehtic archive filenames from a folder containing
    files_syn_arch = glob.glob(base_dir_syn_arch+'/*')
    print("Synthetic archive files: ",files_syn_arch)
    
    for filename in files_syn_arch:        
        # get the ground turth re-evaluations of all
        fit_gt, bd_gt, z = get_gt_results(filename)

        # compute bd and fitness error of each ind in the synthetic archive vs its ground truth - reevaluation
        dataset = load_dataset(filename)
        x,y = dataset
        fit_m = y[:,0]
        bd_m = y[:,1:7]
        bd_err = compute_batch_bd_error(bd_gt, bd_m)
        fit_err = compute_fitness_error(fit_gt, fit_m)
        mean_bd_err.append(bd_err)
        mean_fit_err.append(fit_err)
        
        # get a GT archive of the synthetic archive with direct add
        gt_direct_add_archive = create_ground_truth_repertoire(z, fit_gt, bd_gt)

        # get archive metrics
        best_fit_archive_direct.append(np.amax(fit_gt)) # compute best fitness
        avg_fit_archive_direct.append(np.mean(fit_gt)) # compute avg fitness
        archive_size_archive_direct.append(len(gt_direct_add_archive)) # compute archive size 
        
        # get a GT archive of the synthetic archive with the addition condition
        gt_condition_archive = create_ground_truth_real_repertoire(z, fit_gt, bd_gt)
        fit_cond_arch = []
        bd_cond_arch = []
        for ind in gt_condition_archive:
            fit_cond_arch.append(ind.fitness)
            bd_cond_arch.append(ind.desc)

        fit_cond_arch =	np.array(fit_cond_arch)
        bd_cond_arch = np.array(bd_cond_arch)
            
        # get archive metrics
        best_fit_archive_condition.append(np.amax(fit_cond_arch)) # compute best fitness
        avg_fit_archive_condition.append(np.mean(fit_cond_arch)) # compute avg fitness
        archive_size_archive_condition.append(len(gt_condition_archive)) # compute archive size     
        
    #######--- BASELINE ARCHIVES ---###########
    # load in baseline archives of task done using vanilla ME with no model - full evaluations
    files_baseline_arch = glob.glob(base_dir_baseline_arch+'/*')
    print("Baseline archive files: ",files_baseline_arch)
    
    for filename in files_baseline_arch:
        
        # load archive
        dataset = load_dataset(filename)
        x_base,y_base = dataset
        fit_baseline = y_base[:,0]
        bd_baseline = y_base[:,1:7]
              
        best_fit_baseline.append(np.amax(fit_baseline)) # compute best fitness
        avg_fit_baseline.append(np.mean(fit_baseline)) # compute avg fitness
        archive_size_baseline.append(fit_baseline.shape[0]) # compute archive size 


    # DATA AND METRICS WE WANT AT THE END
    mean_bd_err = np.array(mean_bd_err)
    mean_fit_err = np.array(mean_fit_err) 
    # archive with direct add = replications
    best_fit_archive_direct = np.array(best_fit_archive_direct) 
    avg_fit_archive_direct = np.array(avg_fit_archive_direct) 
    archive_size_archive_direct = np.array(archive_size_archive_direct) 
    # archive with addition condition = replications
    best_fit_archive_condition = np.array(best_fit_archive_condition) 
    avg_fit_archive_condition = np.array(avg_fit_archive_condition) 
    archive_size_archive_condition = np.array(archive_size_archive_condition) 
    # baseline archives - replications
    best_fit_baseline = np.array(best_fit_baseline) 
    avg_fit_baseline = np.array(avg_fit_baseline) 
    archive_size_baseline = np.array(archive_size_baseline) 

    # NOTE: careful that we have not mean the bd and fit err yet for per replication - so we have the dsitirbution of errors for the archive saved
    print("BD Error: ", mean_bd_err)
    print("Fitness Error", mean_fit_err)
    print("Direct add archive: ", best_fit_archive_direct, avg_fit_archive_direct, archive_size_archive_direct)
    print("Addition condition archive: ", best_fit_archive_condition, avg_fit_archive_condition, archive_size_archive_condition)
    print("Baseline archive: ", best_fit_baseline, avg_fit_baseline, archive_size_baseline)

    np.savez("transfer_learning_analysis_results_120621.npz",
             mean_fit_err=mean_fit_err, mean_bd_err=mean_bd_err,
             best_fit_ad=best_fit_archive_direct, avg_fit_ad=avg_fit_archive_direct, archive_size_ad=archive_size_archive_direct,
             best_fit_ac=best_fit_archive_condition, avg_fit_ac=avg_fit_archive_condition, archive_size_ac=archive_size_archive_condition,
             best_fit_base=best_fit_baseline, avg_fit_base=avg_fit_baseline, archive_size_base=archive_size_baseline,
    )
    

if __name__ == "__main__":
    
    #main()

    
    base_dir_baseline_arch_1 = "hpc_results/hexapod_uni_baseline/archives_10000"
    base_dir_baseline_arch_2 = "hpc_results/hexapod_uni_baseline/archives_3000"
    base_dir_baseline_arch_3 = "hpc_results/hexapod_uni_baseline/archives_3500"
    #get_baseline_results(base_dir_baseline_arch_1)
    get_baseline_results(base_dir_baseline_arch_2)
    #get_baseline_results(base_dir_baseline_arch_3)
    



    
'''
#get_gt_results()
gt_data = np.load("src/new_repertoire_analysis/gt_archive_gen_980.npz")
fit_gt = gt_data["fit"]
bd_gt = gt_data["bd"][:,0,:]
print("bd gt shape: ", bd_gt.shape)

filename = "test_results_hexapod_uni_continual_3/archive_model_gen_980.dat"
dataset = load_dataset(filename)
x,y = dataset
fit_m = y[:,0]
bd_m = y[:,1:7]
print("bd model shape: ",bd_m.shape)


bd_err = compute_batch_bd_error(bd_gt, bd_m)
fit_err = compute_fitness_error(fit_gt, fit_m)
#print("bd error: ",bd_err.shape)
#print("fitness error: ", fit_err.shape)
print("Best fitness from ground truth archive: ", np.amax(fit_gt))
print("Mean fitness from ground truth archive: ", np.mean(fit_gt))


filename = "src/new_repertoire_analysis/archive_200100.dat"
dataset = load_dataset(filename)
x_base,y_base = dataset
fit_baseline = y_base[:,0]
bd_baseline = y_base[:,1:7]
print("Best fitness from baseline archive: ", np.amax(fit_baseline))
print("Mean fitness from baseline archive: ", np.mean(fit_baseline))


filename = "src/new_repertoire_analysis/archive_gt_addcond_980.dat"
dataset = load_dataset(filename)
x,y = dataset
fit_gt_addcond = y[:,0]
bd_gt_addcond = y[:,1:7]
print("Best fitness from gt archive addcond: ", np.amax(fit_gt_addcond))
print("Mean fitness from gt archive addcond: ", np.mean(fit_gt_addcond))


# CREATE GT ARHCHIVE/REPERTOIRE AND SAVE ARCHIVE AS DAT FILE
#gt_archive = create_ground_truth_repertoire(x, fit_gt, bd_gt)
#fg
gt_addcond_archive = create_ground_truth_real_repertoire(x, fit_gt, bd_gt)



# PLOT
#data = pd.DataFrame()
#data["BD Error"] = bd_err
#data["Fitness Error"] = fit_err

# Get names of indexes for which column Age has value 30                                                      
#indexNames = data[data["Fitness Error"] >= 1000].index
# Delete these row indexes from dataFrame                                                                     
#data.drop(indexNames, inplace=True)

#ax3 = sns.boxplot(y="BD Error", data=data)
#plt.ylim(bottom=0)
#plt.show()

#ax4 = sns.boxplot(y="Fitness Error", data=data) 
#plt.ylim(bottom=0)
#plt.show()

'''
