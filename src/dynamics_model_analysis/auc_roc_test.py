import os
import sys
import glob

# Pandas for managing datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics

variant_names = ["det_dyn_me_trainfreq10/",
                 "det_dyn_me_trainfreq20/",
                 "direct_nn_stats/",]

def get_files(variant, filename=""):
    #if(len(arg)!=0):
    #    base = arg[0]
    #else:
    #    base = "."

    base = "./hpc_results/"
    if filename != "":
        filename ="/"+filename

    #return glob.glob(base+'/results_' + variant+"_"+exp+'/202*/' + filename)
    return glob.glob(base+"/"+variant+"/*")

def collect_data(filename = "log_file.dat",
                 fields = ["gen","num_evals","num_model_evals","archive_size",
                           "best_fit", "qd_score", "mean_fit", "median_fit",
                           "per_5", "per_95",
                           "true_pos", "false_pos", "false_neg", "true_neg"]):
    data = pd.DataFrame()
    for variant in variant_names:
        files = get_files(variant,filename)
        print(files)
        
        if(len(files)==0):
            print("NO file called " + filename + " for "+exp+" "+variant)
            continue

        # use a data tmp to store all the data for one variant for one experiment
        # append the data tmp to the full dataframe file later on
        data_tmp = pd.DataFrame()
        for run, f in enumerate(files): 
v            tmp = pd.read_csv(f,delim_whitespace=True,names=fields)
            tmp['run']=run # replication of the same variant
            data_tmp=pd.concat([data_tmp, tmp], ignore_index=True)

        # add variant and experinment name as fieds as well
        data_tmp['variant'] = variant
        #data_tmp['exp'] = exp

        # all variants added into same big overall dataframe and variant used as hue in plot
        data = data.append(data_tmp)

    return data


def main():

    data = collect_data()

    # at gen 1000
    data = data.loc[data["gen"] == 1000]
    
    data["qd_score_norm"] = data["qd_score"] + data["archive_size"]*100

    # stats
    data["precision"] = data["true_pos"]/(data["true_pos"]+data["false_pos"]) 
    data["recall"] = data["true_pos"]/(data["true_pos"]+data["false_neg"]) 

    data["num_pos"] = data["true_pos"] + data["false_pos"]
    data["num_neg"] = data["true_neg"] + data["false_neg"]
    
    # normalize data
    data["true_pos_norm"] = data["true_pos"]/200
    data["false_pos_norm"] = data["false_pos"]/200
    data["false_neg_norm"] = data["false_neg"]/200
    data["true_neg_norm"] = data["true_neg"]/200

    # for auc roc
    data["tpr"] = data["true_pos"]/(data["true_pos"]+data["false_neg"]) 
    data["specificity"] = data["true_neg"]/(data["true_neg"]+data["false_pos"]) 
    data["fpr"] = 1 - data["specificity"]

    
    sns.set(style="whitegrid")
    #sns.palplot(sns.color_palette("colorblind"))
    # Plot the responses for different events and regions
    
    plt.figure()
    #my_lineplot=sns.lineplot
    #_LinePlotter.aggregate = first_second_third_quartile
    sns_plot = sns.lineplot(x="fpr", y="tpr",
                            hue="variant", 
                            data=data)


    print("HERE")
    #plt.title(exp+"_"+item)
    #plt.savefig("./stats_plots/direct_nn/progress_me_baseline"+item+".svg")
    plt.show()


main()
