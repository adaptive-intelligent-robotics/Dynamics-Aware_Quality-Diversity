import os
import sys
import glob

# Pandas for managing datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

variant_names = ["direct_nn_me/",
                 "det_dyn_me_trainfreq50_nostats/",
                 "det_dyn_me_trainfreq20_nostats/",
                 "det_dyn_me_trainfreq50_rand_emit/",]

variant_names = ["det_dyn_me_trainfreq50_rand_emit_with_count/",]
#variant_names = ["baseline_det_dyn",]


def get_files(variant, filename=""):
    
    base = "./time_data/"
    if filename != "":
        filename ="/"+filename

    return glob.glob(base+"/"+variant+"/*")


def collect_data(filename = "time_log_file.dat",
                 fields = ["gen","gen_time","model_eval_time","eval_time",
                           "model_train_time","real_evals", "model_evals"]):
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
            tmp = pd.read_csv(f,delim_whitespace=True,names=fields)
            tmp['run']=run # replication of the same variant
            data_tmp=pd.concat([data_tmp, tmp], ignore_index=True)

        # add variant and experinment name as fieds as well
        data_tmp['variant'] = variant
        #data_tmp['exp'] = exp

        # all variants added into same big overall dataframe and variant used as hue in plot
        data = data.append(data_tmp)

    return data


def plot():

    data = collect_data()

    data["model_eval_time_norm"] = data["model_eval_time"]/data["gen_time"]
    data["eval_time_norm"] = data["eval_time"]/data["gen_time"]
    data["model_train_time_norm"] = data["model_train_time"]/data["gen_time"]
    
    sns.set(style="whitegrid")
    #sns.palplot(sns.color_palette("colorblind"))
    # Plot the responses for different events and regions

    plt.figure()
    sns_plot_1 = sns.lineplot(x="gen", y="model_eval_time",
                              data=data,label="model eval time")
    sns_plot_2 = sns.lineplot(x="gen", y="eval_time", 
                              data=data, label="real eval time")
    plt.ylim([0,200])
    
    #sns_plot_3 = sns.lineplot(x="gen", y="model_train_time_norm", 
    #                          data=data, label="model train")

    #sns_plot_3 = sns.lineplot(x="gen", y="model_evals",
    #                          data=data,label="num of model eval")
    #sns_plot_4 = sns.lineplot(x="gen", y="real_evals", 
    #                          data=data, label="num of real eval")

    
    print("HERE")
    #plt.title(exp+"_"+item)
    #plt.savefig("./stats_plots/direct_nn/progress_me_baseline"+item+".svg")
    plt.legend()
    plt.show()

    '''
    for item in ["archive_size"]:
    #for item in ["qd_score_norm"]:    
    #for item in ["archive_size","best_fit", "mean_fit"]:
    #for item in ["archive_size","num_pos","num_neg","precision", "recall", "true_pos_norm","false_pos_norm", "false_neg_norm","true_neg_norm"]:
    #for item in ["false_pos", "false_neg","true_neg"]:
    #for item in ["false_neg","true_neg"]:
    #for item in ["true_neg"]:

        plt.figure()
        #my_lineplot=sns.lineplot
        #_LinePlotter.aggregate = first_second_third_quartile
        sns_plot = sns.lineplot(x="num_evals", y=item,
                                hue="variant", 
                                data=data )

        #sns_plot = sns.lineplot(x="gen", y=item,
        #                        hue="variant", 
        #                        data=data )

        print("HERE")
        #plt.title(exp+"_"+item)
        #plt.savefig("./stats_plots/direct_nn/progress_me_baseline"+item+".svg")
        plt.show()
    '''

plot()
