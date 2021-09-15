import os
import sys
import glob

# Pandas for managing datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.relational import _LinePlotter

variant_names = ["random_exploration/",
                 "vanilla_me/",
                 "direct_model_archcond/",
                 #"dyn_model_det_archcond/",
                 #"dyn_model_probens_archcond/",
                 "dyn_model_probens_archcond_2/",
]                 

def get_files(variant, filename=""):
    #if(len(arg)!=0):
    #    base = arg[0]
    #else:
    #    base = "."

    base = "./hpc_results/hexapod_exp_015/"
    #base = "./hpc_results/pushing_ee_025/"

    if filename != "":
        filename ="/"+filename

    #return glob.glob(base+'/results_' + variant+"_"+exp+'/202*/' + filename)
    return glob.glob(base+"/"+variant+"/*")

def collect_data(filename = "log_file.dat",):

    model_fields = ["gen","num_evals","num_model_evals","archive_size",
                    "best_fit", "qd_score", "mean_fit", "median_fit",
                    "per_5", "per_95",
                    "true_pos", "false_pos", "false_neg", "true_neg"]
    
    baseline_fields = ["gen","num_evals","archive_size",
                       "best_fit", "qd_score", "mean_fit", "median_fit",
                       "per_5", "per_95",
                       "true_pos", "false_pos", "false_neg", "true_neg"]
    
    data = pd.DataFrame()
    interpolated_data = pd.DataFrame()
    
    for variant in variant_names:
        files = get_files(variant,filename)
        print(files)
        
        if(len(files)==0):
            print("NO file called " + filename + " for "+exp+" "+variant)
            continue
        
        # use a data tmp to store all the data for one variant for one experiment
        # append the data tmp to the full dataframe file later on
        data_tmp = pd.DataFrame()
        data_tmp_2 = pd.DataFrame() # for interpolation data
        
        for run, f in enumerate(files):

            if ("random_exploration" in f) or ("vanilla_me" in f):
                fields = baseline_fields
                print("Baseline field: ", f)
            else:
                fields = model_fields

            #fields = model_fields
                
            tmp = pd.read_csv(f,delim_whitespace=True,names=fields)

            #sns_plot = sns.lineplot(x="num_evals", y="archive_size",data=tmp)

            tmp['run']=run # replication of the same variant

            if "dyn_model_probens_archcond_2" in f:
                data_int = interpolate_data(tmp, end_point=80000) #interpolated data we want
            else:
                data_int = interpolate_data(tmp) #interpolated data we want

            #sns_plot = sns.lineplot(x="num_evals", y="archive_size", data=data_int, marker="x", markersize=3, 'k')
                
            data_tmp=pd.concat([data_tmp, tmp], ignore_index=True)
            data_tmp_2=pd.concat([data_tmp_2, data_int], ignore_index=True)

        plt.show()
        # add variant and experinment name as fieds as well
        data_tmp['variant'] = variant
        data_tmp_2['variant'] = variant
        
        #data_tmp['exp'] = exp
        
        # all variants added into same big overall dataframe and variant used as hue in plot
        data = data.append(data_tmp)
        interpolated_data = interpolated_data.append(data_tmp_2)

    return data, interpolated_data

    
def interpolate_data(data, end_point=1000001):

    data_points = pd.DataFrame()
    #end_point = data["num_evals"].iloc[-1]
    #end_point = 1000000
    #print(end_point)
    interested_points = np.arange(0,end_point,200)
    data_points["num_evals"] = interested_points
    #print(data_points.shape)
    data = data.append(data_points)
    data = data.sort_values(by="num_evals")
    #print(data)
    data['archive_size'] = data['archive_size'].interpolate(method='linear')
    data['qd_score'] = data['qd_score'].interpolate(method='linear')
    #print(data)

    interested_data = data.loc[data['num_evals'].isin(interested_points)]
    interested_data = interested_data[pd.isnull(interested_data["mean_fit"])]
    #interested_data = interested_data.round(2)
    #print(interested_data)

    #s = interested_data.to_csv("interpolated_data.csv")
    
    #print(s)
    return interested_data


def plot():

    data, interpolated_data = collect_data()

    # qd score is the sum of fitness
    # to normalise it - sum each value of fitness by pi and divide by pi
    data["qd_score_norm"] = data["qd_score"] + data["archive_size"]*np.pi
    interpolated_data["qd_score_norm"] = (interpolated_data["qd_score"] + interpolated_data["archive_size"]*np.pi)/np.pi

    '''
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
    '''
    
    sns.set(style="whitegrid")
    #sns.palplot(sns.color_palette("colorblind"))
    # Plot the responses for different events and regions

    def first_second_third_quartile(self, vals, grouper, units=None):
        # Group and get the aggregation estimate
        grouped = vals.groupby(grouper, sort=self.sort)
        est = grouped.agg('median')
        min_val = grouped.quantile(0.25)
        max_val = grouped.quantile(0.75)
        cis = pd.DataFrame(np.c_[min_val, max_val],
                           index=est.index,
                           columns=["low", "high"]).stack()
        
        # Unpack the CIs into "wide" format for plotting
        if cis.notnull().any():
            cis = cis.unstack().reindex(est.index)
        else:
            cis = None
            
        return est.index, est, cis

    for item in ["archive_size"]:
    #for item in ["qd_score_norm"]:    
    #for item in ["archive_size","best_fit", "mean_fit"]:
    #for item in ["archive_size","num_pos","num_neg","precision", "recall", "true_pos_norm","false_pos_norm", "false_neg_norm","true_neg_norm"]:
    #for item in ["num_pos", "num_neg"]:
    #for item in ["true_pos_norm","false_pos_norm", "false_neg_norm","true_neg_norm"]:
    #for item in ["true_pos","false_pos", "false_neg","true_neg"]:
    #for item in ["false_neg","true_neg"]:
    #for item in ["true_neg"]:

        plt.figure()
        my_lineplot=sns.lineplot
        _LinePlotter.aggregate = first_second_third_quartile
        #sns_plot = sns.lineplot(x="num_evals", y=item,
        #                        hue="variant", 
        #                        data=data)

        #sns_plot = sns.lineplot(x="gen", y=item,
        #                        hue="variant", 
        #                        data=data )

        sns_plot = sns.lineplot(x="num_evals", y=item,
                                hue="variant", 
                                data=interpolated_data)

        print("HERE")
        #plt.title(exp+"_"+item)
        #plt.savefig("./stats_plots/direct_nn/progress_me_baseline"+item+".svg")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.vlines(50000, ymin=0, ymax=1000, linestyles="dashed")

        plt.show()


plot()
