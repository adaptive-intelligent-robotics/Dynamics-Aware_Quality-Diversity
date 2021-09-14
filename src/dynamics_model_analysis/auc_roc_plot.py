import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_quality(data, qua_threshold): 
    tpr_list =[]
    fpr_list =[]
    
    for t_qua in qua_threshold:
        data['>'+str(t_qua)] = data['Fitness score'].apply(lambda x: 1 if x>t_qua else 0)
        tp = data[(data["GT labels"]==1)&(data['>'+str(t_qua)]==1)]
        fp = data[(data["GT labels"]==0)&(data['>'+str(t_qua)]==1)]
        tn = data[(data["GT labels"]==0)&(data['>'+str(t_qua)]==0)]
        fn = data[(data["GT labels"]==1)&(data['>'+str(t_qua)]==0)]
        #print(tp.shape[0], fp.shape[0], tn.shape[0], fn.shape[0])
        
        tpr = tp.shape[0]/(tp.shape[0]+fn.shape[0])
        fpr = fp.shape[0]/(tn.shape[0]+fp.shape[0])

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return tpr_list, fpr_list

def plot_novelty(data, nov_threshold):
    tpr_list =[]
    fpr_list =[]

    for t_nov in nov_threshold:
        data['>'+str(t_nov)] = data['Novelty score'].apply(lambda x: 1 if x>t_nov else 0)
        tp = data[(data["GT labels"]==1)&(data['>'+str(t_nov)]==1)]
        fp = data[(data["GT labels"]==0)&(data['>'+str(t_nov)]==1)]
        tn = data[(data["GT labels"]==0)&(data['>'+str(t_nov)]==0)]
        fn = data[(data["GT labels"]==1)&(data['>'+str(t_nov)]==0)]
        
        #print(tp.shape[0], fp.shape[0], tn.shape[0], fn.shape[0])

        tpr = tp.shape[0]/(tp.shape[0]+fn.shape[0])
        fpr = fp.shape[0]/(tn.shape[0]+fp.shape[0])

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return tpr_list, fpr_list



def plot_quality_direct_model(data, qua_threshold): 
    tpr_list =[]
    fpr_list =[]
    
    for t_qua in qua_threshold:
        data['>'+str(t_qua)] = data['Fitness score direct'].apply(lambda x: 1 if x>t_qua else 0)
        tp = data[(data["GT labels"]==1)&(data['>'+str(t_qua)]==1)]
        fp = data[(data["GT labels"]==0)&(data['>'+str(t_qua)]==1)]
        tn = data[(data["GT labels"]==0)&(data['>'+str(t_qua)]==0)]
        fn = data[(data["GT labels"]==1)&(data['>'+str(t_qua)]==0)]
        #print(tp.shape[0], fp.shape[0], tn.shape[0], fn.shape[0])
        
        tpr = tp.shape[0]/(tp.shape[0]+fn.shape[0])
        fpr = fp.shape[0]/(tn.shape[0]+fp.shape[0])

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return tpr_list, fpr_list

def plot_novelty_direct_model(data, nov_threshold):
    tpr_list =[]
    fpr_list =[]

    
    for t_nov in nov_threshold:
        data['>'+str(t_nov)] = data['Novelty score direct'].apply(lambda x: 1 if x>t_nov else 0)
        tp = data[(data["GT labels"]==1)&(data['>'+str(t_nov)]==1)]
        fp = data[(data["GT labels"]==0)&(data['>'+str(t_nov)]==1)]
        tn = data[(data["GT labels"]==0)&(data['>'+str(t_nov)]==0)]
        fn = data[(data["GT labels"]==1)&(data['>'+str(t_nov)]==0)]
        
        #print(tp.shape[0], fp.shape[0], tn.shape[0], fn.shape[0])

        tpr = tp.shape[0]/(tp.shape[0]+fn.shape[0])
        fpr = fp.shape[0]/(tn.shape[0]+fp.shape[0])

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return tpr_list, fpr_list



# baseline of the 0.5 AUC 
plt.plot([0.,1.], [0.,1.], '-.')

qua_threshold = np.arange(-2, 2, 0.05)
nov_threshold = np.arange(0, 0.05, 0.0001)
qua_threshold_dir = np.arange(-2, 2, 0.05)
nov_threshold_dir = np.arange(0, 0.05, 0.0001)

data = pd.read_csv("aucroc_data_compare.csv")
print(data)
tpr_qua, fpr_qua = plot_quality(data, qua_threshold)
tpr_nov, fpr_nov = plot_novelty(data, nov_threshold)
tpr_qua_used, fpr_qua_used = plot_quality(data, [0.0])
tpr_nov_used, fpr_nov_used = plot_novelty(data, [0.01])

tpr_qua_dir, fpr_qua_dir = plot_quality_direct_model(data, qua_threshold)
tpr_nov_dir, fpr_nov_dir = plot_novelty_direct_model(data, nov_threshold)
tpr_qua_dir_used, fpr_qua_dir_used = plot_quality_direct_model(data, [0.0])
tpr_nov_dir_used, fpr_nov_dir_used = plot_novelty_direct_model(data, [0.01])

plt.plot(fpr_qua, tpr_qua, 'x-', label="train int test int - qua dyn")
plt.plot(fpr_nov, tpr_nov, 'x-', label="train int test int - nov dyn")
plt.plot(fpr_qua_dir, tpr_qua_dir, 'o-', label="train int test int - qua dir")
plt.plot(fpr_nov_dir, tpr_nov_dir, 'o-', label="train int test int - nov dir")

plt.plot(fpr_qua_used, tpr_qua_used, 'X', ms=15, label="used - qua dyn")
plt.plot(fpr_nov_used, tpr_nov_used, 'X', ms=15, label="used - nov dyn")
plt.plot(fpr_qua_dir_used, tpr_qua_dir_used, 'X',ms=15,  label="used - qua dir")
plt.plot(fpr_nov_dir_used, tpr_nov_dir_used, 'X',ms=15, label="used - nov dir")


'''
data = pd.read_csv("aucroc_data.csv")
print(data)
tpr_qua, fpr_qua = plot_quality(data)
tpr_nov, fpr_nov = plot_novelty(data)
#plt.plot(fpr_qua, tpr_qua, 'x-', label="train int test int - qua")
plt.plot(fpr_nov, tpr_nov, 'x-', label="train int test int - nov")

data_2 = pd.read_csv("aucroc_data_2.csv")
print(data_2)
tpr_qua_2, fpr_qua_2 = plot_quality(data_2)
tpr_nov_2, fpr_nov_2 = plot_novelty(data_2)
#plt.plot(fpr_qua_2, tpr_qua_2, 'x-', label="train int test int - qua")
plt.plot(fpr_nov_2, tpr_nov_2, 'x-', label="train int test int - nov")

data_3 = pd.read_csv("aucroc_data_3.csv")
print(data_3)
tpr_qua_3, fpr_qua_3 = plot_quality(data_3)
tpr_nov_3, fpr_nov_3 = plot_novelty(data_3)
#plt.plot(fpr_qua_3, tpr_qua_3, 'x-', label="train final test final - qua")
plt.plot(fpr_nov_3, tpr_nov_3, 'x-', label="train final test final - nov")

data_4 = pd.read_csv("aucroc_data_4.csv")
print(data_4)
tpr_qua_4, fpr_qua_4 = plot_quality(data_4)
tpr_nov_4, fpr_nov_4 = plot_novelty(data_4)
#plt.plot(fpr_qua_4, tpr_qua_4, 'x-', label="train final test int - qua")
plt.plot(fpr_nov_4, tpr_nov_4, 'x-', label="train final test int - nov")
'''


plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()
