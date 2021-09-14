import numpy as np


def main():
    
    data = np.load("transfer_learning_analysis_results_120621.npz", allow_pickle=True)
    
    mean_fit_err = data["mean_fit_err"] 
    mean_bd_err = data["mean_bd_err"] 

    best_fit_archive_direct = data["best_fit_ad"]
    avg_fit_archive_direct = data["avg_fit_ad"]
    archive_size_archive_direct = data["archive_size_ad"]

    best_fit_archive_condition = data["best_fit_ac"]
    avg_fit_archive_condition = data["avg_fit_ac"]
    archive_size_archive_condition = data["archive_size_ac"]

    best_fit_baseline = data["best_fit_base"]
    avg_fit_baseline = data["avg_fit_base"]
    archive_size_baseline = data["archive_size_base"]

    #print(mean_fit_err.shape)
    #print(mean_fit_err[4].shape)

    ##### MEAN AND STD DEVIATION ##########
    print("#### Synthetic Archive - reeval Direct Add ####")
    print("Best fitness: ", np.mean(best_fit_archive_direct), "+-", np.std(best_fit_archive_direct))
    print("Avg fitness: ", np.mean(avg_fit_archive_direct), "+-", np.std(avg_fit_archive_direct))
    print("Archive size: ", np.mean(archive_size_archive_direct), "+-", np.std(archive_size_archive_direct))
    
    print("#### Synthetic Archive - reeval Addition Condition ####")
    print("Best fitness: ", np.mean(best_fit_archive_condition), "+-", np.std(best_fit_archive_condition))
    print("Avg fitness: ", np.mean(avg_fit_archive_condition), "+-", np.std(avg_fit_archive_condition))
    print("Archive size: ", np.mean(archive_size_archive_condition), "+-", np.std(archive_size_archive_condition))
    
    print("#### Baseline Archives ####")
    print("Best fitness: ", np.mean(best_fit_baseline), "+-", np.std(best_fit_baseline))
    print("Avg fitness: ", np.mean(avg_fit_baseline), "+-", np.std(avg_fit_baseline))
    print("Archive size: ", np.mean(archive_size_baseline), "+-", np.std(archive_size_baseline))


    ######## MEDIAN AND QUARTILES ############
    print("#### Synthetic Archive - reeval Direct Add ####")
    print("Best fitness: ", np.percentile(best_fit_archive_direct, 50), " [", np.percentile(best_fit_archive_direct, 25),",", np.percentile(best_fit_archive_direct, 75),"]")
    print("Avg fitness: ", np.percentile(avg_fit_archive_direct, 50), " [", np.percentile(avg_fit_archive_direct, 25), ",",np.percentile(avg_fit_archive_direct, 75), "]")
    print("Archive size: ", np.percentile(archive_size_archive_direct, 50), " [", np.percentile(archive_size_archive_direct, 25),",", np.percentile(archive_size_archive_direct, 75), "]")

    print("#### Synthetic Archive - reeval Addition condition ####")
    print("Best fitness: ", np.percentile(best_fit_archive_condition, 50), " [", np.percentile(best_fit_archive_condition, 25),",", np.percentile(best_fit_archive_condition, 75),"]")
    print("Avg fitness: ", np.percentile(avg_fit_archive_condition, 50), " [", np.percentile(avg_fit_archive_condition, 25), ",",np.percentile(avg_fit_archive_condition, 75), "]")
    print("Archive size: ", np.percentile(archive_size_archive_condition, 50), " [", np.percentile(archive_size_archive_condition, 25),",", np.percentile(archive_size_archive_condition, 75), "]")

    print("#### Baseline Archives ####")
    print("Best fitness: ", np.percentile(best_fit_baseline, 50), " [", np.percentile(best_fit_baseline, 25),",", np.percentile(best_fit_baseline, 75),"]")
    print("Avg fitness: ", np.percentile(avg_fit_baseline, 50), " [", np.percentile(avg_fit_baseline, 25), ",",np.percentile(avg_fit_baseline, 75), "]")
    print("Archive size: ", np.percentile(archive_size_baseline, 50), " [", np.percentile(archive_size_baseline, 25),",", np.percentile(archive_size_baseline, 75), "]")
    
   



    

main()
