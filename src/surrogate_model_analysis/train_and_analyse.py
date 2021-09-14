import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import src.torch.pytorch_util as ptu

from src.models.surrogate_models.det_surrogate import DeterministicQDSurrogate
from src.trainers.qd.surrogate import SurrogateTrainer

from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble

def load_dataset(filename):

    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
    
    #print(data.shape)
    #print(data)
    # 41 columns
    # fitness, bd1, bd2, bd_ground1, bdground2, genotype(36dim)
    
    x = data.iloc[:,-36:]
    y = data.iloc[:,0:3]

    x = x.to_numpy()
    y = y.to_numpy()

    '''
    print("Y mean: ", np.mean(y, axis=0, keepdims=True))
    print("X mean: ", np.mean(x, axis=0, keepdims=True))

    print("Y std: ", np.std(y, axis=0, keepdims=True))
    print("X std: ", np.std(x, axis=0, keepdims=True))

    print("Y min: ", np.min(y, axis=0, keepdims=True))
    print("X min: ", np.min(x, axis=0, keepdims=True))

    print("Y max: ", np.max(y, axis=0, keepdims=True))
    print("X max: ", np.max(x, axis=0, keepdims=True))
    '''

    print("x dataset shape: ", x.shape)
    print("y dataset shape: ", y.shape)

    dataset = [x,y]
    
    return dataset

def get_model_pred(ctrl, model):
    
    with torch.no_grad():
        ctrl_torch = ptu.from_numpy(ctrl)
        ctrl_torch = ctrl_torch.view(1, -1)
        #pred = model.forward(ref_ctrl_torch)
        #print("Ref ctrl torch: ", ctrl_torch.shape)
        pred = model.output_pred(ctrl_torch)
        #print("Model prediction: ", pred)

    return pred


def plot_losses(model_trainer):

    train_loss_list = model_trainer.train_loss_list
    test_loss_list = model_trainer.test_loss_list
    tot_epochs = model_trainer.total_num_epochs
    
    plt.plot(np.arange(tot_epochs), train_loss_list, label="Train loss")
    plt.plot(np.arange(tot_epochs), test_loss_list, label="Test loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    return 1

def gt_vs_model(dataset, env):

    data_x, data_y = dataset

    for i in range(10):
        
        ctrl_idx = np.random.randint(data_x.shape[0])
        ref_ctrl = data_x[ctrl_idx]
        ref_answer = data_y[ctrl_idx]

        #ref_ctrl = np.array([1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0.5, 0.5, 0.25, 0.75, 0.5, 1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5, 1, 0.5, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5])

        # get model prediction for a reference controller to visualize
        pred = get_model_pred(ref_ctrl, model)

        
        fitness, desc, _, _ = env.evaluate_solution(ref_ctrl)
        #print(desc) 
        
        #print("Ground truth: ", fitness, desc)
        #print("Reference answer from dataset: ", ref_answer) 
        plt.scatter(x=pred[0][1] , y=pred[0][2] , c=pred[0][0], cmap='viridis', s=30.0, marker='x', label='Model prediction '+str(i))
        plt.scatter(x=desc[0][0] , y=desc[0][1] , c=fitness, cmap='viridis', s=30.0, marker='o', label='Ground truth '+str(i))

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

    return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()

    filename = "src/surrogate_model_analysis/pi4_3s_100hz_data/final_archive/archive_1000100.dat"
    #filename = "src/surrogate_model_analysis/pi4_3s_100hz_data/initial_archive/archive_100100.dat" 
    #filename = "src/surrogate_model_analysis/pi4_3s_100hz_data/initial_all_evals/archive_100100.dat" 
    
    # load dataset of genotype to BD
    dataset = load_dataset(filename)
    #dataset_final = load_dataset(filename_final)
    #dataset_all = load_dataset(filename_all)
    dim_x = 36 #genotype size

    # initialize model and model trainer
    model = DeterministicQDSurrogate(gen_dim=dim_x, bd_dim=2, hidden_size=64)
    model_trainer = SurrogateTrainer(model, batch_size=64)
    
    # train model on dataset
    model_trainer.train_from_dataset(dataset, holdout_pct=0.2, max_grad_steps=100000)
    #plot_losses(model_trainer)
    #plt.show()

    ## SAVE MODEL ##
    model_path = args.log_dir + "surrogate_nn_finalarch.pth"
    ptu.save_model(model, model_path)
    print("Saved model at: ", model_path)

    
    variant = dict(
        mbrl_kwargs=dict(
            ensemble_size=4,
            layer_size=256,
            learning_rate=1e-3,
            batch_size=256,
        )
    )
    obs_dim = 48
    action_dim = 18
    M = variant['mbrl_kwargs']['layer_size']
    dynamics_model = ProbabilisticEnsemble(
        ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M]
    )

    env = HexapodEnv(dynamics_model=dynamics_model, render=False, ctrl_freq=100)

    gt_vs_model(dataset, env)
