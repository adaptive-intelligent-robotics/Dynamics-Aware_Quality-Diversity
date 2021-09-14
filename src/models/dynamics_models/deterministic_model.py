import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import src.torch.pytorch_util as ptu


class DeterministicDynModel(nn.Module):
    """
    Predicts residual of next state given current state and action
    """
    def __init__(
            self,
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_size,         # Hidden size for model
    ):
        super(DeterministicDynModel, self).__init__()

        torch.set_num_threads(16)
        
        self.obs_dim, self.action_dim = obs_dim, action_dim

        self.input_dim = self.obs_dim + self.action_dim
        self.output_dim = self.obs_dim # fitness always a scalar
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_dim)

        self.MSEcriterion = nn.MSELoss()
        self.L1criterion = nn.L1Loss()

        # the trainer computes the mu and std of the train dataset
        self.input_mu = nn.Parameter(ptu.zeros(1,self.input_dim), requires_grad=False).float()
        self.input_std = nn.Parameter(ptu.ones(1,self.input_dim), requires_grad=False).float()

        
        self.output_mu = nn.Parameter(ptu.zeros(1,self.output_dim), requires_grad=False).float()
        self.output_std = nn.Parameter(ptu.ones(1,self.output_dim), requires_grad=False).float()

        # xavier intialization of weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        
    def forward(self, x_input):

        # normalize the inputs
        h = self.normalize_inputs(x_input)

        #h = x_input
        x = torch.relu(self.fc1(h))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    
    def get_loss(self, x, y, return_l2_error=False):

        # predicted normalized outputs given inputs
        pred_y = self.forward(x)

        # normalize the output/label as well
        norm_y = self.normalize_outputs(y)

        # calculate loss - we want to predict the normalized residual of next state
        loss = self.MSEcriterion(pred_y, norm_y)
        #loss = self.L1criterion(pred_y, y)

        return loss

    # get input data mean and std for normalization
    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0

        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = ptu.from_numpy(mean)
        self.input_std.data = ptu.from_numpy(std)

    # get output data mean and std for normalization
    def fit_output_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        
        if mask is not None:
            mean *= mask
            std *= mask

        self.output_mu.data = ptu.from_numpy(mean)
        self.output_std.data = ptu.from_numpy(std)
    

    #output predictions after unnormalized
    def output_pred(self, x_input):
        
        # batch_preds is the normalized output from the network
        batch_preds = self.forward(x_input)
        y = self.denormalize_output(batch_preds)
        output = ptu.get_numpy(y)

        return output

    
    def normalize_inputs(self, data):
        data_norm = (data - self.input_mu)/(self.input_std)
        return data_norm

    def normalize_outputs(self, data):
        data_norm = (data - self.output_mu)/(self.output_std)
        return data_norm

    def denormalize_output(self, data):
        data_denorm = data*self.output_std + self.output_mu
        return data_denorm
    
    
    
