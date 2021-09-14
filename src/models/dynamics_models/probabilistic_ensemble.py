import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import src.torch.pytorch_util as ptu
from src.models.networks import ParallelizedEnsemble

def identity(x):
    return x


class ProbabilisticEnsemble(ParallelizedEnsemble):

    """
    Probabilistic ensemble (Chua et al. 2018).
    Implementation is parallelized such that every model uses one forward call.
    Each member predicts the mean and variance of the next state.
    Sampling is done either uniformly or via trajectory sampling.
    """

    def __init__(
            self,
            ensemble_size,        # Number of members in ensemble
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_sizes,         # Hidden sizes for each model
            spectral_norm=False,  # Apply spectral norm to every hidden layer
            **kwargs
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=2*(obs_dim),  # We predict (next_state - state)
            hidden_activation=torch.relu,
            output_activation=identity,
            spectral_norm=spectral_norm,
            **kwargs
        )

        torch.set_num_threads(24)

        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.output_size = obs_dim

        # only needed if using trajectory shooting during predictions
        self.num_particles = 1 # number of particles per model
        
        # Note: we do not learn the logstd here, but some implementations do
        # modified max and min variance - see above original
        self.max_logstd = nn.Parameter(
            ptu.ones(obs_dim), requires_grad=False)
        self.min_logstd = nn.Parameter(
            -ptu.ones(obs_dim), requires_grad=False)

         
    def forward(self, input, deterministic=False, return_dist=False):
        output = super().forward(input)
        mean, logstd = torch.chunk(output, 2, dim=-1)

        # Variance clamping to prevent poor numerical predictions
        # see Appendix A.1 of Chua et al. (2018) for explanation
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)

        if deterministic:
            if return_dist:
                return mean, logstd
            else:
                return mean
            #return mean, logstd if return_dist else mean

        # randn returns tensor filled with rand numbs from a norm dist with mean 0 and var 1
        std = torch.exp(logstd)
        eps = ptu.randn(std.shape)
        samples = mean + std * eps

        if return_dist:
            return samples, mean, logstd
        else:
            return samples

    def get_loss(self, x, y, split_by_model=False, return_l2_error=False):
        # Note: we assume y here already accounts for the delta of the next state

        # normalize the output/label as well
        y = self.normalize_outputs(y)
        
        mean, logstd = self.forward(x, deterministic=True, return_dist=True)
        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        #print("Mean shape", mean.shape)
        #print("y shape", y.shape)

        # Maximize log-probability of transitions (negative log likelihood)
        inv_var = torch.exp(-2 * logstd) # 1/var 
        sq_l2_error = (mean - y)**2 # error squared
        if return_l2_error:
            l2_error = torch.sqrt(sq_l2_error).mean(dim=-1).mean(dim=-1)

        # nll loss    
        loss = (sq_l2_error*inv_var + 2*logstd).sum(dim=-1).mean(dim=-1) 

        # mseloss - works for mean only (i.e. if determinisitc)
        #loss = torch.sqrt(sq_l2_error).mean(dim=-1).mean(dim=-1) 

        # debug
        #print("sq_l2_error: ", sq_l2_error)
        #print("inv_var:  ",inv_var)
        #print("log_std: ", logstd)
        
        if split_by_model:
            losses = [loss[i] for i in range(self.ensemble_size)]
            if return_l2_error:
                l2_errors = [l2_error[i] for i in range(self.ensemble_size)]
                return losses, l2_errors
            else:
                return losses
        else:
            if return_l2_error:
                return loss.mean(), l2_error.mean()
            else:
                return loss.mean()

    def output_pred(self, x_input, mean=False):
        '''
        outputs prediction of only one model in the ensemble
        - acts as just a single probabilstic nueral network
        '''
        with torch.no_grad():
            if mean:     
                y = self.forward(x_input, deterministic=True, return_dist=False)
                #print("y shape: ",y.shape)
            else: 
                y = self.forward(x_input)
                
        # testing with just one of the models in ensemble
        y = self.denormalize_output(y[0])

        output = ptu.get_numpy(y)
        return output

    def output_pred_ts_ensemble(self, obs, actions, mean=False):
        # parallelized ensemble handles TS cleanly
        with torch.no_grad():
            preds = self.forward(torch.cat(
                (self._expand_to_ts_form(obs), self._expand_to_ts_form(actions)), dim=-1),
                                 deterministic=mean) #[4,1,48]
            preds = self._flatten_from_ts(preds) # [4,48]
            
        for i in range(self.ensemble_size):
            preds[i] = self.denormalize_output(preds[i])
            
        preds = ptu.get_numpy(preds)
        
        #print("Preds shape: ", preds.shape) # [4,48] = [ensemble_size, obs_dim]
        return preds
    
    def sample_with_disagreement(self, input, return_dist=False, disagreement_type='mean'):
        #print("input shape: ", input.shape[:-2])
        preds, mean, logstd = self.forward(input, deterministic=False, return_dist=True)
        #print("preds shape: ", preds.shape)
        #print("mean shape: ", mean.shape)
        preds = mean # to get a deterministic restuls for now hard coded 
        
        # Standard uniformly from the ensemble - randomly sample model from ensembles
        inds = torch.randint(0, preds.shape[0], input.shape[:1])

        # Ensure we don't use the same member to estimate disagreement
        # if same int is sampled, mod it to ensure it wont be the same
        inds_b = torch.randint(0, mean.shape[0], input.shape[:1])
        inds_b[inds == inds_b] = torch.fmod(inds_b[inds == inds_b] + 1, mean.shape[0])

        # Repeat for multiplication
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, preds.shape[2])
        inds_b = inds_b.unsqueeze(dim=-1).to(device=ptu.device)
        inds_b = inds_b.repeat(1, preds.shape[2])

        
        # Uniformly sample from ensemble
        samples = (inds == 0).float() * preds[0]
        #print("inds == 0: ",(inds == 0).shape)
        #print("preds 0: ",preds[0].shape)
        #print((inds==0).float())
        #print(samples)
        for i in range(1, preds.shape[0]):
            samples += (inds == i).float() * preds[i]

        # CODE TO GET ACTUAL PREDCTIONS - STILL UNSURE WHAT SAMPLES DO ABOVE
        samples = self._flatten_from_ts(preds)
        # denormalize
        for i in range(self.ensemble_size):
            samples[i] = self.denormalize_output(samples[i])

        # just for plotting purposes
        #for i in range(self.ensemble_size):
        #    mean[i] = self.denormalize_output(mean[i])
        
        if disagreement_type == 'mean':
            # Disagreement = mean squared difference in mean predictions (Kidambi et al. 2020)
            means_a = (inds == 0).float() * mean[0]
            means_b = (inds_b == 0).float() * mean[0]
            #print("Means a: ",means_a)
            #print("Means b: ",means_b)
            for i in range(1, preds.shape[0]):
                means_a += (inds == i).float() * mean[i]
                means_b += (inds_b == i).float() * mean[i]

            disagreements = torch.mean((means_a - means_b) ** 2, dim=-1, keepdim=True)
            
        elif disagreement_type == 'var':
            # Disagreement = max Frobenius norm of covariance matrix (Yu et al. 2020)
            vars = (2 * logstd).exp()
            frobenius = torch.sqrt(vars.sum(dim=-1))
            disagreements, *_ = frobenius.max(dim=0)
            disagreements = disagreements.reshape(-1, 1)

        else:
            raise NotImplementedError

        if return_dist:
            return samples, disagreements, mean, logstd
        else:
            return samples, disagreements

        
    def normalize_inputs(self, data):
        data_norm = (data - self.input_mu)/(self.input_std)
        return data_norm

    def normalize_outputs(self, data):
        data_norm = (data - self.output_mu)/(self.output_std)
        return data_norm

    def denormalize_output(self, data):
        data_denorm = data*self.output_std + self.output_mu
        return data_denorm

    # extra functions for trajectory shooting of each model in ensemble
    # only if you consider particles
    def _expand_to_ts_form(self, x):
        d = x.shape[-1]
        reshaped = x.view(-1, self.ensemble_size, self.num_particles, d)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(self.ensemble_size, -1, d)
        #print("reshaped shape: ",reshaped.shape)
        return reshaped

    def _flatten_from_ts(self, y):
        d = y.shape[-1]
        reshaped = y.view(self.ensemble_size, -1, self.num_particles, d)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, d)
        return reshaped
