from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import src.torch.pytorch_util as ptu
from src.trainers.trainer import TorchTrainer


'''
Surrogate Model for QD
Given datasets of BD and fitness corresponding to genotype
Learn to predict BD and fitness based on genotype
'''

class SurrogateTrainer(TorchTrainer):
    def __init__(
            self,
            model,
            learning_rate=1e-3,
            batch_size=32,
            optimizer_class=optim.Adam,
            train_call_freq=1,
            **kwargs
    ):
        super().__init__()

        torch.set_num_threads(16)
        self.model = model
        
        #self.obs_dim = ensemble.obs_dim
        #self.action_dim = ensemble.action_dim

        self.batch_size = batch_size
        self.train_call_freq = train_call_freq

        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

        self.train_loss_list = []
        self.test_loss_list = []
        self.total_num_epochs = 0 
        
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()


    def train_from_dataset(self, dataset, holdout_pct=0.2, max_grad_steps=100000, epochs_since_last_update=2):

        self._n_train_steps_total += 1

        #if self._n_train_steps_total % self.train_call_freq > 0 and self._n_train_steps_total > 1:
        #return
        
        x = dataset[0]  # inputs - genotype
        y = dataset[1]  # predict fitness and bd

        # shuffle data
        inds = np.random.permutation(x.shape[0])
        x, y = x[inds], y[inds]

        #normalize network inputs
        self.model.fit_input_stats(x)
        self.model.fit_output_stats(y)

        # generate holdout set (test and train set)
        n_train = int((1-holdout_pct) * x.shape[0])
        n_test = x.shape[0] - n_train

        x_train, y_train = x[:n_train], y[:n_train]
        x_test, y_test = x[n_train:], y[n_train:]
        x_test, y_test = ptu.from_numpy(x_test), ptu.from_numpy(y_test)

        # train until holdout set convergence
        num_epochs, num_steps = 0, 0
        num_epochs_since_last_update = 0
        best_holdout_loss = float('inf')
        num_batches = int(np.ceil(n_train / self.batch_size))

        while num_epochs_since_last_update < epochs_since_last_update and num_steps < max_grad_steps:

            # generate idx for each model to bootstrap
            self.model.train()

            #print("Steps: ", num_steps)
            #print("Num epochs since last update: ", num_epochs_since_last_update)
            epoch_loss_sum = 0
            
            for b in range(num_batches):
                # bootstrapping - sampling with replacement
                #b_idxs = np.random.randint(n_train, size=(self.batch_size))
                #x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]

                # standard batch
                if (b+1)*self.batch_size > n_train:
                    end = n_train
                else:
                    end = (b+1)*self.batch_size
                b_idxs = np.arange(b*self.batch_size,end)
                x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]

                x_batch, y_batch = ptu.from_numpy(x_batch), ptu.from_numpy(y_batch)

                x_batch = x_batch.view(b_idxs.shape[0], -1)
                y_batch = y_batch.view(b_idxs.shape[0], -1)
                loss = self.model.get_loss(x_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss_sum += loss
                
            num_steps += num_batches

            # stop training based on holdout loss improvement
            # i.e. if the holdout loss does not improve after 5 epochs
            # if holdout loss does better, num_epochs since last update is set back to 0
            # if holdout loss does not improve, counter counts up
            # after 5 counts, this exits the while loop
            self.model.eval()
            with torch.no_grad():
                holdout_losses = self.model.get_loss(x_test, y_test)
            holdout_loss = holdout_losses

            if num_epochs == 0 or \
               (best_holdout_loss - holdout_loss) / abs(best_holdout_loss) > 0.01:
                best_holdout_loss = holdout_loss
                num_epochs_since_last_update = 0
            else:
                num_epochs_since_last_update += 1

            #print("Epoch no: ",num_epochs,"     AverageTraining loss: ", epoch_loss_sum/num_batches) 
            #print("Epoch no: ",num_epochs,"     Holdout loss: ", holdout_loss) 

            self.train_loss_list.append(epoch_loss_sum/num_batches)
            self.test_loss_list.append(holdout_loss)
            
            num_epochs += 1

        self.total_num_epochs = num_epochs
        
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['Model Training Epochs'] = num_epochs
            self.eval_statistics['Model Training Steps'] = num_steps

                
    def train_from_torch(self, batch, idx=None):
        raise NotImplementedError

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.model
        ]

    def get_snapshot(self):
        return dict(
            model=self.model
        )
