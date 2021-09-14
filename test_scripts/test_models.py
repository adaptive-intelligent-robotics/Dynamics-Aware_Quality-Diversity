import numpy as np
import torch

from src.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from src.trainers.mbrl.mbrl import MBRLTrainer
from src.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
import src.torch.pytorch_util as ptu


variant = dict(
    mbrl_kwargs=dict(
        ensemble_size=4,
        layer_size=256,
        learning_rate=1e-3,
        batch_size=256,
    )
)

obs_dim = 10
action_dim = 4
M = variant['mbrl_kwargs']['layer_size']

# initialize dynamics model
dynamics_model = ProbabilisticEnsemble(
    ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_sizes=[M, M]
)

# initialize model trainer class
model_trainer = MBRLTrainer(
    ensemble=dynamics_model,
    **variant['mbrl_kwargs'],
)

# initialize replay buffer
replay_buffer = SimpleReplayBuffer(
    max_replay_buffer_size=10000,
    observation_dim=obs_dim,
    action_dim=action_dim,
    env_info_sizes=dict(),
)

import gym
import panda_gym

def runsim(render=False, print_pred=False):

    env = gym.make('PandaReach-v0', render=render)

    obs = env.reset()
    done = False

    total_mean_error = 0
    
    for i in range(100):
        action = env.action_space.sample() # random action
        next_obs, reward, done, info = env.step(action)

        s = obs["observation"]
        ns = next_obs["observation"]

        if print_pred:
            s = ptu.from_numpy(s)
            a = ptu.from_numpy(action)
            ns = ptu.from_numpy(ns)

            s = s.view(1,-1)
            a = a.view(1,-1)

            preds = dynamics_model.forward(torch.cat((s, a), dim=-1))
            pred_ns = preds[0]+s[0]

            model_error = ptu.get_numpy(ns - pred_ns)
            mean_error = np.linalg.norm(model_error)
            total_mean_error += mean_error
            #print("Model error: ", model_error) 
            #print("Model prediction of next state: ", preds[0]+s[0])
            #print("ground truth of next state: ", ns)
            
        #env_transition = (obs, action, reward, done, next_obs, info)
        replay_buffer.add_sample(s, action, reward, done, ns, info)            

    print("Total mean error: ", total_mean_error)
            
    env.close()

    return 1


def main():

    for epoch in range(5):
        for i in range(100):
            runsim()
            print("Replay buffer size",replay_buffer._size)
            
        print("Train model")
        model_trainer.train_from_buffer(replay_buffer)

        runsim(print_pred=True)

main()
