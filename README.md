# model_based_qd

Project repository for model-based QD. Performing QD evaluations using learnt models to save efficiency on number of QD evaluations.


Some key points about input dimensions of the models:
For Direct NN Surrogate and Deterministic Dynamics Model (which was both made by me):
- inputs should be of shape [batch_size, input_dim]

However, for probablistic ensemble which performns the ensemble in parallelized manner
- taken from the code repository of https://github.com/kzl/lifelong_rl
- inputs should be of the shape [ensemble_size, batch_size, input_dim]
- However, if there is only one observation input i.e [batch_size, input_dim], there is a conditional if check (if len(input.shape) < 3) to replicate the input across all models in the ensemble and reshape it to [ensemble_size, batch_size, input_dim].







# Model-based QD flow
# create copy of the archive - archive copy
# select and mutate individuals from the  archive copy
# evaluate in models
# take only novel or better inds found using models (with some diff conditions) for eval in real
# evaluate in real
# Use real data - to update model,
