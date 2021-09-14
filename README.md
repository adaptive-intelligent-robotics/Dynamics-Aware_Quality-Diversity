# Dynamics-Aware Quality-Diversity (DA-QD)

Project repository for Dynamics-Aware Quality-Diversity (DA-QD). 
Performing Quality-Diversity (QD) optimisation using learnt models, to save efficiency on the number of QD evaluations.

## Some key points about input dimensions of the models

For Direct Neural Network Surrogate and Deterministic Dynamics Model:
- inputs should be of shape [batch_size, input_dim]

However, for probablistic ensemble which performs the ensemble in parallelized manner:
- taken from the code repository of https://github.com/kzl/lifelong_rl
- inputs should be of the shape [ensemble_size, batch_size, input_dim]
- However, if there is only one observation input i.e [batch_size, input_dim], there is a conditional if check (if len(input.shape) < 3) to replicate the input across all models in the ensemble and reshape it to [ensemble_size, batch_size, input_dim].


## DA-QD flow

1. Create copy of the repertoire. This copy becomes the _imagined repertoire_.
2. Select and mutate policies from the imagined repertoire.
3. Perform QD in imagination. Policies are evaluated in imagination using the model.
4. Select policies from the imagined repertoire - take only the most novel or the best policies found in imagination. 
5. Evaluate those policies in the environment.
6. Use the data collected from those policies to update the model.
7. Update the repertoire.

## Documentation on code structure

### Main algorithm class MBQD

The main class `ModelBasedQD` (present in `src/map_elites/mbqd`) takes as argument all the components:
  - Environment
  - Data collector (Rollout collector)
  - Policy/Actor
  - Dynamics_model, Dynamics Model Trainer
  - Replay buffer

Its `compute` method runs the main algorithm loop, consisting of:
1. Initialization of the repertoire with random policies
2. Copy this repertoire to produce an imagined repertoire.
3. Performing QD in imagination with this imagined repertoire.
4. Selection from the imagined repertoire.
5. Act in environment - evaluate selected solutions.
6. Update model/train model.


### Archive

- `src/map_elites/archive.py` contains the class `Archive` which is essentially just a container in the form of a python list containing individuals. Those individuals are instances of the `Species` class (implemented in `src/map_elites/common.py`).

### Selectors

- `src/map_elites/selectors.py` contains the functions for selection from the imagined repertoire i.e. UCB selector based on some metric
