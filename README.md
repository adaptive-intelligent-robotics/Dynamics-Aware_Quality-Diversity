# Dynamics-Aware Quality-Diversity (DA-QD)
Project repository for Dynamics-Aware Quality-Diversity (DA-QD).

[Project Webpage](https://sites.google.com/view/da-qd/home)

[Paper](https://arxiv.org/abs/2109.08522)


Performing Quality-Diversity (QD) optimisation using learnt models, to reduce the number of evaluations/trials/samples needed during QD skill discovery.
DA-QD allows QD to be performed in imagination, to discover and only executte novel or higher-performing skills.

## Dependencies and Installation
We recommend using singularity containers to easily run this code. The singularity directory contains all the required files to build a container (i.e. singularity.def).
The code and experiments in this repository uses the [DART](https://dartsim.github.io) physics simulator and [RobotDART](https://github.com/resibots/robot_dart) wrapper.
If you want to build this without a singularity container, all the dependencies can be found in the singularity.def file. 

## How to run the code?
1. Create a directory or identify directory in which to store results (i.e. $tmp_dir) 

2. Run the line below for skill-discovery using DAQD on the hexapod robot:
```
python3 run_scripts/hexapod_omni_daqd_main.py --num_cores 30 --log_dir $tmp_dir --dump_period 5000
```

3. For analysis and visualization of results, use:
```
python3 vis_repertoire_hexapod.py --archive_519.dat --plot_type grid
```
This plots the resulting repertoire. This is also an interactive plot which shows the resulting skill/behaviour in a rendered simulation when you select a skill in the repertoire on the plot using the mouse.

and
```
python3 analysis.py
```
This plots the performance curves for QD-score and coverage metrics. You will need to go into this file to specify the paths of the directories of the log_file.dat from your different variants of your experiments.


## Documentation on code structure
### Main algorithm class MBQD
The main class `ModelBasedQD` (present in `src/map_elites/mbqd`) takes as argument all the components:
  - Environment
  - Direct Model, Direct Model Trainer
  - Dynamics Model, Dynamics Model Trainer
  - Replay Buffer

Its `compute` method runs the main algorithm loop, consisting of:
1. Initialization of the repertoire with random policies
2. Copy this repertoire to produce an imagined repertoire.
3. Performing QD in imagination with this imagined repertoire.
4. Selection from the imagined repertoire.
5. Act in environment - evaluate selected solutions.
6. Update the skill repertoire.
7. Update model/train model.
8. Repeat step 2-7.

## Some points about input dimensions of the models
For Direct Neural Network Surrogate and Deterministic Dynamics Model:
- inputs should be of shape [batch_size, input_dim]

However, for probablistic ensemble which performs the ensemble in parallelized manner:
- taken from the code repository of https://github.com/kzl/lifelong_rl
- inputs should be of the shape [ensemble_size, batch_size, input_dim]
- However, if there is only one observation input i.e [batch_size, input_dim], there is a conditional if check (if len(input.shape) < 3) to replicate the input across all models in the ensemble and reshape it to [ensemble_size, batch_size, input_dim].

## Acknowledgements
- The QD code is built and modified from the <https://github.com/resibots/pymap_elites> repository.
- Code for dynamics models using probabalistic ensembles is built and modified from the <https://github.com/kzl/lifelong_rl> repository.

