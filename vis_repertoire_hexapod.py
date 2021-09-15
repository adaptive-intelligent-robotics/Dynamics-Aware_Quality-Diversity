import sys, os
import argparse

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import src.torch.pytorch_util as ptu

#from src.envs.hexapod_dart.hexapod_env_v2 import simulate
from src.envs.hexapod_dart.hexapod_env import HexapodEnv
from src.models.dynamics_models.deterministic_model import DeterministicDynModel

dynamics_model = DeterministicDynModel(48,18,500)
#dynamics_model = ptu.load_model(dynamics_model, "hexapod_detdyn_archcond_trained.pth")
env = HexapodEnv(dynamics_model=dynamics_model, render=True, ctrl_freq=100)

def key_event(event, args):
    if event.key == 'escape':
        sys.exit(0)

def click_event(event, args):
    '''
    # reutrns a list of tupples of x-y points
    click_in = plt.ginput(1,-1) # one click only, should not timeout
    print("click_in: ",click_in)
    
    selected_cell = [int(click_in[0][0]), int(click_in[0][1])]
    print(selected_cell)
    selected_x = selected_cell[0]
    selected_y = selected_cell[1]
    '''
    #event.button ==1 is the left mouse click
    if event.button == 1:
        selected_x = int(event.xdata)
        selected_y = int(event.ydata)
        selected_solution = data[(data["x_bin"] == selected_x) & (data["y_bin"] == selected_y)]
        #selected_solution = data[(data["y_bin"] == selected_x) & (data["z_bin"] == selected_y)]

        # For hexapod omnitask
        print("SELECTED SOLUTION SHAPE: ",selected_solution.shape)
        selected_solution = selected_solution.iloc[0, :]
        print("Selected solution shape: ", selected_solution.shape)
        selected_ctrl = selected_solution.iloc[4:-2].to_numpy()
        print(selected_ctrl.shape) #(1,36)

        #hexapod uni
        #selected_solution = selected_solution.iloc[0, :]
        #selected_ctrl = selected_solution.iloc[8:-2].to_numpy()
        
        #print("Selected ctrl shape: ", selected_ctrl.shape) # should be 3661
        print("Selected descriptor bin: " ,selected_x, selected_y)
        print("Selected descriptor from archive: ", selected_solution.iloc[1:3].to_numpy())
        #print("Selected fitness from archive: ", selected_solution.iloc[0])

        # ---- SIMULATE THE SELECTED CONTROLLER -----#
        #simulate(selected_ctrl, 5.0, render=True) # Hexapod
        #env.evaluate_solution(selected_ctrl)
        
        #fit, desc, _, _ = env.evaluate_solution_uni(selected_ctrl, render=True)
        #print("fitness from simulation real eval:", fit)
        #fit, desc, _, _ = env.evaluate_solution_model_uni(selected_ctrl)
        #print("fitness from dynamics model :", fit)
        
        fit, desc, _, _ = env.evaluate_solution(selected_ctrl, render=True)
        #simulate(selected_ctrl, render=True) # panda bullet
        #simulate(selected_ctrl, 5.0, render=True) # panda dart 
        #evaluate_solution(selected_ctrl, gui=True) 
        print("SIMULATION DONE")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str) # file to visualize rollouts from
    parser.add_argument("--sim_time", type=float, help="simulation time depending on the type of archive you chose to visualize, 3s archive or a 5s archive")
    parser.add_argument("--plot_type", type=str, default="scatter", help="scatter plot or grid plot")

    args = parser.parse_args()
    
    data = pd.read_csv(args.filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
    #data = np.loadtxt(args.filename)

    #fit_bd = data.iloc[:,0:3]

    #print("ARCHIVE SHAPE ORIGINAL: ", data.shape)
    # Get names of indexes for which column Age has value 30
    #indexNames = data[data.iloc[:,0] <= -0.1].index
    # Delete these row indexes from dataFrame
    #data.drop(indexNames, inplace=True)
    
    #print("ARCHIVE SHAPE AFTER DROPPING: ", data.shape)
    #print(data)
    #print(fit_bd)
    
    #For Hexapod
    data['x_bin']=pd.cut(x = data.iloc[:,1],
                        bins = [p/100 for p in range(101)], 
                        labels = [p for p in range(100)])
    data['y_bin']=pd.cut(x = data.iloc[:,2],
                        bins = [p/100 for p in range(101)],
                        labels = [p for p in range(100)])
    
    '''
    # hexapod uni
    data['bd1_bin']=pd.cut(x = data.iloc[:,1],
                        bins = [p/100 for p in range(101)], 
                        labels = [p for p in range(100)])
    data['bd2_bin']=pd.cut(x = data.iloc[:,2],
                        bins = [p/100 for p in range(101)],
                        labels = [p for p in range(100)])
`    data['bd3_bin']=pd.cut(x = data.iloc[:,3],
                        bins = [p/100 for p in range(101)], 
                        labels = [p for p in range(100)])
    data['bd4_bin']=pd.cut(x = data.iloc[:,4],
                        bins = [p/100 for p in range(101)],
                        labels = [p for p in range(100)])
    data['bd5_bin']=pd.cut(x = data.iloc[:,5],
                        bins = [p/100 for p in range(101)], 
                        labels = [p for p in range(100)])
    data['bd6_bin']=pd.cut(x = data.iloc[:,6],
                        bins = [p/100 for p in range(101)],
                        labels = [p for p in range(100)])

    '''

    #cmap = matplotlib.cm.get_cmap('Spectral') # Getting a list of color values
    #data['color_dict'] = pd.Series({k:cmap(1) for k in data['scaled_x']})
    
    #=====================PLOT DATA===========================#

    # FOR BINS / GRID
    if args.plot_type == "grid":
        fig, ax = plt.subplots()
        data.plot.scatter(x="x_bin",y="y_bin",c=0,colormap="viridis", s=2, ax=ax)
        plt.xlim(0,100)
        plt.ylim(0,100)
    elif args.plot_type == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = data.iloc[:,1]
        ys = data.iloc[:,2]
        zs = data.iloc[:,3]

        ax.scatter(xs, ys, zs, marker="x")
        
    else:
        #fig, ax = plt.subplots(nrows=1, ncols=2)
        fig, ax = plt.subplots()

        # FOR JUST A SCATTER PLOT OF THE DESCRIPTORS - doesnt work for interactive selection
        #data.plot.scatter(x=2,y=3,c=0,colormap='Spectral', s=2, ax=ax, vmin=-0.1, vmax=1.2)
        data.plot.scatter(x=1,y=2,c=0,colormap='viridis', s=2, ax=ax)

        #data.plot.scatter(x=1,y=2,s=2, ax=ax[0])
        #data.plot.scatter(x=3,y=4,c=0,colormap='viridis', s=2, ax=ax)
        #data.plot.scatter(x=4,y=5,s=2, ax=ax[1])
        #plt.xlim(-0.5,0.5)
        #plt.ylim(-0.5,0.5)

        
    # event to look out. visualization or closing the plot
    fig.canvas.mpl_connect('key_press_event', lambda event: key_event(event, args))
    fig.canvas.mpl_connect('button_press_event', lambda event: click_event(event, args))
    
    plt.show() 

    
