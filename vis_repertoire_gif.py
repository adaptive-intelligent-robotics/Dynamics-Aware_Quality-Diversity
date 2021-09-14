import sys, os
import argparse

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

import imageio


def get_plot(filename): 
    
    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma after last value i a line
    #data = np.loadtxt(args.filename)

    fit_bd = data.iloc[:,0:3]
    
    print("ARCHIVE SHAPE: ", data.shape)
    
    #For Hexapod
    data['x_bin']=pd.cut(x = data.iloc[:,1],
                        bins = [p/100 for p in range(101)], 
                        labels = [p for p in range(100)])
    data['y_bin']=pd.cut(x = data.iloc[:,2],
                        bins = [p/100 for p in range(101)],
                        labels = [p for p in range(100)])
    

    '''
    data['scaled_x'] = ((data.iloc[:,1]+1)/2)
    data['scaled_y'] = ((data.iloc[:,2]+1)/2)
    data['scaled_z'] = ((data.iloc[:,3]+0.5))
    
    data['x_bin']=pd.cut(x = data['scaled_x'],
                        bins = [p/100 for p in range(101)], 
                        labels = [p for p in range(100)])
    data['y_bin']=pd.cut(x = data['scaled_y'],
                        bins = [p/100 for p in range(101)],
v                        labels = [p for p in range(100)])
    data['z_bin']=pd.cut(x = data['scaled_z'],
                        bins = [p/100 for p in range(101)],
                        labels = [p for p in range(100)])
    '''
    
    #cmap = matplotlib.cm.get_cmap('Spectral') # Getting a list of color values
    #data['color_dict'] = pd.Series({k:cmap(1) for k in data['scaled_x']})
    
    #=====================PLOT DATA===========================#

    # FOR BINS / GRID
    if args.plot_type == "grid":
        fig, ax = plt.subplots()
        data.plot.scatter(x="x_bin",y="y_bin",c=0,colormap='viridis', s=2, ax=ax)
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
        data.plot.scatter(x=1,y=2,c=0,colormap='viridis', s=2, ax=ax)
        #data.plot.scatter(x=1,y=2,s=2, ax=ax[0])
        #data.plot.scatter(x=3,y=4,c=0,colormap='viridis', s=2, ax=ax)
        #data.plot.scatter(x=4,y=5,s=2, ax=ax[1])
        #plt.xlim(0,1)
        #plt.ylim(0,1)
        
    #plt.show() 
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str) # file to visualize rollouts from
    parser.add_argument("--plot_type", type=str, default="scatter", help="scatter plot or grid plot")

    args = parser.parse_args()


    filenames_data = []
    filenames_img = []
    for gen in range(1,47): 
        for i in range(0,61,10):
            print(i)
            filename_data = f'archive_{gen}_{i}.dat'
            filenames_data.append(filename_data)
            filename_img = f'archive_{gen}_{i}.png'
            filenames_img.append(filename_img)
            
            get_plot(filename_data)
            plt.savefig(filename_img)
            plt.close()

    # build gif
    print('Creating gif\n')
    with imageio.get_writer('opt_emit.gif', mode='I') as writer:
        for filename in filenames_img:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')


    print('Removing Images\n')
    # Remove files
    for filename in set(filenames_img):
        os.remove(filename)
    print('DONE')
