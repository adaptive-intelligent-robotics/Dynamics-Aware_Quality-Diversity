import os, sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import imageio


class AIRL_ColorMap(object):
    @classmethod
    def get_2D_colormap(cls, data: np.ndarray, angle_offset=0, rank_based_coloring=False):

        if rank_based_coloring:
            arg_sort = np.argsort(data, axis=0)
            res = np.zeros_like(data)
            for i in range(np.size(data, 0)):
                for j in range(np.size(data, 1)):
                    res[arg_sort[i, j], j] = i
            data = res

        data = data.reshape((-1, 2))
        # N = np.size(data, axis=0)
        # minXY = np.min(data, axis=0)
        # maxXY = np.max(data, axis=0)
        # center = (maxXY + minXY) / 2
        center = np.median(data, axis=0)

        # center data
        data = data - center

        # "normalise data"
        data = data / np.max(np.abs(data), axis=0)

        angle = np.arctan2(data[:, 1], data[:, 0]) + angle_offset
        angle = angle.reshape((-1, 1))
        print(np.min(angle))

        l = np.cbrt(np.sum(data ** 2, axis=1))
        l = l / np.max(l)

        l = l.reshape((-1, 1))

        C = cls.get_color_function(angle)

        C = C * 0.7 + 0.3
        C = l * C + (1. - l) * 0.75
        return C

    @classmethod
    def get_1D_colormap(cls, data: np.ndarray, angle_offset=0, rank_based_coloring=False):

        data = data.reshape((-1, 1))
        data = data + np.min(data)
        data = 2 * np.pi * data / np.max(data)

        C = cls.get_color_function(data)

        C = C * 0.7 + 0.3

        return C

    @classmethod
    def get_color_function(cls, t: np.ndarray) -> np.ndarray:
        t = t.flatten()
        r = (np.sin(t + 1.3 * np.pi) / 2. + 0.5) * 0.8 + 0.1
        g = (np.sin(t) / 2.) * 0.6 + 0.6
        b = 1. - ((np.sin(t + 1.6 * np.pi) / 2.) + 0.5)

        return np.vstack((r, g, b)).T


def convert_to_rgb(dataset: np.ndarray, indexes_color_component, center=None, numpy_max=None, rank_based_coloring=True, max_l=None):
    assert len(indexes_color_component) in (2, 3)

    if len(indexes_color_component) == 2:
        rgb_color_component = AIRL_ColorMap.get_2D_colormap(dataset[:, np.array(list(indexes_color_component))], rank_based_coloring=rank_based_coloring)
    elif len(indexes_color_component) == 3:
        sub_component = dataset[:, np.array(list(indexes_color_component))]

        rgb_max = np.max(sub_component, axis=0)
        rgb_min = np.min(sub_component, axis=0)
        rgb_color_component = np.asarray(255 * (sub_component - rgb_min) / (rgb_max - rgb_min), dtype=np.int)
    else:
        raise ValueError

    list_str_colors = [
        f'rgb({rgb_color_component[i, 0]}, '
        f'{rgb_color_component[i, 1]}, '
        f'{rgb_color_component[i, 2]})' for i
        in range(rgb_color_component.shape[0])]

    return rgb_color_component, list_str_colors


def load_archive(filename):
    # load in archive.dat file
    data = pd.read_csv(filename)
    data = data.iloc[:,:-1] # drop the last column which was made because there is a comma

    print("Archive file data shape: ", data.shape)
    # 41 columns fitness, bd1, bd2, bd_ground1, bdground2, genotype(36dim)
    genotype = data.iloc[:,-36:]
    fit = data.iloc[:,0]
    desc = data.iloc[:,1:3]

    #genotype = np.expand_dims(genotype.to_numpy(),axis=0)
    #fit = np.expand_dims(fit.to_numpy(), axis=0)
    #desc = np.expand_dims(desc.to_numpy(),axis=0)

    genotype = genotype.to_numpy()
    fit = fit.to_numpy()
    desc = desc.to_numpy()

    print("gen fit desc  data shape: ", genotype.shape, fit.shape, desc.shape)
    
    return genotype, fit, desc


def load_gp_log(filename):

    data = np.load(filename)
    model_desc_log = data["model_desc_log"]
    return model_desc_log




def main():

    # load original archive
    archive_filename = "archive_example.dat"
    genotype, fit, desc = load_archive(archive_filename)
    print("Desc shape: ", desc.shape)
    
    # load gp model log
    model_log_filename = "model_repertoire_log_no_damage.npz"
    model_desc_log = load_gp_log(model_log_filename)
    print("GP model log shape: ", model_desc_log.shape)

    num_points = desc.shape[0]
    print("Num points:", num_points)
    colors = cm.viridis(np.linspace(0, 1, num_points))


    
    filenames_img = []
    for i in range(model_desc_log.shape[0]):
        x = model_desc_log[i,:,0]
        y = model_desc_log[i,:,1]
        plt.scatter(x, y, color=colors)
        plt.xlim(-0.8,0.8)
        plt.ylim(-0.8,0.8)
        plt.title(f"Step {i}")

        filename_img = f'model_archive_{i}.png'
        filenames_img.append(filename_img)
        
        plt.savefig(filename_img)
        plt.close()
        #plt.show()


    # build gif
    print('Creating gif\n')
    with imageio.get_writer('gp_model_evolution_no_damage_2.gif', mode='I') as writer:
        for filename in filenames_img:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')

    print('Removing Images\n')
    # Remove files
    for filename in set(filenames_img):
        os.remove(filename)
    print('DONE')


main()

    
