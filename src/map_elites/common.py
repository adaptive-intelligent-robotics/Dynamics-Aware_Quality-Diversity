#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.

import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.cluster import KMeans

default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": 10000,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters/genotype
        "min": 0,
        "max": 1,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,

        # params for polynomial mutation
        "eta_m": 0.5
    }

class Species:
    def __init__(self, x, desc, fitness, obs_traj=None, act_traj=None, model_dis=None, rwd_traj=None, desc_ground=None, centroid=None):
        self.x = np.array(x) # genotype
        self.desc = np.array(desc) # desciptor of the individual
        self.fitness = fitness #fitness of the individual

        self.model_dis = model_dis # model disagreeement
        self.desc_ground = np.array(desc_ground) # ground truth desciptor (if ther is)
        self.obs_traj = np.array(obs_traj) # trajectory (rollout of the genotype) of the individual
        self.act_traj = np.array(act_traj)
        self.rwd_traj = np.array(rwd_traj)
        
        self.centroid = centroid # only applies if using cvt map elites
        

        
def polynomial_mutation(x, params):
    '''
    Cf Deb 2001, p 124 ; 
    param: eta_m    
    Can choose to clip the mutation to ensure the genotype remains in range [0,1] 
    using np.clip(val, min, max) 
    '''
    y = x.copy()

    p_max = np.array(params["max"])
    p_min = np.array(params["min"])

    mutation_prob = params['mutation_prob']
    eta_m = params['eta_m'] # eta_m = 5.0;

    r = np.random.random(size=len(x))
    p = np.random.random(size=len(x))
    for i in range(0, len(x)):
        if (p[i] < mutation_prob):
            if r[i] < 0.5:
                delta_i = math.pow(2.0 * r[i], 1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta_i = 1 - math.pow(2.0 * (1.0 - r[i]), 1.0 / (eta_m + 1.0))
            y[i] += delta_i

    return np.clip(y, p_min, p_max)



def sbx(x, y, params):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.

    Requires two parent genotypes for mutation x and y
    '''
    eta = 10.0
    xl = params['min']
    xu = params['max']
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    return z

def iso_dd(x, y, params):
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation. GECCO 2018

    Requires two parent genotypes x and y
    '''
    assert(x.shape == y.shape)
    p_max = np.array(params["max"])
    p_min = np.array(params["min"])
    a = np.random.normal(0, params['iso_sigma'], size=len(x))
    b = np.random.normal(0, params['line_sigma'])
    norm = np.linalg.norm(x - y)
    z = x.copy() + a + b * (x - y)

    return np.clip(z, p_min, p_max)
    #return z


# selected mutation operator (from the functions above) on the genotypes
def variation(x, z, params):
    assert(x.shape == z.shape)
    if (params["mutation"] == "polynomial"):
        y = polynomial_mutation(x, params)
    elif (params["mutation"] == "sbx"):
        y = sbx(x, z, params)
    elif (params["mutation"] == "iso_dd"):
        y = iso_dd(x, z, params)

    return y

def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'

def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')

def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")
    k_means.fit(x)
    __write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_

def grid_centroids(n_bins):
    centers = []
    for x in n_bins:
        diff = 0.5/x
        centers.append([ ((1/x)*i) - diff for i in range(1, x + 1) ])
    output = [ [x] for x in centers[0]]
    for x in centers[1:]:
        new_list = []
        for y in output:
            for i in x:
                curr = y.copy()
                curr.append(i)
                new_list.append(curr)
            output=new_list.copy()
    return np.array(output)


# hashable makes it usable as a dictionary key
def make_hashable(array):
    return tuple(map(float, array))


'''
- pool.map(f,x) applies the list of x to the function f and returns the outputs of the function as a list as well which corresponds in order/index to the inputs
- evaluate_function returns fitness and the descriptor as tuple
- to evaluate is a list of genotypes/ind to evaluate
'''
def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params['parallel'] == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)




# format: fitness, desc, genome, centroid(optional) \n- 
# fitness,  desc and x are vectors
def save_archive(archive, gen, params, log_dir):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ',') # save comma inbetween so easier to read as pandas

    def get_array_string(a):
        array_str = ''
        for i in a:
            array_str += str(i) + ','
        return array_str
        
    filename = log_dir + '/' + 'archive_' + str(gen) + '.dat'

    with open(filename, 'w') as f:
        if (params['type'] == "cvt") or (params["type"] == "grid"): 
            for k in archive.values():
                f.write(str(k.fitness) + ',') # write fitness
                write_array(k.desc, f) # write desriptor
                write_array(k.x, f) # write genotype
                #write_array(k.centroid, f)
                f.write("\n") # newline to store new individual
        elif (params["type"] == "unstructured"):
            for k in archive:

                ind_string = str(k.fitness) + ','
                ind_string += get_array_string(k.desc)
                ind_string += str(k.model_dis) + ','
                ind_string += get_array_string(k.x)

                f.write(ind_string + "\n")
                
                '''
                f.write(str(k.fitness) + ',') # write fitness
                write_array(k.desc, f) # write desriptor
                f.write(str(k.model_dis) + ',') # model_dis
                #write_array(k.desc_ground, f) # write ground truth desriptor
                write_array(k.x, f) # write genotype
                #write_array(k.obs_traj, f) # write obervation trajectory
                #write_array(k.act_traj, f) # write action trajectory
                f.write("\n") # newline to store new individual
                '''
