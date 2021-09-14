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
import os, sys
import time
import math
import numpy as np
import multiprocessing
from multiprocessing import get_context


# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from src.map_elites import common as cm
from src.map_elites import unstructured_container, cvt
from src.map_elites import model_condition_utils

import torch
import src.torch.pytorch_util as ptu

import cma

def evaluate_(t):
    # evaluate a single vector (x) with a function f and return a species
    # evaluate z with function f - z is the genotype and f is the evalution function
    # t is the tuple from the to_evaluate list
    z, f = t
    fit, desc, obs_traj, act_traj = f(z) 
    
    # becasue it somehow returns a list in a list (have to keep checking sometimes)
    desc = desc[0] # important - if not it fails the KDtree for cvt and grid map elites
    desc_ground = desc
    
    # return a species object (containing genotype, descriptor and fitness)
    return cm.Species(z, desc, fit, obs_traj=None, act_traj=None)


class QD:
    def __init__(self,
                 dim_map, dim_x,
                 f_real,
                 n_niches=1000,
                 params=cm.default_params,
                 bins=None,
                 log_dir='./',):

        #torch.set_num_threads(24)
        
        self.qd_type = params["type"]    # QD type - grid, cvt, unstructured
        self.dim_map = dim_map           # number of BD dimensions  
        self.dim_x = dim_x               # gemotype size (number of genotype dim)
        self.n_niches = n_niches         # estimated total population in archive
        self.bins = bins                 # grid shape - only for grid map elites
        self.params = params
        
        # eval functions
        self.f_real = f_real
        
        # Init logging directory and log file
        self.log_dir = log_dir
        log_filename = self.log_dir + '/log_file.dat'
        self.log_file = open(log_filename, 'w')
        
        if params['log_time_stats']:
            time_stats_filename = self.log_dir + '/time_log_file.dat'
            self.time_log_file = open(time_stats_filename, 'w')
            self.gen_time = 0
            self.model_eval_time = 0
            self.eval_time = 0
            self.model_train_time = 0 
        
        # Init cvt and grid - only if cvt and grid map elites used
        if (self.qd_type=="cvt") or (self.qd_type=="grid"):
            c = []
            if (self.qd_type=="cvt"):
                c = cm.cvt(self.n_niches,
                           self.dim_map,params['cvt_samples'], \
                           params['cvt_use_cache'])
            else:
                c = cm.grid_centroids(self.bins)

            self.kdt = KDTree(c, leaf_size=30, metric='euclidean')
            cm.__write_centroids(c)

            
        if (self.qd_type == "cvt") or (self.qd_type=="grid"):
            self.archive = {} # init archive as dic (empty)
            self.model_archive = {}
        elif self.qd_type == "unstructured":
            self.archive = [] # init archive as list
            self.model_archive = []        


    def random_archive_init(self, to_evaluate):
        for i in range(0, self.params['random_init_batch']):
            x = np.random.uniform(low=self.params['min'], high=self.params['max'], size=self.dim_x)
            to_evaluate += [(x, self.f_real)]
        
        return to_evaluate


    def select_and_mutate(self, to_evaluate, archive, f, params, variation_operator=cm.variation, batch=False):

        if (self.qd_type=="cvt") or (self.qd_type=="grid"):
            keys = list(archive.keys())
        elif (self.qd_type=="unstructured"):
            keys = archive
                    
        # we select all the parents at the same time because randint is slow
        rand1 = np.random.randint(len(keys), size=self.params['batch_size'])
        rand2 = np.random.randint(len(keys), size=self.params['batch_size'])
            
        for n in range(0, params['batch_size']):
            # parent selection - mutation operators like iso_dd/sbx require 2 gen parents
            if (self.qd_type == "cvt") or (self.qd_type=="grid"):
                x = archive[keys[rand1[n]]]
                y = archive[keys[rand2[n]]]
            elif (self.qd_type == "unstructured"):                    
                x = archive[rand1[n]]
                y = archive[rand2[n]]
                
            # copy & add variation
            z = variation_operator(x.x, y.x, params)

            if batch:
                to_evaluate += [z]
            else: 
                to_evaluate += [(z, f)]

        return to_evaluate
    
    def addition_condition(self, s_list, archive, params):
        add_list = [] # list of solutions that were added
        discard_list = []
        for s in s_list:
            if self.qd_type == "unstructured":
                success = unstructured_container.add_to_archive(s, archive, params)
            else:
                success = cvt.add_to_archive(s, s.desc, self.archive, self.kdt)
            if success:
                add_list.append(s)
            else:
                discard_list.append(s) #not important for alogrithm but to collect stats
                
        return archive, add_list, discard_list
    
    def compute(self,
                num_cores_set,
                max_evals=1e6,
                params=None,):

        if params is None:
            params = self.params

        # setup the parallel processing pool
        if num_cores_set == 0:
            num_cores = multiprocessing.cpu_count() # use all cores
        else:
            num_cores = num_cores_set
            
        #pool = multiprocessing.Pool(num_cores)
        pool = get_context("spawn").Pool(num_cores)
        #pool = ThreadPool(num_cores)
        
        gen = 0 # generation
        n_evals = 0 # number of evaluations since the beginning
        b_evals = 0 # number evaluation since the last dump

        print("################# Starting QD algorithm #################")

        # main loop
        while (n_evals < max_evals):
            # lists of individuals we want to evaluate (list of tuples) for this gen
            # each entry in the list is a tuple of the genotype and the evaluation function
            to_evaluate = []

            ## intialize for time related stats ##
            gen_start_time = time.time()
            self.model_train_time = 0

            # random initialization of archive - start up
            if len(self.archive) <= params['random_init']*self.n_niches:
                to_evaluate = self.random_archive_init(to_evaluate)
                start = time.time()
                s_list = cm.parallel_eval(evaluate_, to_evaluate, pool, params)
                self.eval_time = time.time() - start 
                self.archive, add_list, _ = self.addition_condition(s_list, self.archive, params)
                
            else:
                # variation/selection loop - select ind from archive to evolve
                
                #tmp_archive = self.archive.copy() # tmp archive for stats of negatives
                # uniform selection of emitter 
                emitter = 0 #np.random.randint(3)
                if emitter == 0: 
                    add_list, to_evaluate = self.random_emitter(to_evaluate, pool, params, gen)
                elif emitter == 1:
                    add_list_model, to_evaluate = self.optimizing_emitter(to_model_evaluate, pool, params, gen)
                elif emitter == 2: 
                    add_list_model, to_evaluate = self.random_walk_emitter(to_model_evaluate, pool, params, gen)


            # count evals
            gen += 1 # generations
            n_evals += len(to_evaluate) # total number of  real evals
            b_evals += len(to_evaluate) # number of evals since last dump
            
            #print("n_evals: ", n_evals)
            print("b_evals: ", b_evals)

            # write archive during dump period
            if b_evals >= params['dump_period'] and params['dump_period'] != -1:
                # write archive
                print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
                cm.save_archive(self.archive, n_evals, params, self.log_dir)
                b_evals = 0

            # write log -  write log every generation 
            if (self.qd_type=="cvt") or (self.qd_type=="grid"):
                fit_list = np.array([x.fitness for x in self.archive.values()])
                self.log_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(gen,
                                         n_evals,
                                         n_model_evals, 
                                         len(self.archive.keys()),
                                         fit_list.max(),
                                         np.sum(fit_list),
                                         np.mean(fit_list),
                                         np.median(fit_list),
                                         np.percentile(fit_list, 5),
                                         np.percentile(fit_list, 95)))

            elif (self.qd_type=="unstructured"):
                fit_list = np.array([x.fitness for x in self.archive])
                self.log_file.write("{} {} {} {} {} {} {} {} {}\n".format(
                    gen,
                    n_evals,
                    len(self.archive),
                    fit_list.max(),
                    np.sum(fit_list),
                    np.mean(fit_list),
                    np.median(fit_list),
                    np.percentile(fit_list, 5),
                    np.percentile(fit_list, 95),))
                
            self.log_file.flush() # writes to file but does not close stream

            self.gen_time = time.time() - gen_start_time 

            #print("Archive size: ", len(self.archive))
                
        print("==========================================")
        print("End of QD algorithm - saving final archive")        
        cm.save_archive(self.archive, n_evals, params, self.log_dir)
        return self.archive


    ##################### Emitters ##############################
    def random_emitter(self, to_evaluate, pool, params, gen):
        start = time.time()
        add_list_final = []
        all_eval = []
        
        to_model_evaluate = self.select_and_mutate(to_evaluate, self.archive, self.f_real, params)
        s_list = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)
        self.archive, add_list, discard_list = self.addition_condition(s_list, self.archive, params)
        add_list_final += add_list
        all_eval += to_evaluate # count all inds evaluated by model
        #print("to model eval length: ",len(to_model_evaluate)) 
        #print("s list length: ",len(s_list_model)) 
        #print("model list length: ",len(add_list_model_final)) 
        #print("all model evals length: ", len(all_model_eval))
        self.model_eval_time = time.time() - start         
        return add_list_final, all_eval
    
    def optimizing_emitter(self, to_model_evaluate, pool, params, gen):
        '''
        uses CMA - no mutations
        '''
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        rand1 = np.random.randint(len(self.archive))
        mean_init = (self.archive[rand1]).x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 100
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})
        
        #for i in range(max_iterations):
        i = 0 
        while not es.stop():
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_real)]
            s_list_model = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)

            self.archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for optimizing emitter fitness of CMAES is fitness of the ind
            reward_list = []
            for s in s_list_model:
                reward_list.append(s.fitness)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            es.disp()

            # save archive at every cmaes iteration
            if i%5==0:
                cm.save_archive(self.archive, str(gen)+"_"+str(i), params, self.log_dir)
            i +=1
        self.model_eval_time = time.time() - start
        
        return add_list_model_final, all_model_eval

    def random_walk_emitter(self, to_model_evaluate, pool, params, gen):
        start = time.time()
        add_list_model_final = []
        all_model_eval = []

        # sample an inidivudal from the archive to init cmaes
        rand1 = np.random.randint(len(self.archive))
        ind_init = self.archive[rand1]
        mean_init = ind_init.x
        sigma_init = 0.01
        popsize = 50
        max_iterations = 50
        es = cma.CMAEvolutionStrategy(mean_init,
                                      sigma_init,
                                      {'popsize': popsize,
                                       'bounds': [0,1]})

        # sample random vector/direction in the BD space to compute CMAES fitness on
        # BD space is 2 dim
        desc_init = ind_init.desc
        target_dir = np.random.uniform(-1,1,size=2)
        
        #for i in range(max_iterations):
        i = 0
        while not es.stop(): 
            to_model_evaluate = []
            solutions = es.ask()
            for sol in solutions:
                to_model_evaluate += [(sol, self.f_real)]
            s_list_model = cm.parallel_eval(evaluate_, to_model_evaluate, pool, params)

            self.archive, add_list_model, discard_list_model = self.addition_condition(s_list_model, self.archive, params)
            add_list_model_final += add_list_model
            all_model_eval += to_model_evaluate # count all inds evaluated by model
            #print("model list length: ",len(add_list_model_final)) 
            #print("all model evals length: ", len(all_model_eval))

            # convert maximize to minimize
            # for random walk emitter, fitnes of CMAES is the magnitude of vector in the target_direction
            reward_list = []
            for s in s_list_model:
                s_dir = s.desc - desc_init
                comp_proj = (np.dot(s_dir, target_dir))/np.linalg.norm(target_dir)
                reward_list.append(comp_proj)

            cost_arr = -np.array(reward_list)
            es.tell(solutions, list(cost_arr))
            es.disp()

            # save archive at every cmaes iteration
            if i%10==0:
                cm.save_archive(self.archive, str(gen)+"_"+str(i), params, self.log_dir)
            i +=1
            
        self.model_eval_time = time.time() - start
        return add_list_model_final, all_model_eval
    
    def improvement_emitter():
    
        return 1

    
    def model_disagr_emitter():
        
        #return add_list_model_final, all_model_eval
        return 1
    
    
    ################## Custom functions for Model Based QD ####################
    def serial_eval(self, evaluate_function, to_evaluate, params):
        s_list = map(evaluate_function, to_evaluate)
        return list(s_list)
