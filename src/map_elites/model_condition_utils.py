import numpy as np

'''
For model based QD
After evaluating solutions in the model - use these conditions to see if we should ACTUALLY evaluate them.
Evaluate solutions based on the some novelty threshold and quality threshold

Find n nearest neighbours of the solution we are intereted in
- compute novelty score based on n nearest neighbours (avg distance with k-nn)
- compute fitness score based on n nearest neighbours (avg fitness difference with k-nn)

there are hyperparameters which are threholds which determine how much we trust the model
- if the novelty score is greater than the threhold OR
- if the fitness socre is greater than the threshold 

we will let it be evaluated in the real world

'''

def sign(d):
    if d < 0:
        return -1
    return 1

def get_novelty(ind, nn_list):
    '''
    computes novelty score based on nn given
    - sum the distance to nn in nn_list
    - nn_list already contained the k nn that we are interested in
    '''
    c_sum = 0 
    for nn in nn_list:
        c_sum = c_sum + np.linalg.norm(nn.desc-ind.desc)
    nov = c_sum/len(nn_list)
    return nov

def get_fitness_score(ind, nn_list):
    '''
    compute fitness score based on nn
    Fitness is a scalar - and always higher fitness (more positive or less negative) is better
    So the difference in fitness  fi the fitness of ind is always better will alwasy be > 0
    and always be < 0 if the fitness of ind is lower/worse (no reason for a absolute)
    '''
    c_sum = 0 
    for nn in nn_list:
        c_sum = c_sum + (ind.fitness-nn.fitness)

    fit_score = c_sum/len(nn_list)

    return fit_score


'''
archive is a dictionary in normal cvt/pymapelites grid version
probably easier to be a list for unstructured archive due to sorting
therefore, if using this ARCHIVE IS A LIST

Input: 
a Species object (i.e. an individual) - containing s.fitness, s.desc, s.x  
archive - full list of all existing species objects in the archive 
- we need the archive because we need to compute the nn of the input individual

IMPORTANT: DIFFERENT from the addition condition to add to archive for unstructure archive
- as this is addition condition in order to see whether you want the solutions evaluated by the model
In the normal addition condition of the unstructure archive: 
- notion of a fitness score does not exists 
- quality threhold does not really exists - it is somewhat captured in the epsilon parameter
'''
def add_to_archive(s, archive, params):

    t_nov = params["t_nov"]  # novelty threshold
    t_qua = params["t_qua"]  # quality threshold
    k = params["k_model"] 

    # get the k nn of the individual we are evaluating
    neighbours_cur, _ = knn(s.desc, k, archive)

    # get novelty score
    nov_score_cur = get_novelty(s, neighbours_cur)
    # get fitness score 
    fit_score_cur = get_fitness_score(s, neighbours_cur)

    if (nov_score_cur > t_nov) or (fit_score_cur > t_qua):
        archive.append(s)
        return 1
    else:
        return 0


def add_to_archive_2(s, archive, add_params):
    '''
    used for more flexibilitiy to change the thresholds as input to the function
    '''
    t_nov = add_params[1]  # novelty threshold
    t_qua = add_params[0]  # quality threshold
    k = add_params[2] 

    # get the k nn of the individual we are evaluating
    neighbours_cur, _ = knn(s.desc, k, archive)

    # get novelty score
    nov_score_cur = get_novelty(s, neighbours_cur)
    # get fitness score 
    fit_score_cur = get_fitness_score(s, neighbours_cur)

    if (nov_score_cur > t_nov) or (fit_score_cur > t_qua):
        archive.append(s)
        return 1
    else:
        return 0

    
def knn(desc, k, archive):
    '''
    finds the k-nearest neightbour of a particular desciptor
    INPUT: 
    selected descriptor
    k
    OUTPUT: 
    a smaller dataset of the k nearest neighbours to this point
    list of tuples
    '''
    k = np.amin([len(archive),k])

    distances = np.zeros(len(archive))   
    for i in range(len(archive)):
        #print("desc: ",desc)
        #print("archive i desc: ",archive[i].desc)
        distances[i] = np.linalg.norm(desc-archive[i].desc)

    # can use full sort (np.argsort) as well - but partition is partial sort which would proabably save comptutation
    # note that partition does not care about order wihthin the before k and after k by default - have to use a seqeunce/range of k for it to make sure it is sorted
    indices = np.argpartition(distances, range(k)) 

    # select all the data values which are the k-nearest neighbours
    k_nearest_n = []
    for i in range(k):
        k_nearest_n.append(archive[indices[i]])

    # returns the list of objects AND indices of the knn in the archive
    return k_nearest_n, indices

def nearest(desc, archive):
    '''
    finds the nearest neighbour (1-nn)
    returns the individual
    '''
    #compute distances of all the data points to the selected 
    distances = np.zeros(len(archive))   
    for i in range(len(archive)):
        #print("desc: ",desc)
        #print("archive i desc: ",archive[i].desc)
        distances[i] = np.linalg.norm(desc-archive[i].desc)

    nearest_n_index = np.argmin(distances)    
    nearest_n = archive[nearest_n_index]

    # also returns the nn index in the archive
    return nearest_n, nearest_n_index


'''
# test functions
random_list = np.random.randint(10,size=10)
print(random_list)

#print(np.argmin(random_list))
sorted_list = np.partition(random_list, range(4))
print(sorted_list)
sorted_idx = np.argpartition(random_list, range(4))
print(sorted_idx)


#output = np.argpartition(random_list, (0,1))
#print(output)
'''

