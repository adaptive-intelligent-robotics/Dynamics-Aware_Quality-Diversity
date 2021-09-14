import numpy as np

'''
Unstructured container - or sort based archive
if behavior space cannot be discretized in an efficient manner or not known in advance
No bin or grids
New rules for addition of new individuals based on: 
1. Novelty
2. Fitness 
3. Some threshold lower than the novelty and fitness

archive/container used to store descriptors that we do not know how to discretize yet
'''

def sign(d):
    if d < 0:
        return -1
    return 1


def get_novelty(ind, nn_list):
    '''
    computes novelty based on nn given
    '''
    c_sum = 0 
    for nn in nn_list:
        c_sum = c_sum + np.linalg.norm(nn.desc-ind.desc)
    nov = c_sum/len(nn_list)
    return nov

'''
archive is a dictionary in normal cvt/pymapelites grid version
probably easier to be a list for unstructured archive due to sorting
therefore, if using this ARCHIVE IS A LIST
Input: 
a Species object (i.e. an individual) - containing s.fitness, s.desc, s.x  
archive - full list of all existing species objects in the archive
'''
def add_to_archive(s, archive, params):
    # move externally to params - take argument params
    nov_l = params["nov_l"] #0.01 # novelty threshold - can be moved externally to params
    eps = params["eps"] #0.01 # therhold to accept can be moved exterally to params 
    k = params["k"] #4

    #print(s)
    # if fitness set to dead - dont add
    if s.fitness == "dead":
        #print("Solution is dead (conditional) - do not add to archive")
        #This dead method works - if something is dead set fitness to dead
        return 0

    # if empty archive or if nearest neighbour is far enought (i.e. novel enough)
    if (len(archive) == 0):
        # if empty archive - add
        archive.append(s)
        return 1
    elif (np.linalg.norm(nearest(s.desc,archive)[0].desc - s.desc) > nov_l):
        # if novel enough - add
        archive.append(s)
        return 1
    elif len(archive) == 1:
        # if this ind is too close and there is only 1 ind in archive - dont add
        return 0
    else:
        # consider replacement by computing all of the following
        
        # find 2 nearest neighbours of this individual based on decriptor
        neighbours, nn2_idx = knn(s.desc, 2, archive) 
        
        # if the 2 nearest neighbour is too close
        if np.linalg.norm(s.desc - neighbours[1].desc) < (1-eps)*nov_l:
            return 0

        # get nearest neighbour - but pop it from archive - meaning it is removed tmp
        nn = neighbours[0]        

        #initialize a score array - conatining fit score, novelty score
        #for current ind and the nn of the ind
        score_cur = np.zeros(2)
        score_nn = np.zeros(2)

        # set fitness scores
        score_cur[0] = s.fitness
        score_nn[0] = nn.fitness

        # compute novlety score - get k+1 nn for both 
        neighbours_cur, _ = knn(s.desc, k+1, archive)
        neighbours_nn, _ = knn(nn.desc, k+1, archive)
        # get novelty score
        score_cur[1] = get_novelty(s, neighbours_cur)
        score_nn[1] = get_novelty(nn, neighbours_nn)

        if (score_cur[0] >= (1 - sign(score_nn[0])*eps)*score_nn[0]) and \
           (score_cur[1] >= (1 - sign(score_nn[1])*eps)*score_nn[1]) and \
           ((score_cur[0]-score_nn[0])*abs(score_nn[1]) > -(score_cur[1]-score_nn[1]*abs(score_nn[0]))):
            # replace nn point wiht current point
            archive.pop(nn2_idx[0]) # remove the nn from the archive to be replaced
            archive.append(s)
            #print("Replaced")
            return 1
        else:
            # leave the archive as is - no replacement
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

