import numpy as np


def distance(bd_pop):
    """
    Inputs an array of all the BD in the population
    Outputs the distances
    # BD pop is (N, bd_dim)
    # N=number of samples in dataset
    # bd_dim= 2 (for ballistic task) otuput of autoencoder
    """
    X = bd_pop
    N = len(bd_pop)
    # compute norms for each BD 
    XX = np.reshape(np.sum(np.square(X), axis=1), (-1,1)) # (N,1)
    XY = (2*X)@np.transpose(X) # (N,N)

    dist = XX@(np.ones((1,N)))
    dist = dist + (np.ones((N,1)))@(np.transpose(XX))
    dist = dist - XY
    
    return dist
    
def get_new_l(bd_pop, params):
    """
    Takes BD of population as input, outputs new l value
    AURORA, Cully et al. 
    Works only for low dimnesional latent space BD (works bad for higher than 2)
    Use another adpative l for higher dimension latent space BD better perfromance
    """
    dist = distance(bd_pop)
    maxdist = np.sqrt(np.amax(dist))

    K = 60000. # arbitrary value to have a specific resolution
    new_l = maxdist/np.sqrt(K)

    return new_l


def get_new_l_Luca(bd_pop, params):
    """
    get new l value: 
    Luca et al. 
    works better for high dimensional latent space
    """
    resolution = 9000 # todo: PLACE THIS in the params 
    alpha = 2e-5 # hyperparameter
    pop_size = len(bd_pop)
    print("Current population size: ", pop_size)
    print("Resolution: ", resolution)
    
    new_l = params["nov_l"]*(1 - alpha*(resolution-pop_size))

    return new_l






if __name__=="__main__":

    """
    #quick test to check if it is working
    bd_pop = np.random.rand(10,2)
    print(bd_pop)
    print(bd_pop*bd_pop)
    print(np.square(bd_pop))

    print(np.sum(bd_pop*bd_pop, axis=1))
    print(np.sum(np.square(bd_pop), axis=1))

    #print("here", np.matmul((2*bd_pop),np.transpose(bd_pop)))    
    #print("here", (2*bd_pop)@np.transpose(bd_pop))

    dist = distance(bd_pop)
    print(dist)
    print(np.amax(dist))
    print(np.argmax(dist))

    print(get_new_l(bd_pop))
    
    """
