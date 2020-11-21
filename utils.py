from scipy.special import binom
import numpy as np

def uniform_error_channel(n_max, x):
    '''
    n_max = maximum number of bits to be changed
    x = message to be modified
    '''
    res = x.copy()
    
    #computing the probability of changing n bits
    representativity = np.array([binom(len(x), i) for i in range(n_max + 1)])
    representativity /= np.sum(representativity)

    #choosing how many bits to change 
    k = np.random.choice(n_max + 1, p=representativity)
    #choosing which bits to change
    bit_flips = np.random.choice(len(x), size=(k), replace=False)

    #flipping the bits
    for i in bit_flips:
        res[i] += 1

    return res % 2

def channel(x):

    return uniform_error_channel(1, x), uniform_error_channel(3, x)

