from scipy.special import binom
from random import randrange 
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

def random_binning_encoder(d):
    '''
    d = input word to be encoded
    '''
    hamming = np.array([0b0000000,0b1000110,0b0100101,0b0010011,0b0001111,0b1100011,0b1010101,0b1001001,0b0110110,0b0101010,0b0011100,0b1110000,0b1101100,0b1011010,0b0111001,0b1111111])

    #generate Tx|u(d)

    prefix = "{0:04b}".format(d)
    codeword = 0 #binary codeword
    
    #look for matches between hamming code words and the prefix
    for i in range(np.shape(hamming)[0]):
        #obtain string representation of the binary hamming[i]
        binary = "{0:07b}".format(hamming[i]) 
        
        #chechk if prefix matches with the word
        if binary.startswith(prefix):
            codeword = int(binary, base = 2)
            
    #choose whether to return the codeword or the binary complement
    i = randrange(2)
        
    if i:
        bin_string = "{0:07b}".format(codeword)
        #return array of numbers
        return np.array(list(bin_string), dtype=int)
    else:
        bin_string = "{0:07b}".format(~codeword & 0b1111111)
        #return array of numbers
        return np.array(list(bin_string), dtype=int)