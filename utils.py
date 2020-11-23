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

def binary_channel(x, eps):

    error = np.random.choice(2, size=(len(x)), p = (1 - eps, eps))

    return (x + error) % 2

def hamming_distance(x, y):
    return np.sum(np.abs(x-y))

def random_binning_encoder(d):
    '''
    d = numpy array of integers (0s and 1s), the input word to be encoded
    '''
    hamming = np.array([0b0000000,0b1000110,0b0100101,0b0010011,0b0001111,0b1100011,0b1010101,0b1001001,0b0110110,0b0101010,0b0011100,0b1110000,0b1101100,0b1011010,0b0111001,0b1111111])
    
    #get prefix from input array
    prefix = "0"
    for i in range(np.shape(d)[0]):
        prefix += "{0:00b}".format(d[i])

    #---------------------------------generate Tx|u(d)

    codeword = 0 #binary codeword
    
    #look for matches between hamming code words and the prefix
    for i in range(np.shape(hamming)[0]):
        #obtain string representation of the binary hamming[i]
        binary = "{0:07b}".format(hamming[i]) 
        
        #chechk if prefix matches with the word
        if binary.startswith(prefix):
            #need an int in order to quickly do the complement later
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
        
def random_binning_decoder(y):
	X = ['0000000', '1000110', '0100101', '0010011', '0001111', '1100011', '1010101', '1001001', '0110110', '0101010', '0011100', '1110000', '1101100', '1011010', '0111001', '1111111']
	X = [np.asarray(list(a), dtype=int) for a in X]
	min_distance = 7
	min_a = 0 
	for a in X:
		h = hamming_distance(a, y)
		if h < min_distance:
			min_distance = h
			min_a = a
	x = min_a[1:4]
	if min_a[0] == 1:
		x = np.asarray([1-i for i in x])
	return x
	
def encoder_eavesdropper(d):
	x = random_binning_encoder(d)
	x = "{0:07b}".format(x)
	x = np.asarray(list(x), dtype = int)
	return uniform_error_channel(3, x)
