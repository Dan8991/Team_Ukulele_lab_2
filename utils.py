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

def binary_channel(eps, x):

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
	return uniform_error_channel(3, x)

def verify_encoder_decoder(n_iter = 10**4):
	print('Verify encoder + decoder')
	errors = 0
	for _ in range(n_iter):
		d = np.random.randint(2, size=3)
		x = random_binning_encoder(d)
		decoded = random_binning_decoder(x)
		if np.sum(np.abs(d-decoded)) != 0:
			error += 1

	print('Error rate: ' + str(errors/n_iter))
	
	
def verify_encoder_channel_decoder(channel, param, n_iter=10**4):
	errors = 0
	for _ in range(n_iter):
		d = np.random.randint(2, size=3)
		x = random_binning_encoder(d)
		y = channel(param, x)
		decoded = random_binning_decoder(y)
		if np.sum(np.abs(d-decoded)) != 0:
			errors += 1
	print('Error rate: ' + str(errors/n_iter))


def get_distribution(u, n_tests = 10**4):
    
    z = []
    for _ in range(n_tests):
        z.append(encoder_eavesdropper(u))

    return np.unique(z, axis = 0, return_counts=True)

def np_to_number(x):

    power = np.fromfunction(lambda i, j: 2**(len(x)-1-j), (1, len(x)))[0]
    return int(np.dot(power, x))

def compute_joint(conditional_z, p_u):
    return conditional_z * p_u.reshape((-1, 1))

def compute_marginal_z(joint_prob, axis = 0):
    #with axis = 0 we compute the marginal over z
    return np.sum(joint_prob, axis = axis)



def compute_mutual(joint_prob, marginal_u, marginal_z):
    '''
    joint labels = int representation of the concatenation of u and z
    joint probability = probability of [u, z]
    marginal_u = P(u) where index i contains the probability of the int representation of u
    marginal_z = P(z) where index i contains the probability of the int representation of z
    '''

    information = np.zeros((2 ** 3, 2 ** 7))

    for i in range(2**3):
        for j in range(2**7):
            log_arg = joint_prob[i, j] / marginal_u[i] / marginal_z[j]
            information[i, j] = joint_prob[i, j]*np.log2(log_arg)

    return np.sum(information)



