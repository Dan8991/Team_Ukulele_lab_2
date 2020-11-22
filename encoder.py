import numpy as np
from utils import random_binning_encoder
from tqdm import tqdm

a = random_binning_encoder(0b100)

#print the value as a binary string
print("{0:07b}".format(a))