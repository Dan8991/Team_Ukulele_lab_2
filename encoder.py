import numpy as np
from utils import random_binning_encoder
from tqdm import tqdm

d = np.array([1,0,0])

a = random_binning_encoder(d)

print(a)