import numpy as np
from utils import channel, binary_channel, hamming_distance
from tqdm import tqdm
import matplotlib.pyplot as plt

x = np.zeros(7)
z = []
y = []

for i in tqdm(range(10**4)):
    legitimate, eve = channel(x)
    y.append(legitimate)
    z.append(eve)

y_unique, y_counts = np.unique(y, return_counts=True, axis = 0)
z_unique, z_counts = np.unique(z, return_counts=True, axis = 0)
combinations = np.concatenate([y, z], axis = 1)
comb_unique, comb_counts = np.unique(combinations, return_counts=True, axis = 0)

'''
fig, axs = plt.subplots(1, 3)
axs[0].bar(np.arange(len(y_counts)), y_counts)
axs[1].bar(np.arange(len(z_counts)), z_counts)
axs[2].bar(np.arange(len(comb_counts)), comb_counts)
plt.show()
plt.savefig("distributions.png")
''' 
epsilon = 0.8
n_bits = 10**4
x = np.random.randint(2, size=n_bits)
y = binary_channel(x, epsilon)
print(f"epsilon: {epsilon}, estimated = {hamming_distance(x, y)/n_bits}")
