import numpy as np
from utils import channel, binary_channel, hamming_distance, get_distribution, np_to_number
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product

print("TASK 1\n")
x = np.zeros(7)
z = []
y = []

for i in tqdm(range(10**2)):
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

print("\nTASK 4\n")

plt.figure()
joint_labels = [] 
joint_counts = []
marginal_z = np.zeros((2**7))
marginal_u = 1/8
information = np.zeros((2**10))
conditional = np.zeros((8, 2**7))

for i, u in tqdm(enumerate(product([0, 1], [0, 1], [0, 1]))):
    u = np.array(u)
    unique, counts = get_distribution(u,10**4)

    #concatenating in order to compute the joint distribution
    u_concat = np.repeat(u.reshape((1, len(u))), len(unique),axis = 0)
    joint = np.concatenate([u_concat, unique], axis = 1)
    joint = [np_to_number(j) for j in joint]
    joint_labels.append(joint)
    joint_counts.append(counts)

    cond_prob = counts / np.sum(counts)
    uniques = [np_to_number(uni) for uni in unique]

    for j, unique in enumerate(uniques):
        marginal_z[unique] += counts[j]

    # plt.bar(uniques, cond_prob, label=np_to_number(u))
    conditional[i, :] = cond_prob

print(np.min(conditional), np.max(conditional))
plt.imshow(conditional)
plt.legend()
plt.show()

print("\nTASK 5\n")
epsilon = 0.8
n_bits = 10**4
x = np.random.randint(2, size=n_bits)
y = binary_channel(x, epsilon)
print(f"epsilon: {epsilon}, estimated = {hamming_distance(x, y)/n_bits}")

