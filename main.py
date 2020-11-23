import numpy as np
from utils import channel, binary_channel, hamming_distance, get_distribution, np_to_number
from utils import compute_mutual, compute_joint, compute_marginal_z
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

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

#here I am supposing that the input messages are uniformly distributed
marginal_u = np.ones(8) / 8
conditional = np.zeros((8, 2**7))

for i, u in tqdm(enumerate(product([0, 1], [0, 1], [0, 1]))):
    u = np.array(u)
    #keep this high enough so we are sure that there is one 
    #sample for each of the possible z
    unique, counts = get_distribution(u, 10 ** 4)

    cond_prob = counts / np.sum(counts)
    uniques = [np_to_number(uni) for uni in unique]
    conditional[i, uniques] = cond_prob

joint = compute_joint(conditional, marginal_u)
marginal_z = compute_marginal_z(joint)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(2 ** 3)
Y = np.arange(2 ** 7)
X, Y = np.meshgrid(X, Y)
ax.plot_surface(
        X,
        Y, 
        conditional.T,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
        vmin = 0,
        vmax = 1
        )
ax.set_zlim(0, 0.04)
plt.show()

print(f"sum of the conditional probabilities: {np.sum(conditional, axis = 1)}")
print(f"sum of the marginal probability for z: {np.sum(marginal_z)}")
print(f"sum of the joint probabilities: {np.sum(joint)}")

mutual_information = compute_mutual(
        joint,
        marginal_u,
        marginal_z
)

print(f"Mutual Information = {mutual_information}")

print("\nTASK 5\n")
epsilon = 0.8
n_bits = 10**4
x = np.random.randint(2, size=n_bits)
y = binary_channel(x, epsilon)
print(f"epsilon: {epsilon}, estimated = {hamming_distance(x, y)/n_bits}")

