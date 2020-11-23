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

for i in tqdm(range(10**6)):
    legitimate, eve = channel(x)
    y.append(legitimate)
    z.append(eve)

y_unique, y_counts = np.unique(y, return_counts=True, axis = 0)
z_unique, z_counts = np.unique(z, return_counts=True, axis = 0)
combinations = np.concatenate([y, z], axis = 1)
comb_unique, comb_counts = np.unique(combinations, return_counts=True, axis = 0)

p_y = y_counts/np.sum(y_counts)
p_z = z_counts/np.sum(z_counts)
p_yz = comb_counts/np.sum(comb_counts)

fig, axs = plt.subplots(1, 3, figsize=(40, 7))
axs[0].scatter(np.arange(len(y_counts)), p_y)
axs[1].scatter(np.arange(len(z_counts)), p_z)
axs[2].scatter(np.arange(len(comb_counts)), p_yz)
axs[0].set_ylabel("P(y)")
axs[1].set_ylabel("P(z)")
axs[2].set_ylabel("P(y, z)")
axs[0].set_ylim([0, 0.2])
axs[1].set_ylim([0, 0.02])
axs[2].set_ylim([0, 0.004])

plt.show()
fig.savefig("distributions.png")

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

