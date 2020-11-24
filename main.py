import numpy as np
from math import sqrt
from utils import channel, binary_channel, hamming_distance, get_distribution, np_to_number
from utils import compute_mutual, compute_joint, compute_marginal_z, uniform_error_channel
from utils import verify_encoder_decoder, verify_encoder_channel_decoder, get_many_z_bsc
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

print("TASK 1\n")
x = np.array([1, 0, 0, 1, 0, 0, 0])
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
p_z = np.zeros((2**7))
p_y = np.zeros((2**7))
p_yz = np.zeros((2**14))
y_unique = [np_to_number(i) for i in y_unique]
z_unique = [np_to_number(i) for i in z_unique]
comb_unique = [np_to_number(i) for i in comb_unique]
p_z[z_unique] = z_counts/np.sum(z_counts)
p_y[y_unique] = y_counts/np.sum(y_counts)
p_yz[comb_unique] = comb_counts/np.sum(comb_counts)

fig, axs = plt.subplots(1, 3, figsize=(40, 7))
axs[0].scatter(np.arange(2**7), p_y)
axs[1].scatter(np.arange(2**7), p_z)
axs[2].scatter(np.arange(2**14), p_yz)
axs[0].set_ylabel("P_y|x(b|1001000)")
axs[1].set_ylabel("P_z|x(c|1001000)")
axs[2].set_ylabel("P_y,z|x(b, c|1001000)")
axs[0].set_ylim([0, 0.2])
axs[1].set_ylim([0, 0.02])
axs[2].set_ylim([0, 0.004])

plt.show()
fig.savefig("distributions.png")

print("\nTASKS 2 and 3\n")
# verify that encoder + decoder makes no error

verify_encoder_decoder()

# verify that encoder + legitimate channel + decoder makes no error

verify_encoder_channel_decoder(uniform_error_channel, 1)

print("\nTASK 4\n")

#here I am supposing that the input messages are uniformly distributed
marginal_u = np.ones(8) / 8
conditional = np.zeros((8, 2**7))

for i, u in tqdm(enumerate(product([0, 1], [0, 1], [0, 1]))):
    u = np.array(u)
    #keep this high enough so we are sure that there is one 
    #sample for each of the possible z
    unique, counts = get_distribution(u, 10 ** 3)

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
ax.set_xlabel("u")
ax.set_ylabel("z")
ax.set_zlabel("p(z|u)")
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
y = binary_channel(epsilon, x)
print(f"epsilon: {epsilon}, estimated = {hamming_distance(x, y)/n_bits}")

print("\nTASK 6\n")

eps = [10**-3, 10**-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1-10**-2, 1-10**-3]
errors = []
for i in tqdm(eps):
    errors.append(verify_encoder_channel_decoder(binary_channel, i,n_iter = 10**2, verbose=False))
	
plt.plot(eps, errors, 'bo', eps, errors, 'r--')
plt.yscale('log')
plt.xlabel('epsilon')
plt.ylabel('error rate')
plt.show()

mutual_information_eps = []
#searching for the mutual information
for e in tqdm(eps):
    #here I am supposing that the input messages are uniformly distributed
    marginal_u = np.ones(8) / 8
    conditional = np.zeros((8, 2**7))

    for i, u in enumerate(product([0, 1], [0, 1], [0, 1])):
        u = np.array(u)
        #keep this high enough so we are sure that there is one 
        #sample for each of the possible z
        unique, counts = get_many_z_bsc(u, e, 10 ** 3)

        cond_prob = counts / np.sum(counts)
        uniques = [np_to_number(uni) for uni in unique]
        conditional[i, uniques] = cond_prob

    joint = compute_joint(conditional, marginal_u)
    marginal_z = compute_marginal_z(joint)

    # print(f"sum of the conditional probabilities: {np.sum(conditional, axis = 1)}")
    # print(f"sum of the marginal probability for z: {np.sum(marginal_z)}")
    # print(f"sum of the joint probabilities: {np.sum(joint)}")

    mutual_information_eps.append(compute_mutual(
            joint,
            marginal_u,
            marginal_z
    ))

plt.plot(eps, mutual_information_eps, 'bo', eps, mutual_information_eps, 'r--')
# plt.yscale('log')
plt.xlabel('delta')
plt.ylabel('mutual information between u and z')
plt.show()


# compute an upper bound to the mechanism security 
# in terms of distinguishability from the ideal counterpart

bounds = []

#compute bounds for each (eps, delta) pair
for epsilon in range(len(eps)):
    #form an array of bounds for each epsilon
    results = []
    
    for delta in range(len(eps)):
        bound = errors[epsilon] + 1/2*(sqrt(mutual_information_eps[delta]))
        results.append(bound) 
    
    results = np.asarray(results)    
    bounds.append(results)

bounds = np.asarray(bounds)

#plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.asarray(eps)
Y = np.asarray(eps)
X, Y = np.meshgrid(X, Y)
ax.plot_surface(
        X,
        Y, 
        bounds,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        vmin = 0,
        vmax = 1
        )
ax.set_zlim(0, 2)
ax.set_xlabel("delta")
ax.set_ylabel("epsilon")
ax.set_zlabel("bound")
plt.show()
