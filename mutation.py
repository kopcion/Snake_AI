import numpy as np
from random import randint
def simulated_binary_crossover(parent1, parent2):
    c1 = np.copy(parent1)
    c2 = np.copy(parent2)
    eta = 100
    rand = np.random.random(c1.shape)
    gamma = np.empty(c1.shape)

    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))

    return 0.5 * ((1 + gamma) * c1 + (1 - gamma) * c2), 0.5 * ((1 - gamma) * c1 + (1 + gamma) * c2)

def single_point_binary_crossover(parent1, parent2):
    c1 = np.copy(parent1)
    c2 = np.copy(parent2)
    row = np.random.randint(0, parent2.shape[0])
    col = np.random.randint(0, parent2.shape[1])
    c1[:row, :] = parent2[:row, :]
    c2[:row, :] = parent1[:row, :]
    c1[row, :col+1] = parent2[row, :col+1]
    c2[row, :col+1] = parent1[row, :col+1]
    
    return c1, c2

#t = np.array([1 for i in range(100)])
#t = t.reshape(10,10)
#t1 = np.zeros((10,10))

#print(single_point_binary_crossover(t, t1))
