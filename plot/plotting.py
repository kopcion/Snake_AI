import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


averages_24_bin = np.load('all_averages(24,18,12,4)bin.npy', allow_pickle=True)[:200]
maxes_24_bin = np.load('all_maxes(24,18,12,4)bin.npy', allow_pickle=True)[:200]

averages_32_bin = np.load('all_averages(32,20,14,4)bin.npy', allow_pickle=True)[:200]
maxes_32_bin = np.load('all_maxes(32,20,14,4)bin.npy', allow_pickle=True)[:200]

averages_24_dist = np.load('all_averages(24,18,12,4)dist.npy', allow_pickle=True)[:200]
maxes_24_dist = np.load('all_maxes(24,18,12,4)dist.npy', allow_pickle=True)[:200]

averages_24_binBatched = np.load('all_averages(24,18,12,4)binBatched.npy', allow_pickle=True)[:200]
maxes_24_binBatched = np.load('all_maxes(24,18,12,4)binBatched.npy', allow_pickle=True)[:200]

fig = plt.figure(1, figsize=(20,8))

averages = fig.add_subplot(211)
averages.plot(range(0, 200), averages_24_bin,
              range(0, 200), averages_32_bin,
              range(0, 200), averages_24_dist,
              range(0, 200), averages_24_binBatched)
averages.xaxis.set_major_locator(MaxNLocator(integer=True))
averages.legend(('binary 24','binary 32', 'distance', 'binary batched'))

maxes = fig.add_subplot(212)
maxes.scatter(range(0, 200), maxes_24_bin)
maxes.scatter(range(0, 200), maxes_32_bin)
maxes.scatter(range(0, 200), maxes_24_dist)
maxes.scatter(range(0, 200), maxes_24_binBatched)
maxes.legend(('binary 24','binary 32', 'distance', 'binary batched'))
maxes.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()