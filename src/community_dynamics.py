import numpy as np
from time import time
from numpy.linalg import norm

## Network libraries
import networkx as nx
import community

### Variation of Information (normalised) between 2 partitions
def entropy(categorical_vec):
    _, counts = np.unique(categorical_vec, return_counts=True)
    probs = counts / len(categorical_vec)

    return -np.sum(probs * np.log2(probs))

def joint_entropy(categorical_vec1, categorical_vec2):
    assert len(categorical_vec1) == len(categorical_vec2)

    unique_categs1 = np.unique(categorical_vec1)
    unique_categs2 = np.unique(categorical_vec2)

    probs = np.zeros((len(unique_categs1), len(unique_categs2)))

    for i, c in enumerate(unique_categs1):
        idx_c = (categorical_vec1 == c)
        for j, d in enumerate(unique_categs2):
            idx_d = (categorical_vec2 == d)
            probs[i,j] = np.sum(np.bitwise_and(idx_c, idx_d))

    probs /= len(categorical_vec1)
    probs = probs[probs.nonzero()] # we use the fact that lim(x->0) x log x = 0

    return -np.sum(probs * np.log2(probs))

def varinfo(x, y): return (2*joint_entropy(x, y) - entropy(x) - entropy(y)) / np.log2(len(x))

## Sweeping community detection for different Markov times as discussed in [3].
def community_resolution(G, λ=20/19, num_rounds=1):
    '''
    Maximization of modularity at different resolution Markov Times. Returns the average of all the rounds
    in `num_rounds` calculated.

    Inputs:
        G - A networkx graph; it can be directed and weighted
        λ - fraction of time forward to compare VI of partitions.
        num_rounds - Number of rounds the graph is optimised for a given Markov time
    Outputs:
        resolution - Markov Times array.
        modularity - Modularity for each partition at each time in `resolution`.
        num_comms  - Number of communities at each time in `resolution`.
        vi         - Variation of information between the best community at time `t` and `λt`, where λ=20/19.
    '''
    #This could be extended for further control with more inputs.

    resolution = np.logspace(-1,1.5, 100)

    modularity = np.zeros(len(resolution))
    vi         = np.zeros(len(resolution))
    num_comms  = np.zeros(len(resolution), int)

    for _round in range(num_rounds):
        for (ix,t) in enumerate(resolution):
            partition_1 = community.best_partition(G, resolution=t, random_state=_round)
            partition_2 = community.best_partition(G, resolution=λ*t, random_state=_round)

            modularity[ix] += community.modularity(partition_1, G)
            num_comms[ix]  += len(set(partition_1.values()))
            vi[ix]         += varinfo(list(partition_1.values()), list(partition_2.values()))

    return resolution, modularity/num_rounds, num_comms/num_rounds, vi/num_rounds
