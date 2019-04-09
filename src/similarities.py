import numpy as np
from time import time
from numpy.linalg import norm

## Network libraries
import networkx as nx
import community

# Constructed as in [1]
### Role-Based Similarity Matrix
def RBS(G, α=0.95, K=20):
    '''Nx2K Feature Matrix with in & out paths of length `K` for every node. Column convergence is weighted by α.'''
    A = nx.adjacency_matrix(G)
    λ = np.max( nx.adjacency_spectrum(G) )
    β = np.real(α/λ)

    n = len(G.nodes)
    X = np.zeros( (n, 2*K) )
    for l in range(K):
        # is it efficient to elevate the matrix to its power everytime?
        X[:, l]   = ((β*np.transpose(A))**(l+1)).toarray().sum(axis=1)
        X[:, l+K] = ((β*A)**(l+1)).toarray().sum(axis=1)

    return X

### Dynamical Embedding Similarity Matrix
def DES(G, t):
    ''' Markov process of a walker. It is adapted for both directed and undirected graphs.
    For directed graphs, the out-degree is taken for the transition matrix M. '''
    try: # for directed graphs

        D = np.diag(list(dict(G.out_degree).values()))

        # the case where D_ii = 0 is substituted with D_ii = 1
        for i in range(len(D)):
            if D[i,i] == 0:
                D[i,i] = 1
    except: # for non-directed graphs
        D = np.diag(list(dict(G.degree).values()))

    A = nx.adjacency_matrix(G)
    M = np.linalg.inv(D) * A

    return np.exp(M*t)

### Cosine Similarity
def CosineSimilarity(X):
    '''Cosine-similarity of a matrix.'''
    n = len(X)

    Y = np.zeros( (n,n) )
    for i in range(n):
        Y[i,i] = 1.0
        for j in range(n):
            if i < j:
                Y[i,j] = np.inner(X[i,:], X[j,:])/ (norm(X[i,:])*norm(X[j,:]))
                Y[j,i] = Y[i,j]

    return Y
