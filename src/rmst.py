import numpy as np
from time import time
from numpy.linalg import norm

## Network libraries
import networkx as nx
import community

## Relaxed-Minimum spanning tree as described in [2]
def RMST(Z, γ):
    E = nx.minimum_spanning_tree(Z)
    # clear weight from MST
    for (n1, n2, w) in E.edges(data=True):
        w.clear()

    # init RMST
    R = nx.Graph(E.edges)
    for (ix_i, node_i) in enumerate(E.nodes):
        for (ix_j, node_j) in enumerate(E.nodes):

            if ix_i < ix_j:
                # there will not always be a connection between 2 nodes.
                try:
                    z_ij = Z[node_i][node_j]['weight']
                except:
                    z_ij = np.inf

                mlink_ij = -np.inf
                path_ij = list(nx.all_shortest_paths(E, source=node_i, target=node_j))[0]

                node_source = node_i
                for node_target in path_ij[1:]:
                    mlink_ij = max(mlink_ij, Z[node_source][node_target]['weight'])
                    node_source = node_target

                if (mlink_ij + γ > z_ij):
                    R.add_edge(node_i, node_j)

    return R, E
