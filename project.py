import numpy as np
from time import time
from numpy.linalg import norm

## Network libraries
import networkx as nx
import community

## Plotting libraries
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
#         'weight' : 'bold',
        'size'   : 16}

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
plt.rc('legend', fontsize=15)
matplotlib.rc('font', **font)

# global G
#
# # Parse the command
#     parser = argparse.ArgumentParser(description='Similarity Comparisson')
#     parser.add_argument(action='store', dest='G', help='NetworkX Graph')
#     args = parser.parse_args()

from community_dynamics import *
from rmst import *
from similarities import *


### Plotting
def draw_community_graph(G, partition, seed=182, pos=None, figsize=None, num_comm_threshold=None, with_labels=False, degree_thres=10):
    '''
    Plots a network separated by colors depending on their communities. Node size are proportional
    to the pagerank of each node.

    Inputs:
        G          - A networkx graph; it can be directed and weighted
        partition  - Partition of communities in G as a {node: community} dict.
        -----------------------------------------------------------------------
        pos                - Defaults to a spring_layout representation of the network for a fixed seed.
        seed=182           - seed for the spring_layout of the network.
        fisize             - tuple. Defaults at (7, 7)
        num_comm_threshold - Number of communities coloured. Defaults to none, which means every community is coloured differently.

    Outputs:
        pagerank - Returns the pagerank of each node.
    '''

    if pos == None:
        pos = nx.layout.spring_layout(G, seed=seed)

    if num_comm_threshold != None:
        for (ix,com) in enumerate(partition):
            if com >= num_comm_threshold:
                partition[ix] = num_comm_threshold

    pagerank = nx.pagerank(G)
    node_size = 30000 * np.array([pagerank[node] for node in list(G.nodes)])

    if figsize == None:
        plt.figure( figsize=(8, 8) )
    else:
        plt.figure( figsize=figsize )

    nx.draw_networkx_nodes(G, pos = pos, node_color = list(partition.values()), node_size = node_size, \
            alpha=0.7, cmap=plt.get_cmap("jet"))

    nx.draw_networkx_edges(G, pos = pos, alpha=0.10, arrowsize=15, width=0.55)
    if with_labels:
        nodes = [node for (node, degree) in dict(G.degree).items() if degree > degree_thres]
        labels_printed = {}
        for n in nodes:
            labels_printed[n] = n

        nx.draw_networkx_labels(G, pos=pos, labels=labels_printed)

    plt.axis('off')
    plt.grid(False)
    return pagerank

def draw_community_over_time(resolution, vi, num_comms):
    '''
    Plots `resolution` in the x-axis, `num_comms` on the left y-axis and `vi` on the right y-axis, where
    the inputs are the outputs of `community_resolution`.

    Inputs:
        resolution - Markov time range used for optimising stability of a partition.
        vi         - Variation of information.
        num_comms  - Number of communities.
    '''

    fig, ax1 = plt.subplots( figsize=(10,6) )
    ax2 = ax1.twinx()

    ax2.plot(resolution, vi, label='', color='green', lw=2)
    ax1.plot(resolution, num_comms, label='num', lw=2)

    ax1.set_yscale('log')
    plt.xscale('log')

    ax1.set_xlabel('Markov Time', size=20)
    ax1.set_ylabel('Num. of Communities', color='b', size=16)
    ax2.set_ylabel('Variation of Information', color='g', size=16)

        ## Tracking stable communities...
    seen = {}
    dupes = []
    for (ix,x) in enumerate(num_comms):
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    for (s,a) in seen.items():
        if a > 4:
            print("Plateau of {} time units of {} communities.".format(a,s))
    print('')

def calculate_everything(G, rounds=20, graphs=True, τ=10):
    '''Calculates role-based similarity and Dynamical-embedding similarity (of time `τ=10`) for a graph `G`. It also makes their
    relaxed minimum spanning tree, and looks for an optimal partition over a range of Markov times averaged over `rounds` rounds.'''

    tic = time()

    X_RBS = RBS(G, α=0.95, K=50)
    X_DES = DES(G, τ)

    Y_RBS = CosineSimilarity(X_RBS)
    Y_DES = CosineSimilarity(X_DES)

    γ_RBS = 0.2*np.mean(1 - Y_RBS)
    γ_DES = 0.2*np.mean(1 - Y_DES)

    Z_RBS = nx.Graph( 1 - Y_RBS )
    R_RBS, E_RBS = RMST(Z_RBS, γ=γ_RBS )

    Z_DES = nx.Graph( 1 - Y_DES )
    R_DES, E_DES = RMST(Z_DES, γ=γ_DES )

    print("Calculated all of the RMSTs.")

    if not(nx.is_directed(G)):
        resolution, modularity, num_comms, vi = community_resolution(G, num_rounds=rounds)
    else:
        resolution, modularity, num_comms, vi = ([0], [0], [0], [0])
    resolution_RBS, modularity_RBS, num_comms_RBS, vi_RBS = community_resolution(R_RBS, num_rounds=rounds)
    resolution_DES, modularity_DES, num_comms_DES, vi_DES = community_resolution(R_DES, num_rounds=rounds)

    # Draw subplots...
    if graphs:
        if not(nx.is_directed(G)):
            draw_community_over_time( resolution, vi, num_comms)
            plt.tight_layout()
#             plt.savefig('./media/comm_dynamics_standard.png')

        draw_community_over_time( resolution_RBS, vi_RBS, num_comms_RBS)
        plt.tight_layout()
#         plt.savefig('./media/comm_dynamics_RBS.png')

        draw_community_over_time( resolution_DES, vi_DES, num_comms_DES)
        plt.tight_layout()
#         plt.savefig('./media/comm_dynamics_DES.png')

    # Ask for resolution and plot the others.

    toc = time() - tic
    print("It took {} minutes to do everything".format( round(toc/60, 2) ))

    return Y_RBS, Y_DES, R_RBS, E_RBS, R_DES, E_DES,\
           resolution_RBS, modularity_RBS, num_comms_RBS, vi_RBS, \
           resolution_DES, modularity_DES, num_comms_DES, vi_DES, \
           resolution, modularity, num_comms, vi
