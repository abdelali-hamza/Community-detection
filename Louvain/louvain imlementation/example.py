#!/usr/bin/env python3

import networkx as nx

from louvain import Louvain
from collections import defaultdict

if __name__ == "__main__":

    G = nx.Graph()
    edges = [(1,2),(1,3),(1,4),(2,4),(3,4),(4,5),(5,6),(5,7),(6,7),(7,8),(4,8),(4,12),(10,12),(10,8),(9,10),(8,9),(11,9),(11,10),(11,12)]
    G.add_weighted_edges_from([(i,j,1.0) for i,j in edges])

    sample_graph = G
    louvain = Louvain()
    partition = louvain.getBestPartition(sample_graph, verbose=True, plot=False)

    p = defaultdict(list)
    for node, com_id in partition.items():
        p[com_id].append(node)

    for com, nodes in p.items():
        print(com, nodes)

    print(list(p.values()))
