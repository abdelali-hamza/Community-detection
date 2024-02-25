import networkx as nx

def load_synth_graph(path):
    # Read network.dat
    G = nx.Graph()
    with open(f'{path}network.dat', 'r') as f:
        for line in f:
            parts = line.split()
            node1 = int(parts[0]) - 1
            node2 = int(parts[1]) - 1
            G.add_edge(node1, node2)

    # Read community.dat
    communities = []
    with open(f'{path}community.dat', 'r') as f:
        for line in f:
            parts = line.split()
            node = int(parts[0]) - 1
            community = int(parts[1])
            communities.append(community)

    return G, communities