#import libraries
import networkx as nx
import matplotlib.pyplot as plt
import random

# function to read LFR graphs
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

# function to plot the network
def plot(G, node_size = 50, title=""):
  plt.figure(figsize=(10, 8))
  #nx.draw(G, pos=nx.spring_layout(G), node_size=node_size, font_color="white", font_size=10, width= 0.5)
  nx.draw(
        G,
        pos=nx.spring_layout(G),
        node_size=node_size,
        font_color="white",
        font_size=10,
        width=0.5,
        node_color='white',  # Set node color to white
        edgecolors='black',  # Set edge color to black
    )
  plt.title(title)
  plt.show()

# Function used to calculate modularity
def group_indices(partitioning):
    clusters = {}

    for i, value in enumerate(partitioning):
        if value not in clusters:
            clusters[value] = [i]
        else:
            clusters[value].append(i)

    result = [set(indices) for indices in clusters.values()]
    return result

def random_color():
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        return color