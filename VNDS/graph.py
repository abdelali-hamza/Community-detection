import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as NMI
import copy
import random

class Graph():
  def __init__(self, adj_matrix, labels=None, unused_labels=None, epsilon=0.5):
    assert np.all(adj_matrix == adj_matrix.T)
    self.adj_matrix = adj_matrix
    cardV = self.adj_matrix.shape[0]
    #########################
    # Function to generate a random hexadecimal color code
    def random_color():
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        return color

    # Generate random colors
    if unused_labels is None:
      self.unused_labels = list({random_color() for _ in range(cardV)})
      assert len(self.unused_labels) == cardV
    else:
      self.unused_labels = unused_labels
    #########################
    # nodes
    self.V = list(self.adj_matrix.index)
    assert np.all(np.sort(self.V) == np.sort(list(set(self.V))))
    #########################
    # edges
    self.E = [(from_node, to_node) for idx, from_node in enumerate(self.V) for to_node in self.V[:idx+1] if self.adj_matrix.loc[from_node, to_node] > 0]
    #########################
    # dict of pair label: list_nodes
    if labels is None:
      self.labels = {}
      self.labeled_nodes(epsilon)
    else:
      self.labels = labels
      self.used_labels = [label for label in self.labels]

    assert np.all(np.sort(self.used_labels) == np.sort(list(self.labels.keys())))
    #########################
    # design graph with networkx
    self.G = nx.Graph()
    self.G.add_nodes_from([(node, {"color": label}) for label, label_nodes in self.labels.items() for node in label_nodes])
    self.G.add_edges_from([edge + ({'weight': self.adj_matrix.loc[edge[0], edge[1]]},) for edge in self.E])
    #########################
    # Buv matrix
    self.Buv = pd.DataFrame([[self.B(u,v) for v in self.V] for u in self.V], index=self.V, columns=self.V)

  def labeled_nodes(self, epsilon):
    self.used_labels = []
    self.labels = {}

    u = random.choice(self.V)
    label = self.unused_labels.pop()
    self.labels[label] = [u]
    self.used_labels.append(label)

    for v in self.V:
      if v == u:
        continue

      if random.uniform(0, 1) < epsilon:
        label = random.choice(self.used_labels)
      else:
        label = self.unused_labels.pop()
        self.used_labels.append(label)

      if label in self.labels:
        self.labels[label].append(v)
      else:
        self.labels[label] = [v]

  ## weights of node
  def W(self, u):
    return np.sum(self.adj_matrix.loc[u]) if (u,u) not in self.E else np.sum(self.adj_matrix.loc[u]) + self.adj_matrix.loc[u,u]

  ## Buv matrix
  def B(self, u,v):
    return self.adj_matrix.loc[u,v] - (self.W(u) * self.W(v)) / (2 * len(self.E))

  ## modularity function
  def Q(self):
    q = 0
    total_sum_weights_edges = np.sum(np.triu(self.adj_matrix))
    for label, list_nodes in self.labels.items():
      edges_in_classes = [(from_node, to_node) for from_node in list_nodes for to_node in list_nodes if (from_node, to_node) in self.E]
      sum_weights_edges_in_classes = 0
      sum_weights_vertices_in_classes = 0
      for e in edges_in_classes:
        sum_weights_edges_in_classes += self.adj_matrix.loc[e[0], e[1]]
      frac_weights_edges_in_classes = sum_weights_edges_in_classes / total_sum_weights_edges
      for node in list_nodes:
        sum_weights_vertices_in_classes += self.W(node)
      frac_weights_vertices_in_classes = (sum_weights_vertices_in_classes ** 2) / (4 * (total_sum_weights_edges ** 2))

      q += frac_weights_edges_in_classes - frac_weights_vertices_in_classes

    return q

  def get_neighbors_node(self, node):
    neighbor_colors = set()
    for neighbor in self.G.neighbors(node):
        neighbor_colors.add(self.G.nodes[neighbor]['color'])
    return list(neighbor_colors)

  def get_neighbors_cluster(self, cluster):
    nodes_of_cluster = self.labels[cluster]
    neighboring_cluster = set()
    for node in nodes_of_cluster:
      neighboring_cluster.update(self.get_neighbors_node(node))

    neighboring_cluster.discard(cluster)

    return list(neighboring_cluster)

############################################# Neighborhood structures ##########################################################
  def singleton(self, cluster):
    list_nodes = copy.deepcopy(self.labels[cluster])
    random.shuffle(list_nodes)

    self.used_labels.remove(cluster)
    self.unused_labels.append(cluster)
    del self.labels[cluster]

    new_labels = []
    random.shuffle(self.unused_labels)
    for node in list_nodes:
      label = self.unused_labels.pop()
      self.used_labels.append(label)
      self.labels[label] = [node]
      self.G.nodes[node]["color"] = label
      new_labels.append(label)

    return new_labels

  def division(self, cluster):
    list_nodes = copy.deepcopy(self.labels[cluster])

    if len(list_nodes) > 1:
      random.shuffle(list_nodes)
      split_index = len(list_nodes) // 2
      first_part , second_part = list_nodes[:split_index], list_nodes[split_index:]

      self.used_labels.remove(cluster)
      self.unused_labels.append(cluster)
      del self.labels[cluster]
      random.shuffle(self.unused_labels)

      first_label = self.unused_labels.pop()
      self.used_labels.append(first_label)

      self.labels[first_label] = first_part
      for node in first_part:
        self.G.nodes[node]["color"] = first_label

      second_label = self.unused_labels.pop(0)
      self.used_labels.append(second_label)

      self.labels[second_label] = second_part
      for node in second_part:
        self.G.nodes[node]["color"] = second_label

      return [first_label, second_label]

    return [cluster]

  def neighbor(self, cluster, epsilon):
    list_nodes = copy.deepcopy(self.labels[cluster])
    random.shuffle(list_nodes)

    new_labels = []
    neighbors = self.get_neighbors_cluster(cluster)

    for node in list_nodes:
      if random.uniform(0, 1) < epsilon and len(self.unused_labels) > 0:
        self.labels[cluster].remove(node)
        label = self.unused_labels.pop(0)
        self.used_labels.append(label)
        self.labels[label] = [node]
        self.G.nodes[node]["color"] = label
        new_labels.append(label)
      elif len(neighbors) > 0:
        self.labels[cluster].remove(node)
        label = random.choice(neighbors)
        self.labels[label].append(node)
        self.G.nodes[node]["color"] = label

    if len(self.labels[cluster]) == 0:
      self.used_labels.remove(cluster)
      self.unused_labels.append(cluster)
      del self.labels[cluster]

    return list(set(new_labels))

  def fusion(self, cluster):
    clusters = self.get_neighbors_cluster(cluster)
    random.shuffle(clusters)
    if len(clusters) >= 1:
      n = random.randint(1, len(clusters))
      to_merge = random.sample(clusters, k=n)
      to_merge.append(cluster)

      list_nodes = []
      for module in to_merge:
        list_nodes += self.labels[module]
        self.used_labels.remove(module)
        self.unused_labels.append(module)
        del self.labels[module]

      label = self.unused_labels.pop(0)
      self.used_labels.append(label)
      self.labels[label] = list_nodes

      for node in list_nodes:
        self.G.nodes[node]["color"] = label

      return to_merge

    return None

  def redistribution(self, cluster):
    list_nodes = copy.deepcopy(self.labels[cluster])
    random.shuffle(list_nodes)

    new_labels = []
    neighbors = self.get_neighbors_cluster(cluster)

    if (len(neighbors) > 0):
      self.used_labels.remove(cluster)
      self.unused_labels.append(cluster)
      del self.labels[cluster]

      for node in list_nodes:
          label = random.choice(neighbors)
          self.labels[label].append(node)
          self.G.nodes[node]["color"] = label
          new_labels.append(label)

    return list(set(new_labels))

  def sub_graph(self, list_labels):
    sub_labels = {}
    list_nodes = []

    for label in list_labels:
      list_nodes_in_label = copy.deepcopy(self.labels[label])
      sub_labels[label] = list_nodes_in_label
      list_nodes += list_nodes_in_label

    sub_graph = copy.deepcopy(self.adj_matrix.loc[list_nodes, :].loc[:, list_nodes])
    return Graph(sub_graph, sub_labels, self.unused_labels)

## display network
  def draw(self, title="Graph with Nodes Colored by Class", figsize=(10, 8), seed=420, node_size=300, font_size=8):

    node_colors = {node: self.G.nodes[node]['color'] for node in self.G.nodes()}

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(self.G, seed=seed)  # positions for all nodes
    nx.draw(self.G, pos, with_labels=True, node_size=node_size, node_color=node_colors.values(), font_size=font_size, font_color='white',
            font_weight='bold')
    plt.title(title)
    plt.show()
