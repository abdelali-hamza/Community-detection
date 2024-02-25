
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as NMI
import copy
import random
from graph import Graph



class VNDS():
  def __init__(self, alpha=[0.3, 0.3 , 0.1 , 0.2, 0.1]):
    self.alpha = alpha
    self.pertubation = ["singleton", "division","neighbor", "fusion", "redistribution"]

  ## LPAm function for local search
  def LPAm(self, graph, draw=False):

    L = copy.deepcopy(graph.used_labels)
    random.shuffle(L)

    while len(L) > 0:
      label = L.pop(0)
      list_nodes_in_labels = copy.deepcopy(graph.labels[label])

      for node in list_nodes_in_labels:
        lold = graph.G.nodes[node]['color']
        lnew = lold
        b = -np.inf
        adj_classes = graph.get_neighbors_node(node)
        for c in adj_classes:
          list_nodes = graph.labels[c]
          if len(list_nodes) == 0:
            continue
          if b < np.sum(graph.Buv.loc[node, list_nodes]):
            b = np.sum(graph.Buv.loc[node, list_nodes])
            lnew = c

        if lold != lnew:
          qold = graph.Q()
          graph.G.nodes[node]['color'] = lnew
          graph.labels[lold].remove(node)
          graph.labels[lnew].append(node)
          qnew = graph.Q()

          if qold < qnew:
            L.extend([lold, lnew] + [adj for adj in graph.get_neighbors_node(node) if ((adj != lold) and (adj != lnew))])

            if len(graph.labels[lold]) == 0:
              graph.used_labels.remove(lold)
              graph.unused_labels.append(lold)

            if draw == True:
                graph.draw()
          else:
            graph.G.nodes[node]['color'] = lold
            graph.labels[lnew].remove(node)
            graph.labels[lold].append(node)

    keys = list(graph.labels.keys())
    for label in keys:
      if len(graph.labels[label]) == 0:
        del graph.labels[label]

    return graph

## verify possibility of merging two communities
  def merge(self, graph, labels_to_merge):

    qold = graph.Q()
    idx = -1
    labels_original = copy.deepcopy(graph.labels)

    for i, (t1, t2) in enumerate(labels_to_merge):
      labels_copy = copy.deepcopy(labels_original)
      labels_copy[t1] = labels_copy[t1] + labels_copy[t2]
      del labels_copy[t2]
      graph.labels = labels_copy
      qnew = graph.Q()
      graph.labels = copy.deepcopy(labels_original)
      if (qnew > qold):
        idx = i

    return idx != -1

## calculate difference of modularity before and after merging
  def deltaQ(self,graph,  t1, t2): # classes

    if (len(graph.labels[t1]) == 0) or (len(graph.labels[t2]) == 0):
      return 0

    qold = graph.Q()
    labels_original = copy.deepcopy(graph.labels)

    labels_copy = copy.deepcopy(labels_original)
    labels_copy[t1] = labels_copy[t1] + labels_copy[t2]
    del labels_copy[t2]
    graph.labels = labels_copy
    qnew = graph.Q()
    graph.labels = copy.deepcopy(labels_original)

    return qnew - qold

  ## Ensure that the merging of the two communities is the most suitable option.
  def best_merge(self, graph, t1, t2): # verfy

    deltaQ_t1t2 = self.deltaQ(graph, t1, t2)
    classes = copy.deepcopy(graph.used_labels)
    classes.remove(t1)
    classes.remove(t2)

    possible_merging = []

    if (len(graph.labels[t1]) > 0):
      possible_merging += [(t1, t) for t in classes if ((len(graph.labels[t]) > 0) and (t != t1) and (t != t2))]
    if (len(graph.labels[t2]) > 0):
      possible_merging += [(t2, t) for t in classes if ((len(graph.labels[t]) > 0) and (t != t1) and (t != t2))]

    for t in possible_merging:
      if self.deltaQ(graph, *t) > deltaQ_t1t2:
        return False

    return True

  ## LPAm + algorithm for local search
  def LPAm_plus(self, graph, draw=False):
    graph = self.LPAm(graph, draw)

    possible_merging = [(t1, t2) for idx, t1 in enumerate(graph.used_labels) for t2 in graph.used_labels[:idx]]

    count = 0
    while(self.merge(graph, possible_merging)):
      for t1, t2 in possible_merging:
        if ((self.deltaQ(graph, t1, t2) > 0) and (self.best_merge(graph, t1, t2))):
          if (len(graph.labels[t1]) == 0) or (len(graph.labels[t2]) == 0):
            continue

          graph.labels[t1] = graph.labels[t1] + graph.labels[t2]
          for node in graph.labels[t2]:
            graph.G.nodes[node]['color'] = t1

          graph.labels[t2] = []

          graph.used_labels.remove(t2)
          graph.unused_labels.append(t2)

          if draw == True:
              graph.draw()


      graph = self.LPAm(graph, draw)
      possible_merging = [(t1, t2) for idx, t1 in enumerate(graph.used_labels) for t2 in graph.used_labels[:idx]]

    return graph

  ## select a random community and its neighbors
  def select(self, graph, s):
    random_selected_cluster = random.choice(graph.used_labels)
    neighboring_cluster = graph.get_neighbors_cluster(random_selected_cluster)
    if len(neighboring_cluster) > 0 and s < len(neighboring_cluster)+1:
        sub_neighboring_cluster = random.sample(neighboring_cluster, k=s-1)
    elif s == len(neighboring_cluster)+1:
        sub_neighboring_cluster = neighboring_cluster
    else:
        sub_neighboring_cluster = []

    return random_selected_cluster, sub_neighboring_cluster

  ## change the current solution by applying the shaking on the community choosing
  def shaking(self, graph, cluster):
    pertubation = np.random.choice(self.pertubation, p=self.alpha)

    if pertubation == "singleton":
      graph.singleton(cluster)
    elif pertubation == "division":
      graph.division(cluster)
    elif pertubation == "fusion":
      graph.fusion(cluster)
    elif pertubation == "redistribution":
      graph.redistribution(cluster)
    else:
      graph.neighbor(cluster, 0.4)

    return graph, pertubation

  ## main algorithm VNDS
  def VNDS_main(self, adj_matrix, MAX_SIZE=15, stoping_condition=4):
    x = Graph(adj_matrix, epsilon=0)
    x.draw(title="Initial random solution")
    print("starting Execution")
    x = self.LPAm_plus(copy.deepcopy(x))
    non_improvement_modularity_iteration = 0
    s = 1
    count = 0
    while(non_improvement_modularity_iteration < stoping_condition):
      random_selected_cluster, neighboring_clusters = self.select(copy.deepcopy(x), s)
      sub_problem = neighboring_clusters + [random_selected_cluster]
      x_prim = x.sub_graph(copy.deepcopy(sub_problem))

      removed_labels = {label: x.labels[label] for label in x.used_labels if label not in sub_problem}
      x_prim, p = self.shaking(copy.deepcopy(x_prim), random_selected_cluster)
      x_prim = self.LPAm_plus(copy.deepcopy(x_prim))

      labels = {**x_prim.labels, **removed_labels} # update labels
      x_prim = Graph(adj_matrix=copy.deepcopy(x.adj_matrix), labels=copy.deepcopy(labels), unused_labels=copy.deepcopy(x_prim.unused_labels))
      if x.Q() < x_prim.Q():
        x = self.LPAm(copy.deepcopy(x_prim))
        s = 1
      else:
        s = s + 1
        if(s > min(MAX_SIZE, len(x.used_labels))):
          s = 1

        non_improvement_modularity_iteration += 1

      count += 1
    print("Execution finished ")
    return x