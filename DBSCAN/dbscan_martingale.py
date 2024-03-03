#import libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import rand_score, normalized_mutual_info_score
import random
import pandas as pd

class DBSCAN_Martingale():

  # ================================================== Neigborhood function ==================================================
  def neighborhood_ordre1(self, graph, eps):
    "Function to return neighborhoods of every node in the network based on the density level epsilon"
    nodes = list(graph.nodes)
    shortest_path = {node: [] for node in nodes}
    for node in nodes:
      neighbors = list(graph.neighbors(node))
      shortest_path[node] = neighbors

    return shortest_path

  # ============================================== DBSCAN inner function (used by DBSCAN* to label nodes)  ===========================================
  def dbscan_inner(self, is_core, neighborhoods, labels):
    label_num = 1
    stack = []
    nodes = list(range(labels.shape[0]))
    #random.shuffle(nodes)
    for i in nodes:
      if labels[i] != 0 or not is_core[i]:
          continue

      # Depth-first search starting from i, ending at the non-core points.
      # This is very similar to the classic algorithm for computing connected
      # components, the difference being that we label non-core points as
      # part of a cluster (component), but don't expand their neighborhoods.
      while True:
        if labels[i] == 0:
          labels[i] = label_num
          if is_core[i]:
            neighb = neighborhoods[i]
            for v in neighb:
              if labels[v] == 0:
                stack.append(v)

        if len(stack) == 0:
            break
        i = stack.pop()

      label_num += 1

  # ===================================================== DBSCAN* density-based clustering algorithm  =================================================
  def dbscan_star(self, graph, eps, min_points):
    neighborhoods = self.neighborhood_ordre1(graph, eps)

    # Core points
    core_samples = np.array([1 if len(neighborhoods[node]) >= min_points else 0 for node in range(len(graph.nodes))])

    # Vector of labels C initialized to 0
    C = np.full(len(graph.nodes), 0, np.intp)

    # Apply DBSCAN inner on C vector
    self.dbscan_inner(core_samples,neighborhoods, C)

    return C

  # ====================================================== Propagation function ======================================================
  def propagate(self, graph, eps, min_points, labels):
    # Get neighborhood of each node
    neighborhoods = self.neighborhood_ordre1(graph, eps)

    # Calculate the number of points in the nighborhood
    n_neighbors = np.array([len(neighborhoods[neighbors]) for neighbors in range(len(graph.nodes))])

    # extract all clusters in the graph
    clusters = np.unique(labels [labels!=0])

    for cluster in clusters:

      # get the nodes of the current cluster
      nodes_in_cluster = list(np.where((labels == cluster))[0])
      L = nodes_in_cluster

      # Propagate labels within the cluster
      while len(L) > 0:
        node = L.pop()

        # Get labels of neighboring nodes
        neighbors = labels[neighborhoods[node]]

        # Filter unassigned nodes
        k = np.array(neighborhoods[node])
        k = k[neighbors == 0]

        # Assign unassigned nodes to the current cluster
        neighbors[neighbors == 0] = cluster

        # Update C vector for the neighborhood
        labels[neighborhoods[node]] = neighbors

        # Add newly assigned nodes to the propagation queue
        L.extend(list(k))

  # ================================================ DBSCAN* with Martingale process algorithm =================================================
  def fit_predict(self, graph, eps, MinPts, S, realizations, ground_truth):
    print("Starting execution")

    # Matrix to store the number of clusters for each MinPts in each realization
    realizations_DBSCAN_martingale = np.zeros((realizations, S))

    # Final result of C vector for each realization
    principal_clustering_all = np.zeros((realizations, len(graph.nodes)))

    # Lists to store normalized mutual information (NMI) and adjusted Rand index (ARI) for each realization
    nmis = []
    rands = []

    for r in range(realizations):
        # List to store the number of clusters for each MinPts in the current realization
        number_of_clusters = []

        # Randomly sample S MinPts values
        random_min_points = random.sample(MinPts, k=S)

        # Sort MinPts values in descending order
        random_min_points = np.array(sorted(random_min_points, key=lambda x: -x))

        # Matrix to store DBSCAN* results for each MinPts in the current realization
        dbscan_results_all = np.zeros((len(graph.nodes), S), dtype=int)

        # Run DBSCAN* for each MinPts value
        for j in range(S):
            dbscan_results_all[:, j] = self.dbscan_star(graph, eps, min_points=random_min_points[j])

        # ===================== Begining of the martingale process =====================

        # Initialize principal clustering using the results of the first iteration of DBSCAN*
        principal_clustering = dbscan_results_all[:, 0]

        for j in range(S):
            # Merge clusters based on common elements in principal_clustering and dbscan_results_all[:, j]
            if np.dot(principal_clustering, dbscan_results_all[:, j]) == 0:
                b = np.max(principal_clustering)
                for i in range(len(dbscan_results_all[:, j])):
                    if dbscan_results_all[i, j] != 0:
                        dbscan_results_all[i, j] += b
                principal_clustering = principal_clustering + dbscan_results_all[:, j]
            else:
                # If there are common elements, identify unassigned nodes in principal_clustering
                h = np.zeros(len(principal_clustering))
                clh = np.zeros(len(principal_clustering))
                for i in range(len(principal_clustering)):
                    if principal_clustering[i] == 0 and dbscan_results_all[i, j] != 0:
                        h[i] = dbscan_results_all[i, j]
                b = np.max(principal_clustering)
                u = 0

                # Check for new labeled elements and assign them to new communities
                if np.max(h) > 0:
                    # set the new labeled elements community to r (r number of new detected communities)
                    for k in range(1, int(np.max(h)) + 1):
                        if np.sum(h == k) >= random_min_points[j]:
                            u += 1
                            clh[h == k] = u
                    # add max of C to each new detected community
                    for i in range(len(principal_clustering)):
                        if clh[i] != 0:
                            clh[i] += b
                    principal_clustering = principal_clustering + clh

            # Record the number of clusters for the current min point
            number_of_clusters.append(np.max(principal_clustering))

        # list of C(community of each node) at each realization
        principal_clustering = np.array(principal_clustering)

        # Propagate labels for unassigned nodes in each cluster
        self.propagate(graph, eps, random_min_points[-1], principal_clustering)

        # Store principal clustering for the current realization (C vector)
        principal_clustering_all[r, :] = principal_clustering

        # Evaluate clustering performance using NMI and RAND
        ari = rand_score(ground_truth, principal_clustering)
        nmi = normalized_mutual_info_score(ground_truth, principal_clustering)

        nmis.append(nmi)
        rands.append(ari)

        # Store the number of clusters for the current realization
        realizations_DBSCAN_martingale[r, :] = number_of_clusters

    print("Execution finished")

    df_stat = pd.DataFrame({
      "C": realizations_DBSCAN_martingale[:, -1],
      "NMI": nmis,
      "RAND": rands
    })

    # Number of clusters probability
    n_clusters, cluster_counts = np.unique(realizations_DBSCAN_martingale[:, -1], return_counts=True)
    arg_n_clusters = np.argsort(n_clusters)
    n_clusters = n_clusters[arg_n_clusters]
    cluster_counts = cluster_counts[arg_n_clusters]
    cluster_probabilities = cluster_counts / (realizations)
    cluster_probabilities[n_clusters ==0 ] = 0

    return principal_clustering_all, df_stat, n_clusters, cluster_probabilities
