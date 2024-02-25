#!/usr/bin/env python3
# coding: utf-8

import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from collections import defaultdict
import time


<<<<<<< HEAD
def draw_communities(G, community_map, node_size=None, alpha=1, k=None, randomized=False, plot_weights=True, verbose=False):
=======
def draw_communities(G, community_map, node_size=5, alpha=1, k=None, randomized=False, plot_weights=True, verbose=False):
>>>>>>> 804bfc3dacabd2ec41aa64839b5a88807decc27f
    if verbose:
        print("Drawing Communities...")
    fig, ax = plt.subplots(figsize=(10, 10))
    node_count = len(G.nodes())

    if node_size is None:
        if node_count < 10:
            node_size = 800
        elif node_count < 20:
            node_size = 600
        elif node_count < 50:
            node_size = 400
        elif node_count < 100:
            node_size = 300
        elif node_count < 200:
            node_size = 200
        elif node_count < 500:
            node_size = 100
        elif node_count < 1000:
            node_size = 20

    cmap = plt.get_cmap("jet")
    pos = nx.spring_layout(G, k=k)
    indexed = [community_map.get(node) for node in G]
    edge_labels = nx.get_edge_attributes(G, 'weight')
    for key in edge_labels.keys():
        edge_labels[key] = int(edge_labels[key])
    ax.axis("off")

    nx.draw_networkx_nodes(G, pos=pos, cmap=cmap, node_color=indexed, node_size=node_size, alpha=alpha, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edgelist=edge_labels, alpha=1., ax=ax)
    if plot_weights:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=10)  # Draw edge labels

    for node in G.nodes():
        x, y = pos[node]
        if G.has_edge(node, node):
            self_loop_weight = G[node][node]['weight']
            ax.annotate(f'{int(self_loop_weight)}', xy=(x, y), xytext=(x, y), ha='center', fontsize=10)

    plt.show()


class Louvain(object):
    def __init__(self):
        self.MIN_VALUE = 0.0000001
        self.node_weights = {}

    @classmethod
    def convertIGraphToNxGraph(cls, igraph):
        node_names = igraph.vs["name"]
        edge_list = igraph.get_edgelist()
        weight_list = igraph.es["weight"]
        node_dict = defaultdict(str)

        for idx, node in enumerate(igraph.vs):
            node_dict[node.index] = node_names[idx]

        convert_list = []
        for idx in range(len(edge_list)):
            edge = edge_list[idx]
            new_edge = (node_dict[edge[0]], node_dict[edge[1]], weight_list[idx])
            convert_list.append(new_edge)

        convert_graph = nx.Graph()
        convert_graph.add_weighted_edges_from(convert_list)
        return convert_graph

    def updateNodeWeights(self, edge_weights):
        node_weights = defaultdict(float)
        for node in edge_weights.keys():
            node_weights[node] = sum([weight for weight in edge_weights[node].values()])
        return node_weights

    def getBestPartition(self, graph, param=1., size=400, verbose=False, plot=True):
        if verbose:
            print("#################Start Louvain############")
        node2com, edge_weights = self._setNode2Com(graph)
        if verbose:
            print("#################Starting pass number: 1############")
        if plot:
            draw_communities(graph, node2com, node_size=size, plot_weights=False, verbose=verbose)
        start = time.time()
        node2com = self._runFirstPhase(node2com, edge_weights, param, verbose=verbose)
        end = time.time()
        partition = node2com.copy()
        com2node = defaultdict(list)
        for node, com_id in node2com.items():
            com2node[com_id].append(node)
        # best_modularity = self.computeModularity(node2com, edge_weights, param)
        best_modularity = nx.community.modularity(graph, com2node.values())
        if plot:
            draw_communities(graph, node2com, node_size=size, plot_weights=False, verbose=verbose)
        new_node2com, new_edge_weights = self._runSecondPhase(node2com, edge_weights)
        new_graph = nx.Graph()
        new_graph.add_weighted_edges_from(
            [(i, j, new_edge_weights[i][j]) for i in new_edge_weights.keys() for j in new_edge_weights[i].keys()])
        pass_num = 2
        history = [(end-start, best_modularity, node2com, edge_weights, graph, partition, pass_num)]
        while True:
            if plot:
                draw_communities(new_graph, new_node2com, node_size=size, verbose=verbose)
            if verbose:
                print('$$$$$$$$$$$$$$$$$starting pass number: ', pass_num, '$$$$$$$$$$$$')
            start = time.time()
            new_node2com = self._runFirstPhase(new_node2com, new_edge_weights, param, verbose=verbose)
            end = time.time()

            com2node = defaultdict(list)
            for node, com_id in new_node2com.items():
                com2node[com_id].append(node)
            
            modularity = nx.community.modularity(new_graph, com2node.values())
            past_partition = partition
            # modularity = self.computeModularity(new_node2com, new_edge_weights, param)
            if verbose:
                print(f'{best_modularity}   to new modularity with value of    {modularity}')
            partition = self._updatePartition(new_node2com, partition)
            history.append((end-start, modularity, new_node2com, new_edge_weights, new_graph, partition, pass_num))
            if plot:
                draw_communities(new_graph, new_node2com, node_size=size, verbose=verbose)
            if abs(best_modularity - modularity) < self.MIN_VALUE:
                partition = past_partition
                break
            best_modularity = modularity
            _new_node2com, _new_edge_weights = self._runSecondPhase(new_node2com, new_edge_weights)
            new_graph = nx.Graph()
            new_graph.add_weighted_edges_from(
                [(i, j, _new_edge_weights[i][j]) for i in _new_edge_weights.keys() for j in
                 _new_edge_weights[i].keys()])
            new_node2com = _new_node2com
            new_edge_weights = _new_edge_weights
            pass_num += 1
        return partition, history

    def computeModularity(self, node2com, edge_weights, param):
        q = 0
        all_edge_weights = sum([weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2

        com2node = defaultdict(list)
        for node, com_id in node2com.items():
            com2node[com_id].append(node)

        for com_id, nodes in com2node.items():
            node_combinations = list(combinations(nodes, 2)) + [(node, node) for node in nodes]
            cluster_weight = sum([edge_weights[node_pair[0]][node_pair[1]] for node_pair in node_combinations])
            tot = self.getDegreeOfCluster(nodes, node2com, edge_weights)
            q += (cluster_weight / (2 * all_edge_weights)) - param * ((tot / (2 * all_edge_weights)) ** 2)
        return q

    def getDegreeOfCluster(self, nodes, node2com, edge_weights):
        weight = sum([sum(list(edge_weights[n].values())) for n in nodes])
        return weight

    def _updatePartition(self, new_node2com, partition):
        reverse_partition = defaultdict(list)
        for node, com_id in partition.items():
            reverse_partition[com_id].append(node)

        for old_com_id, new_com_id in new_node2com.items():
            for old_com in reverse_partition[old_com_id]:
                partition[old_com] = new_com_id
        return partition

    def _runFirstPhase(self, node2com, edge_weights, param, verbose=False):
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2
        self.node_weights = self.updateNodeWeights(edge_weights)
        status = True
        if verbose:
            print("#################Start First Phase############")
        iteration_num = 1
        while status:
            if verbose:
                print(f'\tIteration {iteration_num}')
            statuses = []
            for node in node2com.keys():
                if verbose:
                    print("\t\tNode:", node, "Community:", node2com[node])
                com_id = node2com[node]
                neigh_nodes = [edge[0] for edge in self.getNeighborNodes(node, edge_weights)]

                max_delta = 0.
                max_com_id = com_id
                communities = {}
                for neigh_node in neigh_nodes:
                    node2com_copy = node2com.copy()
                    if node2com_copy[neigh_node] in communities:
                        continue
                    communities[node2com_copy[neigh_node]] = 1
                    node2com_copy[node] = node2com_copy[neigh_node]

                    delta_q = self.getNodeWeightInCluster(node, node2com_copy, edge_weights) / all_edge_weights - \
                              self.getTotWeight(node, node2com_copy, edge_weights) * self.node_weights[node] / (
                                          2 * all_edge_weights ** 2)
                    if verbose:
                        print(f'\t\t\tdelta_Q with node {neigh_node} is {delta_q}')
                    if delta_q > max_delta:
                        max_delta = delta_q
                        max_com_id = node2com_copy[neigh_node]

                node2com[node] = max_com_id
                statuses.append(com_id != max_com_id)
                if com_id != max_com_id and verbose:
                    print("\t\t\t\tNode:", node, "Gain:", max_delta, "assigned to Community:", max_com_id, '!!!!!!!')
                elif verbose:
                    print("\t\t\t\tNode:", node, "Gain:", max_delta, "No change in community")
            iteration_num += 1
            if sum(statuses) == 0:
                break
        return node2com

    def _runSecondPhase(self, node2com, edge_weights, verbose=False):
        if verbose:
            print("#################Start Second Phase############")
        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda: defaultdict(float))

        for node, com_id in node2com.items():
            com2node[com_id].append(node)
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        nodes = list(node2com.keys())
        node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
        for edge in node_pairs:
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][
                edge[1]]
        to_del = []
        for node, weight_list in new_edge_weights.items():
            for n, weight in weight_list.items():
                if weight == 0:
                    to_del.append((node, n))
        for edge in to_del:
            del new_edge_weights[edge[0]][edge[1]]

        return new_node2com, new_edge_weights

    def getTotWeight(self, node, node2com, edge_weights):
        nodes = [n for n, com_id in node2com.items() if com_id == node2com[node] and node != n]

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    def getNeighborNodes(self, node, edge_weights):
        if node not in edge_weights:
            return 0
        return edge_weights[node].items()

    def getNodeWeightInCluster(self, node, node2com, edge_weights):
        neigh_nodes = self.getNeighborNodes(node, edge_weights)
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]
        return weights

    def _setNode2Com(self, graph):
        node2com = {}
        edge_weights = defaultdict(lambda: defaultdict(float))
        for idx, node in enumerate(graph.nodes()):
            node2com[node] = idx
            for edge in graph[node].items():
                edge_weights[node][edge[0]] = edge[1]["weight"]
        return node2com, edge_weights
