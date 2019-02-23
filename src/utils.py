import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(graph.selfloop_edges())
    return graph

def feature_reader(path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    features = json.load(open(path))
    index_1 = [int(k) for k,v in features.items() for fet in v]
    index_2 = [int(fet) for k,v in features.items() for fet in v]
    values = [1.0]*len(index_1) 
    nodes = [int(k) for k,v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.coo_matrix((values,(index_1,index_2)), shape=(node_count, feature_count),dtype=np.float32)
    out_features = dict()
    out_features["indices"] = torch.LongTensor(np.concatenate([features.row.reshape(-1,1), features.col.reshape(-1,1)],axis=1).T)
    out_features["values"] = torch.FloatTensor(features.data)
    out_features["dimensions"] = features.shape
    return out_features


def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = torch.LongTensor(np.array(pd.read_csv(path)["target"]))
    return target

def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()]
    values = [1 for edge in graph.edges()]
    node_count = max(max(index_1)+1,max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1,index_2)),shape=(node_count,node_count),dtype=np.float32)
    return A

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(graph):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values. 
    """
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
    propagator["indices"] = torch.LongTensor(np.concatenate([A_tilde_hat.row.reshape(-1,1), A_tilde_hat.col.reshape(-1,1)],axis=1).T)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator
