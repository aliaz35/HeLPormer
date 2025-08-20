import pickle
import sys
from . import Data_process as dp
import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
from .data_loader import data_loader
import torch
from torch_sparse import SparseTensor
from typing import List

def get_adjm(edges, num_nodes):
    row, col = zip(*edges)
    row = np.array(row).astype(np.int64)
    col = np.array(col).astype(np.int64)
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    val = torch.ones(len(edges), dtype=torch.float32)
    num_nodes = int(num_nodes)
    adjm = SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))
    return adjm

def load_data(args, prefix='IMDB'):
    dl = data_loader('data/'+prefix)
    features = []

    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    # adjM = sum(dl.links['data'].values())

    train_pos, valid_pos = dl.get_train_valid_pos()

    train_pos_head = np.array([])
    train_pos_tail = np.array([])
    train_neg_head = np.array([])
    train_neg_tail = np.array([])
    r_id = np.array([])
    valid_pos_head = np.array([])
    valid_pos_tail = np.array([])
    valid_neg_head = np.array([])
    valid_neg_tail = np.array([])
    valid_r_id = np.array([])

    train_pos_edges = []
    train_neg_edges = []
    valid_pos_edges = []
    valid_neg_edges = []

    node_cnt: List[int] = [feature.shape[0] for feature in features]
    sum_node: int = 0
    for x in node_cnt:
        sum_node += x
    train_num_nodes = valid_num_nodes = test_num_nodes = sum_node

    for test_edge_type in dl.links_test['data'].keys():
        train_neg = dl.get_train_neg(edge_types=[test_edge_type])[test_edge_type]
        train_pos_head = np.concatenate([train_pos_head, np.array(train_pos[test_edge_type][0])])
        train_pos_tail = np.concatenate([train_pos_tail, np.array(train_pos[test_edge_type][1])])
        train_neg_head = np.concatenate([train_neg_head, np.array(train_neg[0])])
        train_neg_tail = np.concatenate([train_neg_tail, np.array(train_neg[1])])
        r_id = np.concatenate([r_id, np.array([test_edge_type] * len(train_pos[test_edge_type][0]))])
        valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[test_edge_type]
        valid_pos_head = np.concatenate([valid_pos_head, np.array(valid_pos[test_edge_type][0])])
        valid_pos_tail = np.concatenate([valid_pos_tail, np.array(valid_pos[test_edge_type][1])])
        valid_neg_head = np.concatenate([valid_neg_head, np.array(valid_neg[0])])
        valid_neg_tail = np.concatenate([valid_neg_tail, np.array(valid_neg[1])])
        valid_r_id = np.concatenate([valid_r_id, np.array([test_edge_type] * len(valid_pos[test_edge_type][0]))])
        with torch.no_grad():
            for src, dst in zip(train_pos_head, train_pos_tail):
                train_pos_edges.append((src, dst))
                # train_num_nodes = max(train_num_nodes, src, dst)
            for src, dst in zip(train_neg_head, train_neg_tail):
                train_neg_edges.append((src, dst))
                # train_num_nodes = max(train_num_nodes, src, dst)
            for src, dst in zip(valid_pos_head, valid_pos_tail):
                valid_pos_edges.append((src, dst))
                # valid_num_nodes = max(valid_num_nodes, src, dst)
            for src, dst in zip(valid_neg_head, valid_neg_tail):
                valid_neg_edges.append((src, dst))
                # valid_num_nodes = max(valid_num_nodes, src, dst)
    # train_num_nodes = train_num_nodes + 1
    # valid_num_nodes = valid_num_nodes + 1
    neigh_2hop, label_2hop = dl.get_test_neigh()
    test_pos_edges_2hop = []
    test_neg_edges_2hop = []
    for test_edge_type in dl.links_test['data'].keys():
        with torch.no_grad():
            current_neigh = neigh_2hop[test_edge_type]
            current_label = label_2hop[test_edge_type]
            head_nodes, tail_nodes = current_neigh
            for src, dst, label in zip(head_nodes, tail_nodes, current_label):
                if label == 1:
                    test_pos_edges_2hop.append((src, dst))
                elif label == 0:
                    test_neg_edges_2hop.append((src, dst))
                # test_num_nodes = max(test_num_nodes, src, dst)
    neigh_random, label_random = dl.get_test_neigh_w_random()
    test_pos_edges_random = []
    test_neg_edges_random = []
    for test_edge_type in dl.links_test['data'].keys():
        with torch.no_grad():
            current_neigh = neigh_random[test_edge_type]
            current_label = label_random[test_edge_type]
            head_nodes, tail_nodes = current_neigh
            for src, dst, label in zip(head_nodes, tail_nodes, current_label):
                if label == 1:
                    test_pos_edges_random.append((src, dst))
                elif label == 0:
                    test_neg_edges_random.append((src, dst))
                # test_num_nodes = max(test_num_nodes, src, dst)
    # test_num_nodes = test_num_nodes + 1

    train = (np.array(train_pos_edges).astype(np.int64), np.array(train_neg_edges).astype(np.int64))
    valid = (np.array(valid_pos_edges).astype(np.int64), np.array(valid_neg_edges).astype(np.int64))
    test_2hop = (np.array(test_pos_edges_2hop).astype(np.int64), np.array(test_neg_edges_2hop).astype(np.int64))
    test_random = (np.array(test_pos_edges_random).astype(np.int64), np.array(test_neg_edges_random).astype(np.int64))
    edges_mat = {'train':train, 'valid':valid, 'test_2hop':test_2hop, 'test_random': test_random}

    train_adjm = get_adjm(train_pos_edges, train_num_nodes)
    valid_adjm = get_adjm(valid_pos_edges, valid_num_nodes)
    test_adjm_2hop = get_adjm(test_pos_edges_2hop, test_num_nodes)
    test_adjm_random = get_adjm(test_pos_edges_random, test_num_nodes)
    adj_mat = {'train':train_adjm, 'valid':valid_adjm, 'test_2hop':test_adjm_2hop, 'test_random':test_adjm_random}

    # adj_mat, edges_mat = dp.mask_test_edges(adjM)
    # test_edge_list_2hop = test_pos_edges_2hop + test_pos_edges_2hop
    # test_edge_list_random = test_pos_edges_random + test_neg_edges_random
    # list_mat = {'2hop':test_edge_list_2hop, 'random':test_edge_list_random}
    return features, train_adjm, edges_mat, dl\
        #, list_mat
           #adjM, \
           #labels,\
           #train_val_test_idx,\
            #dl
