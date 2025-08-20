import dgl
import random
import torch
import numpy as np
import argparse
from collections import deque

def sp_to_spt(mat: list[tuple[int, int]]) -> torch.sparse.Tensor:
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def mat_to_tensor(mat: list[tuple[int, int]]) -> torch.sparse.Tensor:
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def edge_to_g(edges: list[tuple[int, int]], num_nodes: int) -> dgl.graph:
    src, dst = zip(*edges)
    src = torch.tensor(src, dtype=torch.int64)
    dst = torch.tensor(dst, dtype=torch.int64)
    return dgl.graph((src, dst), num_nodes=num_nodes).remove_self_loop()

def bfs(start: int, g: dgl.DGLGraph, length: int) -> list[int]:
    ret = [start]
    hash_table = {start}
    q = deque([start])
    while len(q) > 0:
        successors = [s for s in g.successors(q.popleft()).tolist() if not s in hash_table]
        random.shuffle(successors)
        for s in successors:
            q.append(s)
            ret.append(s)
            hash_table.add(s)
            if len(ret) == length:
                return ret
    return ret + [ret[-1]] * (length - len(ret))

def graph_tokenize(g: dgl.DGLGraph, length: int) -> torch.Tensor:
    return torch.tensor([bfs(node, g, length) for node in range(g.num_nodes())])

def sample_seq(g: dgl.DGLGraph, node_cnt: int, seq_length: int) -> torch.Tensor:
    all_nodes = np.arange(node_cnt)
    node_seq = torch.zeros(node_cnt, seq_length).long()
    successors_cache = {node : list(g.successors(node)) for node in range(node_cnt)}
    n = 0
    for x in all_nodes:
        cnt = 0
        scnt = 0
        node_seq[n, cnt] = x
        cnt += 1
        start = node_seq[n, scnt].item()
        hash_table = set()
        while cnt < seq_length:
            successors = [s for s in successors_cache[start] if not s in hash_table and not hash_table.add(s)]
            sample_list = random.sample(successors, len(successors))
            if not sample_list and scnt == cnt - 1:
                node_seq[n][cnt:] = start
                break
            for node in sample_list:
                node_seq[n, cnt] = node
                cnt += 1
                if cnt == seq_length:
                    break
            scnt += 1
            start = node_seq[n, scnt].item()
        n += 1

    return node_seq

def get_subgraph(sg, features_list, node_cnt, seq, node_type):
    node_cnt_type = [sum(node_cnt[:i+1]) for i in range(len(node_cnt))]
    nodes = sg.nodes()
    ori_node_id = np.array(sg.ndata[dgl.NID][nodes.long()].cpu())
    new_feature_list = [[] for _ in range(len(node_cnt))]

    seq_nodes = []
    cnt = 0

    for node in ori_node_id:
        index = sum(1 for x in node_cnt_type if x <= node)
        if index == 0:
            new_feature = features_list[index][node]
            if node < seq.shape[0]:
                seq_nodes.append(cnt)
        else:
            new_feature = features_list[index][node - node_cnt_type[index-1]]
        new_feature_list[index].append(new_feature)
        cnt += 1
    for i in range(len(new_feature_list)):
        if new_feature_list[i]:
            new_feature_list[i] = torch.stack(new_feature_list[i])
        else:
            device = features_list[i][0].device
            new_feature_list[i] = torch.empty((0,) + features_list[i][0].size(), dtype=features_list[i][0].dtype, device=device)
    # sg_node_type = node_type[ori_node_id]

    return new_feature_list, torch.tensor(seq_nodes).long().unsqueeze(1)# , sg_node_type

def prompt_parse() -> argparse.Namespace:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(
        description="HINormer")
    #args for HINormer
    ap.add_argument("--feats-type",         type=int,   default=3,
                    help="Type of the node features used. " +
                         "0 - loaded features; " +
                         "1 - only target node features (zero vec for others); " +
                         "2 - only target node features (id vec for others); " +
                         "3 - all id vec. Default is 2" +
                         "4 - only term features (id vec for others);" +
                         "5 - only term features (zero vec for others).")
    ap.add_argument("--device",             type=str,   default="cuda:0")
    ap.add_argument("--hidden-dim",         type=int,   default=256,
                    help="Dimension of the node hidden state. Default is 32.")
    ap.add_argument("--dataset",            type=str,   default = "amazon",
                    help="DBLP, IMDB, Freebase, AMiner, DBLP-HGB, IMDB-HGB")
    ap.add_argument("--num-heads",          type=int,   default=2,
                    help="Number of the attention heads. Default is 2.")
    ap.add_argument("--epoch",              type=int,   default=1000,
                    help="Number of epochs.")
    ap.add_argument("--patience",           type=int,   default=20,
                    help="Patience.")
    ap.add_argument("--repeat",             type=int,   default=1,
                    help="Repeat the training and testing for N times. Default is 1.")
    ap.add_argument("--slope",              type=float, default=0.01,
                    help="The negative slope of leaky ReLU. Default is 0.2.")
    ap.add_argument("--L",                  type=int,   default=2,
                    help="The number of layers of Graph Transformer layer")
    ap.add_argument("--K",                  type=int,   default=4,
                    help="The number of layers of GNN, both node feature encoder and heterogeneity encoder")
    ap.add_argument("--lr",                 type=float, default=0.01,
                    help="learning rate")
    ap.add_argument("--seed",               type=int,   default=2024,
                    help="random seed to initialize tensors")
    ap.add_argument("--dropout",            type=float, default=0,
                    help="the p argument to initialize torch.nn.Dropout")#
    ap.add_argument("--weight-decay",       type=float, default=0,
                    help="weight_decay argument of the optimizer")#
    ap.add_argument("--len-seq",            type=int,   default=50,
                    help="The length of node sequence.")#
    ap.add_argument("--mode",               type=int,   default=0,
                    help="Output mode, 0 for offline evaluation and 1 for online HGB evaluation")
    ap.add_argument("--temperature",        type=float, default=1.0,
                    help="Temperature of attention score")#

    ap.add_argument("--beta",      type=float, default=1.0,
                    help="Weight of heterogeneity-level attention score")#
    ap.add_argument("--delta",        type=float, default=1.0,
                    help="Hyper-parameter of calculating multi-hop connectivity pattern")
    ap.add_argument("--gamma",      type=int, default=2,
                    help="Count of hops in heterogeneous multi-hop structure aggregation")

    ap.add_argument('--psi-1-dim', type=int, default=8)
    ap.add_argument('--psi-2-dim', type=int, default=128)
    ap.add_argument('--psi-3-dim', type=int, default=128)


    ap.add_argument("--batch-size",         type=int, default=10240,
                    help="size of mini batch")#
    return ap.parse_args()
