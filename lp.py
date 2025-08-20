import argparse

from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from torch_scatter import scatter_add

import dgl
import dgl.function as fn
from torch_sparse import SparseTensor

def binary_cross_entropy(pos_score: torch.Tensor, neg_score: torch.Tensor):
    pos_loss = -torch.log(pos_score + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_score + 1e-15).mean()
    return pos_loss + neg_loss


def _compute_dot(g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
    with g.local_scope():
        g.ndata["h"] = h.cpu()
        g.apply_edges(fn.u_dot_v("h", "h", "score"))
        return g.edata["score"]

class DotPred(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        return _compute_dot(g, h).to(h.device)

class MultiHopAggregator(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.device = torch.device(args.device)
        self.delta = args.delta
        self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, args.psi_1_dim).double(),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(args.psi_1_dim, 1).double())

        self.f_node = torch.nn.Sequential(torch.nn.Linear(1, args.psi_2_dim).double(),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(args.psi_2_dim, 1).double())

    def forward(self, overlap: coo_matrix, edges: torch.Tensor):
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges)

        col = torch.from_numpy(overlap.col).long().to(self.device)
        values = torch.from_numpy(overlap.data).to(self.device)
        struct = self.f_node(
            scatter_add(
                self.f_edge(values.unsqueeze(-1)), col, dim=0, dim_size=overlap.shape[0])
            .to(self.device)
        ).squeeze()

        src_nodes, dst_nodes = edges.transpose(0, 1)

        overlap_adj = torch.from_numpy(overlap.todense()).cpu()

        src_neighbors = overlap_adj[src_nodes].to(self.device)
        struct_src = src_neighbors * struct

        dst_neighbors = overlap_adj[src_nodes].to(self.device)
        struct_dst = dst_neighbors * struct

        torch.cuda.empty_cache()

        return torch.einsum("ij, ij -> i", struct_src, struct_dst).unsqueeze(-1)


class BatchDotPred(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edges: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges)
        src_nodes, dst_nodes = edges.transpose(0, 1)
        return (torch.einsum("ij, ij -> i", feat[src_nodes], feat[dst_nodes])
                .unsqueeze(1)
                .to(feat.device))


class LinkPred(nn.Module):
    def __init__(
            self,
            args: argparse.Namespace
    ) -> None:
        super().__init__()
        self.dot = BatchDotPred()
        self.aggregator = MultiHopAggregator(args)
        self.psi_3 = nn.Sequential(torch.nn.Linear(1, args.psi_3_dim).double(),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(args.psi_3_dim, 1).double())
        self.alpha = nn.Parameter(torch.FloatTensor([0, 0]))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.alpha, 0)
        self.psi_3.apply(self.weight_reset)
        
    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def forward(
            self,
            edges: torch.Tensor,
            pattern: SparseTensor,
            H: torch.Tensor
    ) -> torch.Tensor:
        h_link = torch.sigmoid(self.dot(edges, H))
        z_link = torch.sigmoid(self.psi_3(self.aggregator(pattern, edges)))
        alpha = torch.softmax(self.alpha, dim=0)
        y_hat = alpha[0] * z_link + alpha[1] * h_link + 1e-15
        return y_hat