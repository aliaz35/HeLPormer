import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import  GraphConv

import torch as th
from dgl import function as fn
from dgl._ffi.base import DGLError
from torch.nn import init

def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.max(x.norm(dim=1, keepdim=True), torch.tensor(1e-12))

class REConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 num_type=4,
                 weight=True,
                 bias=True,
                 activation=None):
        super(REConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.weight_type = nn.Parameter(th.ones(num_type))

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, type_info):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5).to(feat.device)
                else:
                    norm = (1.0 / degs).to(feat.device)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                feat = feat * norm

            feat = th.matmul(feat, self.weight)
            graph.srcdata['h'] = feat * self.weight_type[type_info].reshape(-1, 1)
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h'].to(feat.device)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5).to(feat.device)
                else:
                    norm = (1.0 / degs).to(feat.device)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class AGTLayer(nn.Module):
    def __init__(
            self,
            embeddings_dimension,
            nheads=2,
            att_dropout=0.5,
            emb_dropout=0.5,
            temper=1.0,
            rl=False,
            rl_dim=4,
            beta=1,
            slope=0.2
    ):

        super(AGTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension

        self.head_dim = self.embeddings_dimension // self.nheads


        self.leaky = nn.LeakyReLU(slope)

        self.temper = temper

        self.rl_dim = rl_dim

        self.beta = beta

        self.linear_l = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_r = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)

        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)

       
        if rl:
            self.r_source = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)
            self.r_target = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.LN = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, rh=None):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)

        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)

        if rh is not None:
            r_k = self.r_source(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).transpose(1,2)
            r_q = self.r_target(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).permute(0, 2, 3, 1)
            score_r = r_k @ r_q
            score = score + self.beta * score_r

        score = score / self.temper

        score = F.softmax(score, dim=-1)
        score = self.dropout1(score)

        context = score @ fr

        h_sa = context.transpose(1,2).reshape(batch_size, -1, self.head_dim * self.nheads)
        fh = self.linear_final(h_sa)
        fh = self.dropout2(fh)

        h = self.LN(h + fh)

        return h

class HeLPFormer(nn.Module):
    # def __init__(self, g, input_dimensions, embeddings_dimension=64, num_layers=8, num_gnns=2, nheads=2, dropout=0,  temper=1.0, num_type=4, beta=1):
    def __init__(self, input_dimensions, embeddings_dimension, num_nodes, num_type, args: argparse.Namespace):

        super(HeLPFormer, self).__init__()
        self.embeddings_dimension = embeddings_dimension
        self.L = args.L
        self.K = args.K
        self.nheads = args.num_heads
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension) for in_dim in input_dimensions])

        self.dropout = args.dropout
        self.GCNLayers = torch.nn.ModuleList()
        self.RELayers = torch.nn.ModuleList()
        self.GTLayers = torch.nn.ModuleList()
        for _ in range(self.K):
            self.GCNLayers.append(GraphConv(
                self.embeddings_dimension, self.embeddings_dimension, activation=F.relu))
            self.RELayers.append(REConv(num_type, num_type, activation=F.relu, num_type=num_type))
        for _ in range(self.L):
            self.GTLayers.append(
                AGTLayer(self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=args.temperature, rl=True, rl_dim=num_type, beta=args.beta, slope=args.slope))
        self.Drop = nn.Dropout(self.dropout)

    def forward(self, graph, features_list, seqs, type_emb, node_type) -> torch.Tensor:
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))

        ret = []
        gh = torch.vstack(h)
        ret.append(l2_normalize(gh))

        R = type_emb[node_type]
        with graph.local_scope():
            graph = graph.add_self_loop()
            for layer in range(self.K):
                gh = self.GCNLayers[layer](graph, gh)
                gh = self.Drop(gh)
                R = self.RELayers[layer](graph, R, node_type)

        h = gh[seqs]
        rh = R[seqs]
        for layer in range(self.L):
            h = self.GTLayers[layer](h, rh=rh)
            ret.append(l2_normalize(h[:, 0, :]))

        return torch.hstack(ret)