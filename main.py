import os
import sys

import dgl
import numpy as np
import torch

from ModelTrainer import ModelTrainer
from utils.preprocess import graph_tokenize, prompt_parse, mat_to_tensor, edge_to_g
from utils.data import load_data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append("utils/")

args = prompt_parse()

if __name__ == "__main__":
    print(f"Using device: {args.device}")

    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    feats_type = args.feats_type
    features_list, adj_mat, edges_mat, dl = load_data(args, args.dataset)
    features_list: list[torch.sparse.Tensor] = [mat_to_tensor(features).to(args.device) for features in features_list]
    node_cnt: list[int] = [features.shape[0] for features in features_list]
    sum_node: int = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(args.device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim]))
            features_list[i] = features_list[i].to(args.device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(
                indices, values, torch.Size([dim, dim]))
            features_list[i] = features_list[i].to(args.device)
    
    train_edges, neg_train_edges              = edges_mat["train"]
    val_edges, neg_val_edges                  = edges_mat["valid"]
    test_edges_random, neg_test_edges_random  = edges_mat["test_random"]

    num_nodes = sum_node

    train_g = dgl.to_bidirected(edge_to_g(train_edges, num_nodes)).to(args.device)

    adj_mat = adj_mat.to_scipy(layout="csr")
    adj_mat += adj_mat.T

    pattern = adj_mat
    for l in range(2, args.gamma + 1):
        adj_mat *= adj_mat
        pattern += args.delta ** (l - 1) * adj_mat
    pattern = pattern.tocoo()

    # node_seq = sample_seq(train_g, features_list[0].shape[0], args.len_seq)
    node_seq = graph_tokenize(train_g, args.len_seq)
    type_emb = torch.eye(len(node_cnt)).to(args.device)
    node_type = torch.tensor([i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]).to(args.device)

    trainer = ModelTrainer(
        in_dims,
        num_nodes,
        node_cnt,
        node_type,
        edges_mat,
        features_list,
        node_seq,
        type_emb,
        pattern,
        train_g,
        args
    )
    for i in range(args.repeat):
        trainer.reset()
        H = None
        for epoch in range(args.epoch):
            H = trainer.train_epoch(epoch)
            trainer.valid_epoch(epoch, H)
            if trainer.early_stopping.early_stop:
                print("Early stopping!")
                break
        trainer.test_epoch(epoch, H)
    avg, std = trainer.evaluator.summary()
    print("summary:\n AUC: {:.4f} ± {:.4f} | AP: {:.4f} ± {:.4f}".format(avg["auc"], std["auc"], avg["ap"], std["ap"]))


