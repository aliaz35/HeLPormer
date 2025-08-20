import torch
import time

from lp import LinkPred, binary_cross_entropy
from utils.pytorchtools import EarlyStopping, LinkEvaluator
from model import HeLPFormer
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(
            self,
            in_dims,
            num_nodes,
            node_cnt,
            node_type,
            edges_mat,
            features_list,
            tokens,
            type_emb,
            pattern,
            graph,
            args
    ):
        self.in_dims = in_dims
        self.num_nodes = num_nodes
        self.node_cnt = node_cnt
        self.evaluator = LinkEvaluator()
        self.model          = None
        self.predictor      = None
        self.optimizer      = None
        self.scheduler      = None
        self.early_stopping = None

        self.edges_mat = edges_mat
        self.features_list = features_list
        self.node_type = node_type
        self.tokens = tokens
        self.type_emb = type_emb
        self.pattern = pattern
        self.graph = graph

        self.args = args


    def reset(self):
        self.model = HeLPFormer(input_dimensions=self.in_dims, embeddings_dimension=self.args.hidden_dim, num_nodes=self.num_nodes, num_type=len(self.node_cnt), args=self.args).float().to(self.args.device)
        self.predictor = LinkPred(args=self.args).float().to(self.args.device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.predictor.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, save_path="checkpoint/HeLPormer_{}_{}_{}.pt".format(self.args.dataset, self.args.L, self.args.device))



    def train_epoch(
            self,
            epoch
    ) -> torch.Tensor:
        print("Epoch {:05d}".format(epoch))
        H = None
        pos_edges, neg_edges = self.edges_mat["train"]
        for batch, (pos_edges, neg_edges) in enumerate(
                zip(
                    DataLoader(torch.tensor(pos_edges), shuffle=True, batch_size=self.args.batch_size),
                    DataLoader(torch.tensor(neg_edges), shuffle=True, batch_size=self.args.batch_size),
                )
        ):
            t_start = time.time()

            self.model.zero_grad()
            self.model.train()
            self.predictor.train()

            H = self.model(self.graph, self.features_list, self.tokens, self.type_emb, self.node_type)

            pos_y_hat = self.predictor(pos_edges, self.pattern, H)
            neg_y_hat = self.predictor(neg_edges, self.pattern, H)

            loss = binary_cross_entropy(pos_y_hat, neg_y_hat)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)

            self.optimizer.step()

            scores = self.evaluator(pos_y_hat, neg_y_hat)

            t_end = time.time()

            print("Batch {:05d} | Train_Loss: {:.4f} | Train_AUC: {:.4f} | Train_AP: {:.4f} | Time: {:.4f}".format(
                batch, loss.item(), scores["auc"].item(), scores["ap"].item(), t_end - t_start))
        return H


    @torch.no_grad()
    def valid_epoch(
            self,
            epoch,
            H,
    ) -> None:
        t_start = time.time()
        self.model.eval()
        self.predictor.eval()

        pos_edges, neg_edges = self.edges_mat["valid"]
        pos_y_hat = self.predictor(pos_edges, self.pattern, H)
        neg_y_hat = self.predictor(neg_edges, self.pattern, H)

        loss = binary_cross_entropy(pos_y_hat, neg_y_hat)

        scores = self.evaluator(pos_y_hat, neg_y_hat)
        # scheduler.step(loss)
        t_end = time.time()
        # print validation info
        print("Epoch {:05d} | Val_Loss:   {:.4f} | Val_AUC:   {:.4f} | Val_AP:   {:.4f} | Time: {:.4f}".format(
            epoch, loss.item(), scores["auc"].item(), scores["ap"].item(), t_end - t_start))
        # early stopping
        self.early_stopping(loss, self.model)


    @torch.no_grad()
    def test_epoch(
            self,
            epoch,
            H,
    ) -> None:
        t_start = time.time()
        self.model.load_state_dict(torch.load(
            "checkpoint/HeLPormer_{}_{}_{}.pt".format(self.args.dataset, self.args.L, self.args.device)))

        self.model.eval()
        self.predictor.eval()

        pos_edges, neg_edges = self.edges_mat["test_random"]
        pos_y_hat = self.predictor(pos_edges, self.pattern, H)
        neg_y_hat = self.predictor(neg_edges, self.pattern, H)

        loss = binary_cross_entropy(pos_y_hat, neg_y_hat)

        scores = self.evaluator(pos_y_hat, neg_y_hat, save=True)
        t_end = time.time()
        print("Epoch {:05d} | Test_Loss: {:.4f}  | Test_AUC:  {:.4f} | Test_AP: {:.4f}  | Time: {:.4f}".format(
            epoch, loss.item(), scores["auc"].item(), scores["ap"].item(), t_end - t_start))
