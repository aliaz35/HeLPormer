import numpy as np
import torch
from typing import Callable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=1e-4, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

class Evaluator:
    def __init__(self, metrics: dict[str, Callable]):
        self.scores = {name : [] for name, _ in metrics.items()}
        self.evaluate = {name : method for name, method in metrics.items()}

    def compute(self, method: str, *args, **kwargs) -> float:
        ret = self.evaluate[method](*args)
        if kwargs["save"]:
            self.scores[method].append(ret)

        return ret

    def summary(self):
        return ({name : np.average(score) for name, score in self.scores.items()},
                {name: np.std(score) for name, score in self.scores.items()})

class LinkEvaluator(Evaluator):
    def __init__(self):
        super().__init__({"auc" : roc_auc_score, "ap" : average_precision_score})

    def __call__(self, pos_y_hat, neg_y_hat, save=False):
        labels = (torch.vstack((torch.ones(pos_y_hat.size(0), 1), torch.zeros(neg_y_hat.size(0), 1)))
                  .cpu().detach().numpy())
        predictions = (torch.vstack((pos_y_hat, neg_y_hat))
                 .cpu().detach().numpy())

        return {
            "auc" : self.compute("auc", labels, predictions, save=save),
            "ap": self.compute("ap", labels, predictions, save=save),
        }
        # train_auc = roc_auc_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
        # train_ap = average_precision_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
