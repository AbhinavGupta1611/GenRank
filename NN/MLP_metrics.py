"""

MLP for predicting performance metrics
Author: Abhinav Gupta (Created: 3 Apr 2026)
"""

import os
import numpy as np
from random import choices
from scipy.stats import spearmanr
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# function to compute NSE
def computeNSE(obs, pred):
    sse = np.sum((obs - pred)**2)
    sst = np.sum((obs - np.mean(obs))**2)
    nse = 1 - sse/sst
    return nse

# build a dataset class
class CustomData(Dataset):
    def __init__(self, data, mean_X, std_X, pred_indices, y_indices, weights_index, ymean, ystd):
        super(Dataset, self).__init__()
        self.data = data
        self.mx = mean_X            # mean of predictor variables
        self.sdx = std_X            # standard deviation of predictor variables
        self.pred_indices = pred_indices  # indices of predictor variables in the data array
        self.y_indices = y_indices      # Indices of the target variables in the data array
        self.weights_index = weights_index  # Index of the weights column in the data array
        self.ymean = ymean            # mean value of the target variables
        self.ystd = ystd            # standard deviation of the target variables

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x1 = self.data[idx, self.pred_indices]
        x1 = torch.div(x1 - self.mx, self.sdx)
        y1 = self.data[idx, self.y_indices]
        y1 = (y1 - self.ymean)/self.ystd
        weight = self.data[idx, self.weights_index]
        return x1, y1, weight

# custom sampler function 
class SubsetSampler(Sampler):
    def __init__(self, indices, generator=None):
        super(Sampler, self).__init__()
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)

def weights_for_target_ess(score, target_ess=400, iters=60):
    s = score
    lo, hi = 1e-3, 1e3
    for _ in range(iters):
        T = 0.5*(lo+hi)
        w = np.exp(s / T)
        w /= w.sum()
        ess = 1.0 / np.sum(w**2)
        if ess < target_ess:   # too concentrated → flatten (raise T)
            lo = T
        else:                  # too uniform → sharpen (lower T)
            hi = T
    T = 0.5*(lo+hi)
    w = np.exp(s / T); w /= w.sum()
    return w

# Weighted loss function (Only the third metrics KGE is weighted; Written using Claude)
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y, weights):
        se = (pred - y)**2                    # (batch, n_targets), per-element
        w = weights.view(-1, 1)
        return (w  * se).mean()

# Weighted loss function (Only the third metrics KGE is weighted; Written using Claude)
class WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y, weights):
        se = torch.abs(pred - y)                    # (batch, n_targets), per-element
        w = weights.view(-1, 1)
        return (w  * se).mean()
    
# define MLP model class
class MLP_metrics(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0, activation=nn.ReLU):
        super().__init__()

        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim

        # hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout_p))
            prev_dim = h

        # output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out

# define MLP classification model class
class MLP_classification(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0, activation=nn.ReLU):
        super().__init__()

        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim

        # hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout_p))
            prev_dim = h

        # output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        out = torch.sigmoid(out)
        return out
    
# define the module to train the model
def train_mod(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    
    tr_loss  = 0
    for batch, (X, y, weights) in enumerate(dataloader):
        X, y, weights = X.to(device, non_blocking=True), y.to(device, non_blocking=True), weights.to(device, non_blocking=True)

        # Compute prediction error
        pred = model(X)

        #loss = loss_fn(pred, y, weights)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        """
        if batch % 10 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"Total Gradient Norm: {total_norm}")
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
         
        optimizer.step()
        tr_loss += loss.item()

        del loss, pred

    tr_loss /= size
    return tr_loss, model.state_dict()

# define the module to test the model
def test_mod(dataloader, model):
    model.eval()
    sse = 0
    ynse, pred_list, X_list = [], [], []
    with torch.no_grad():
        for X, y, weights in dataloader:
            X, y, weights = X.to(device), y.to(device), weights.to(device)
            #y = y.view(len(y),1)
            pred = model(X)
            
            sse += torch.sum((pred - y)**2, dim=0)
            ynse.append(y)
            pred_list.append(pred)
            X_list.append(X)

    ynse = torch.cat(ynse, dim=0)
    pred_list = torch.cat(pred_list, dim=0)
    X_list = torch.cat(X_list, dim=0)
    sst = torch.sum((ynse - torch.mean(ynse, dim=0))**2, dim=0)
    nse = 1 - sse/sst

    print(f"NSE: {nse[0].item():>8f} \n")
    return nse, pred_list, ynse, X_list


# COmputation of metrics to evaluate the performance of the ranking model (regret@k, top-k Spearman, and precision@k)
def compute_regret_at_k(obs, pred, k=10):
    """
    Compute regret@k, top-k Spearman, and precision@k per basin.
    
    Parameters
    ----------
    obs  : np.ndarray, shape (n_samples,) — true KGE values (original scale)
    pred : np.ndarray, shape (n_samples,) — predicted KGE values (original scale)
    k         : int — number of top sets to consider
    """

    # --- regret@k ---
    # true best KGE achievable in this basin
    true_best = np.max(obs)

    # regret@k: best true skill among model's top-k picks
    topk_pred_idx = np.argsort(pred)[-k:]           # model's top-k indices
    best_in_topk  = np.max(obs[topk_pred_idx])      # best true skill among them
    regret_at_k   = true_best - best_in_topk        # skill gap vs true best

    # --- top-k Spearman ---
    # rank correlation restricted to the truly top-k sets
    topk_true_idx = np.argsort(obs)[-k:]            # true top-k indices
    if len(topk_true_idx) >= 3:                      # need >=3 for correlation
        sp_corr, _ = spearmanr(obs[topk_true_idx], pred[topk_true_idx])
    else:
        sp_corr = np.nan

    # --- precision@k ---
    # fraction of model's top-k that are in the true top-k
    topk_pred_set = set(topk_pred_idx)
    topk_true_set = set(topk_true_idx)
    precision_at_k = len(topk_pred_set & topk_true_set) / k

    results = {'true_best_kge': true_best, 'regret_at_k': regret_at_k, 'topk_spearman': sp_corr, 'precision_at_k': precision_at_k}

    return results

# Basin-wise bacthing of data so that ranking loss can be computed meaningfully
class BasinBatchSampler(Sampler):
    def __init__(self, basin_seq, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.basin_to_idx = {}
        for i, b in enumerate(basin_seq):
            self.basin_to_idx.setdefault(b, []).append(i)
        self.basins = list(self.basin_to_idx.keys())

    def __iter__(self):
        basins = self.basins[:]
        if self.shuffle:
            random.shuffle(basins)
        for b in basins:
            idx = self.basin_to_idx[b][:]
            if self.shuffle:
                random.shuffle(idx)
            for k in range(0, len(idx), self.batch_size):
                yield idx[k:k + self.batch_size]

    def __len__(self):
        total = 0
        for idx in self.basin_to_idx.values():
            total += (len(idx) + self.batch_size - 1) // self.batch_size
        return total


# L1 loss + ranking loss
class CombinedLoss(nn.Module):
    def __init__(self, rank_weight, n_pairs, top_frac, min_true_gap, kge_col, kge_ystd):
        super().__init__()
        self.rank_weight      = rank_weight
        self.n_pairs          = n_pairs
        self.top_frac         = top_frac
        self.min_true_gap_std = min_true_gap / (kge_ystd + 1e-8)
        self.kge_col          = kge_col

    def forward(self, pred, y, weights):
        # weighted L1 across all metrics — full batch
        ae = torch.abs(pred - y)
        w  = weights.view(-1, 1)
        l1 = (w * ae).mean()

        # ranking loss — top-frac only
        rank = self._ranking_loss(pred[:, self.kge_col], y[:, self.kge_col])
        
        return l1 + self.rank_weight * rank

    def _ranking_loss(self, pred_k, true_k):
        n = pred_k.shape[0]
        if n < 4:
            return torch.tensor(0.0, device=pred_k.device)

        # restrict to true top-frac
        ntop     = max(2, int(self.top_frac * n))
        top_idx  = torch.topk(true_k, ntop).indices
        pred_top = pred_k[top_idx]
        true_top = true_k[top_idx]

        # sample pairs within top-frac
        i = torch.randint(0, ntop, (self.n_pairs,), device=pred_k.device)
        j = torch.randint(0, ntop, (self.n_pairs,), device=pred_k.device)

        # remove self-pairs
        valid = i != j
        if valid.sum() < 2:
            return torch.tensor(0.0, device=pred_k.device)
        i, j = i[valid], j[valid]

        # filter near-tied pairs
        gap  = (true_top[i] - true_top[j]).abs()
        keep = gap > self.min_true_gap_std
        if keep.sum() < 2:
            return torch.tensor(0.0, device=pred_k.device)
        i, j = i[keep], j[keep]

        # orient so i_better is truly better
        swap     = true_top[j] > true_top[i]
        i_better = torch.where(swap, j, i)
        j_worse  = torch.where(swap, i, j)

        # logistic pairwise loss
        diff = pred_top[i_better] - pred_top[j_worse]
        return torch.nn.functional.softplus(-diff).mean()


# Training module to accumulate gradients across each basin before updating the model weights
def train_mod_accumulated(dataloader, model, loss_fn, optimizer):
    """
    Accumulate gradients across all basin batches before each optimizer step.
    One optimizer step per epoch, using gradients from all basins.
    """
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    n_batches  = len(dataloader)
    
    for batch, (X, y, weights) in enumerate(dataloader):
        X, y, weights = (X.to(device, non_blocking=True),
                         y.to(device, non_blocking=True),
                         weights.to(device, non_blocking=True))
        
        pred = model(X)
        
        # divide by n_batches so accumulated gradient has same scale
        # as a single-batch gradient — keeps LR interpretation stable
        loss = loss_fn(pred, y, weights) / n_batches
        loss.backward()                  # accumulate, don't step yet
        
        total_loss += loss.item()

    # single optimizer step after all basins
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss, model.state_dict()