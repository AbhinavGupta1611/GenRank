import os
import numpy as np
from random import choices

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from geomloss import SamplesLoss

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
        self.ymean = ymean            # mean value of the target variables
        self.ystd = ystd            # standard deviation of the target variables
        self.weights_index = weights_index

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x1 = self.data[idx, 0, self.pred_indices]  # Only the first member of the ensemble [0] is used because the predictors are the same across all members
        x1 = torch.div(x1 - self.mx, self.sdx)
        y1 = self.data[idx, :, self.y_indices]
        y1 = (y1 - self.ymean)/self.ystd
        weights = self.data[idx, :, self.weights_index]
        return x1, y1, weights

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

def mahalanobis_cdist(x, cov_inv):
    # x: (N, D)
    diff = x.unsqueeze(1) - x.unsqueeze(0)   # (N, N, D)
    dist = torch.sqrt(torch.einsum('ijk,kl,ijl->ij', diff, cov_inv, diff) + 1e-8)
    return dist

def compute_cov(x):
    mean = torch.mean(x, dim=0, keepdim=True)
    xc = x - mean
    cov = xc.T @ xc / (x.shape[0] - 1)
    return cov


def density_loss(pred, y, sigma=0.1):
    # pred: (N, D), y: (M, D)

    dists = torch.cdist(pred, y)  # (N, M)

    # Gaussian kernel density estimate
    K = torch.exp(-dists**2 / (2 * sigma**2))

    density = K.mean(dim=1)  # (N,)

    # penalize low density regions
    loss = -torch.log(density + 1e-8).mean()

    return loss

# Loss function
class customLoss(nn.Module):
    def __init__(self, p=2, sigma=0.05, spread_weight=0.1):
        super().__init__()
        self.geomloss_fn = SamplesLoss("sinkhorn", p=p, blur=sigma)
        self.spread_weight = spread_weight

    def forward(self, pred, y, weights):

        # Geometric loss
        geoloss, loss_dist = 0, 0
        loss_list = []
        for i in range(pred.shape[0]):
            mask = ~torch.isnan(y[i, :, 0])

            ytmp = y[i, mask, :]
            weights_tmp = weights[i, mask]
            predtmp = pred[i, :, :]

            # Remove padded samples
            weights_pred = torch.ones(predtmp.shape[0], device=predtmp.device) / predtmp.shape[0]
            loss_tmp = self.geomloss_fn(weights_pred, predtmp, weights_tmp, ytmp)
            geoloss += loss_tmp

            # Spread loss
            """
            d_pred = torch.cdist(predtmp, predtmp) 
            d_true = torch.cdist(ytmp, ytmp)

            loss_dist_tmp = torch.abs(d_pred.mean() - d_true.mean())
            loss_dist += loss_dist_tmp
            """
            std_pred     = predtmp.std(dim=0)                          # (n_params,)
            std_true     = ytmp.std(dim=0)                             # (n_params,)
            ratio        = std_pred / (std_true + 1e-8)                # (n_params,)

            # penalize deviation of ratio from 1.0
            loss_dist += torch.mean(torch.abs(ratio - 1.0))

            # Only Sinkhorn loss (excluding spread term) is recorded deliberately for per-basin diagnostics
            loss_list.append(loss_tmp.item())
            #loss_list.append(loss_tmp)

        return geoloss/pred.shape[0] + self.spread_weight * loss_dist/pred.shape[0], loss_list
        #return geoloss/pred.shape[0], loss_list

# define MLP model class
class MLP_ensemble(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0, n_members=10, activation=nn.ReLU):
        super().__init__()

        self.n_members = n_members
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
        layers.append(nn.Linear(prev_dim, output_dim * n_members))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out.view(-1, self.n_members, self.output_dim)
    

class ConditionalGenerator(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, output_dim, theta_range,dropout_p=0.0):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.sigmoid = torch.sigmoid
        self.theta_min = theta_range[:, 0]
        self.theta_max = theta_range[:, 1]
        
        layers = []
        prev_dim = input_dim + latent_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, z):
        # x: (batch, input_dim)
        # z: (batch, latent_dim)
        inp = torch.cat([x, z], dim=1)
        out = self.net(inp)
        out = self.sigmoid(out)  # Ensures that the outputs are in [0, 1]
        out = out * (self.theta_max - self.theta_min) + self.theta_min

        return out

# define the module to train the model
def train_mod(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    
    tr_loss  = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        loss, _ = loss_fn(pred, y)
        #loss = loss.mean()

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
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
         
        optimizer.step()
        tr_loss += loss.item()

        del loss, pred

    tr_loss /= size
    return tr_loss, model.state_dict()

def train_mod_new(dataloader, model, loss_fn, optimizer, latent_dim, n_samples):
    size = len(dataloader)
    model.train()
    
    tr_loss  = 0
    for batch, (X, y, weights) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        weights = weights.to(device)

        batch_size = X.shape[0]
        Z = torch.randn(batch_size, n_samples, latent_dim).to(device)

        X_expanded = X.unsqueeze(1).repeat(1, n_samples, 1)
        X_flat = X_expanded.view(batch_size * n_samples, -1)
        z_flat = Z.view(batch_size * n_samples, -1)

        # Compute prediction error
        pred = model(X_flat, z_flat)
        pred = pred.view(batch_size, n_samples, -1)

        loss, _ = loss_fn(pred, y, weights)
        #loss = loss.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        """
        if batch % 3 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"Total Gradient Norm: {total_norm}")
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
         
        optimizer.step()
        tr_loss += loss.item()

        del loss, pred

    tr_loss /= size
    return tr_loss, model.state_dict()

# define the module to test the model
def test_mod(dataloader, model):
    model.eval()
    sse = 0
    ynse, pred_list = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #y = y.view(len(y),1)
            pred = model(X)
            
            sse += torch.sum((pred - y)**2, dim=0)
            ynse.append(y)
            pred_list.append(pred)

    ynse = torch.cat(ynse, dim=0)
    pred_list = torch.cat(pred_list, dim=0)
    sst = torch.sum((ynse - torch.mean(ynse, dim=0))**2, dim=0)
    nse = 1 - sse/sst

    print(f"NSE: {nse[0].item():>8f} \n")
    return nse, pred_list, ynse

# define the module to get the MC predictions
def ensemble_test_mod(dataloader, model, forward_passes=10):
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(forward_passes):
            ypred_list, yobs_list = [], []
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                
                # Perform a forward pass and store the prediction
                output = model(X)
                ypred_list.append(output)
                yobs_list.append(y)

            ypred_list = torch.cat(ypred_list, dim=0)
            predictions.append(ypred_list)
     
    # Stack predictions
    predictions = torch.cat(predictions, dim=0)
    yobs = torch.cat(yobs_list, axis=0)
    
    return yobs, predictions


def ensemble_test_mod_new(dataloader, model, latent_dim, n_samples):
    model.eval()
    
    ypred_list = []
    yobs_list = []
    weights_list = []

    with torch.no_grad():
        for X, y, weights in dataloader:
            X, y = X.to(device), y.to(device)
            weights = weights.to(device)

            batch_size = X.shape[0]

            # -------------------------------
            # 1. Sample z
            # -------------------------------
            #z = torch.randn(batch_size, n_samples, latent_dim).to(device)
            #z = (torch.rand(batch_size, n_samples, latent_dim) * 2 * 3 - 3).to(device)
            z = sample_z_scaled_2(batch_size, n_samples, latent_dim, tail_frac=0.5, scale_min=1.5, scale_max=1.75, clip_val=3.5, device=device)

            # -------------------------------
            # 2. Repeat X
            # -------------------------------
            X_expanded = X.unsqueeze(1).repeat(1, n_samples, 1)

            # Flatten
            X_flat = X_expanded.view(batch_size * n_samples, -1)
            z_flat = z.view(batch_size * n_samples, -1)

            # -------------------------------
            # 3. Forward pass
            # -------------------------------
            pred = model(X_flat, z_flat)
            pred = pred.view(batch_size, n_samples, -1)

            # -------------------------------
            # 4. Store results
            # -------------------------------
            ypred_list.append(pred)
            yobs_list.append(y)
            weights_list.append(weights)

    # Concatenate across batches
    ypred = torch.cat(ypred_list, dim=0)
    yobs  = torch.cat(yobs_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    return yobs, ypred, weights


def ensemble_test_mod_new_chunk(dataloader, model, latent_dim, n_samples, chunk_size=1000):
    model.eval()
    
    ypred_list   = []
    yobs_list    = []
    weights_list = []

    with torch.no_grad():
        for X, y, weights in dataloader:
            X, y    = X.to(device), y.to(device)
            weights = weights.to(device)
            batch_size = X.shape[0]

            # accumulate chunks on CPU
            pred_chunks = []

            for chunk_start in range(0, n_samples, chunk_size):
                n_this = min(chunk_size, n_samples - chunk_start)

                # 1. Sample z
                #z = torch.randn(batch_size, n_this, latent_dim).to(device)
                #z = sample_z_with_tails(batch_size, n_samples, latent_dim, tail_frac=0.4, tail_scale=1.5, device=device)
                #z  = sample_z_scaled(batch_size, n_samples, latent_dim, tail_frac=0.3, scale_min=1.5, scale_max=2.0, device=device)
                z = sample_z_scaled_2(batch_size, n_this, latent_dim, tail_frac=0.5, scale_min=1.5, scale_max=1.75, clip_val=3.5, device=device)

                # 2. Repeat X
                X_expanded = X.unsqueeze(1).repeat(1, n_this, 1)
                X_flat     = X_expanded.view(batch_size * n_this, -1)
                z_flat     = z.view(batch_size * n_this, -1)

                # 3. Forward pass
                pred = model(X_flat, z_flat)
                pred = pred.view(batch_size, n_this, -1)

                # 4. Move to CPU immediately to free GPU memory
                pred_chunks.append(pred.cpu())

                del z, X_expanded, X_flat, z_flat, pred
                torch.cuda.empty_cache()

            # concatenate chunks along sample dimension
            pred_all = torch.cat(pred_chunks, dim=1)  # (batch, n_samples, output_dim)

            ypred_list.append(pred_all)
            yobs_list.append(y.cpu())
            weights_list.append(weights.cpu())

    ypred   = torch.cat(ypred_list,   dim=0)
    yobs    = torch.cat(yobs_list,    dim=0)
    weights = torch.cat(weights_list, dim=0)
    return yobs, ypred, weights

def sample_z_scaled_2(batch_size, n_samples, latent_dim, tail_frac=0.3, scale_min=1.5, scale_max=2.5, clip_val=3.5, device='cpu'):
    n_tail   = int(n_samples * tail_frac)
    n_center = n_samples - n_tail

    # standard samples
    z_center = torch.randn(batch_size, n_center, latent_dim)

    # scaled tail samples
    z_tail  = torch.randn(batch_size, n_tail, latent_dim)
    scales  = torch.rand(batch_size, n_tail, 1) * (scale_max - scale_min) + scale_min
    z_tail  = z_tail * scales

    # clip individual elements to [-clip_val, clip_val]
    z_tail  = torch.clamp(z_tail, min=-clip_val, max=clip_val)

    z   = torch.cat([z_center, z_tail], dim=1)
    idx = torch.randperm(n_samples)
    return z[:, idx, :].to(device)