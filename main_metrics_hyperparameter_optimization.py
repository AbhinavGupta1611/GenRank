"""

hyperparameter optimization for the MM_metrics model

Author: Abhinav Gupta (Created: 20 Apr 2026)
"""
 
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import copy
from scipy.stats import spearmanr 

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from geomloss import SamplesLoss
 
from NN.MLP_metrics import MLP_metrics, CustomData, SubsetSampler, WeightedMSELoss, WeightedL1Loss, train_mod, test_mod, computeNSE, compute_regret_at_k

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs_max = 500    # Number of epochs
nseeds = 1      # Number of seeds
hidden_dims_list = [[256, 128, 64], [512, 256, 128, 64], [1024, 512, 256, 128], [512, 256, 128, 64, 32], [1024, 512, 256, 128, 64]]       # hidden dimension, number of layers, output dimension
lrate_list = [10**(-4), 5*10**(-4), 10**(-3)]     # Learning rate
N_list = [2**6, 2**7, 2**8, 2**9, 2**10]     # batch size
loss_fn = nn.L1Loss()
loss_fn_eval = nn.L1Loss()

dropout = 0.0

weight_expo = 0
phase = '1'
save_no = '1'

pred_indices = range(8,60)      ###############################################################################
y_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
weight_index = 30
param_cols = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12,\
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
output_dim = len(y_indices)

train_frac = 0.70
val_frac = 2/7
####################################################################
main_dir  = 'D:/Research/Regionalization'
data_dir = os.path.join(main_dir, 'data')
results_dir = os.path.join(main_dir, 'results')

stat_dir = 'CAMELS_raw/camels_attributes_v2.0/camels_attributes_v2.0'
####################################################################

par_name = [
"PT alpha",

# SNOW-17 parameters
"SCF",     #Multiplying factor which adjusts Precipitation (accounts for gage snow catch deficiencies),
"PXTEMP",	#Temperature that separates rain from snow, deg C
"TTI",     #Temperature interval for mixture of snow and rain, deg C
"MFMAX",   #Maximum melt factor during non-rain periods - assumed to occur on June 21
"MFMIN",   #Minimum melt factor during non-rain periods - assumed to occur on Dec 21
"UADJ",    #Average wind function during rain-on-snow periods
"MBASE",   #Base temperature for snowmelt computations during non-rain periods, deg C
"TIPM",    #Antecedent temperature index parameter (0.01 to 1.0)
"PLWHC",   #Percent liquid water holding capacity (maximum value allowed is 0.4)
"NMF",     #Maximum negative melt factor (should be a positive value)
"DAYGM",   #A constant daily rate of melt at the soil-snow interface

#SAC-SMA parameters
"uztwm",  #Upper zone tension water storage maximum [mm]
"uzfwm",  #Upper zone free water storage maximum [mm]
"lztwm",  #Lower zone tension water storage maximum [mm]
"lzfpm",  #Lower zone primary free water storage maximum [mm]
"lzfsm",  #Lower zone supplementary free water storage maximum [mm]
"uzk",    #Upper zone free water lateral depletion rate (btw 0 and 1)
"lzpk",   #Lower zone primary free water depletion rate (btw 0 and 1)
"lzsk",   #Lower zone supplementary free water depletion rate (btw 0 and 1)
"zperc",  #Percolation demand scale parameter: multiplier of the percolation equation
"rexp",   #Percolation demand shape parameter: exponent of the percolation equation
"pfree",  #Percolating water split parameter (btw 0 and 1): fraction of water percolating to lower free water storage
"pctim",  #Impervious fraction of the watershed area (btw 0 and 1): permanent impervious area
"adimp",  #Additional impervious areas (btw 0 and 1): temporary impervious area
"riva",   #Fraction of riparian vegetation area (btw 0 and 1)
"side",   #The ratio of deep recharge to river channel base flow
"rserv",  #Fraction of lower zone free water not transferrable to lower zone tension water (btw 0 and 1)

# Routing parameters
"alpha - IUH parameter",
"beta  - IUH parameter"
]
######################################################################################################################################
# Read the list of basins
filename = os.path.join(data_dir, '531_basins', 'basin_list.txt')
basin_list = pd.read_csv(filename, header=None, dtype=str)[0].tolist()

# Read data on static attributes
filename = os.path.join(data_dir, stat_dir, 'camels_clim.txt')
clim_df = pd.read_csv(filename, delimiter=';', dtype = {'gauge_id': str})
clim_df = clim_df[['gauge_id', 'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',  'low_prec_freq', 'low_prec_dur']]

filename = os.path.join(data_dir, stat_dir, 'camels_geol.txt')
geol_df = pd.read_csv(filename, delimiter=';', dtype = {'gauge_id': str})
geol_df = geol_df[['gauge_id', 'geol_permeability']]

filename = os.path.join(data_dir, stat_dir, 'camels_soil.txt')
soil_df = pd.read_csv(filename, delimiter=';', dtype = {'gauge_id': str})
soil_df = soil_df[['gauge_id', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'sand_frac', 'clay_frac', 'max_water_content', 'organic_frac']]

filename = os.path.join(data_dir, stat_dir, 'camels_topo.txt')
topo_df = pd.read_csv(filename, delimiter=';', dtype = {'gauge_id': str})
topo_df = topo_df[['gauge_id', 'elev_mean', 'slope_mean', 'area_gages2', 'gauge_lat', 'gauge_lon']]

filename = os.path.join(data_dir, stat_dir, 'camels_vege.txt')
vege_df = pd.read_csv(filename, delimiter=';', dtype = {'gauge_id': str})
vege_df = vege_df[['gauge_id', 'frac_forest', 'dom_land_cover_frac', 'lai_max']]

# Merge all static attributes
static_df = clim_df.merge(geol_df, on='gauge_id', how='inner')\
    .merge(soil_df, on='gauge_id', how='inner')\
    .merge(topo_df, on='gauge_id', how='inner')\
    .merge(vege_df, on='gauge_id', how='inner')
######################################################################################################################################

# Collate data in one dataframe
model_data = pd.DataFrame()
for basin in basin_list:

    # Read parameter data
    filename = os.path.join(results_dir, 'DREAM_results', 'param_top_10000_{}.txt'.format(basin))
    param_df = pd.read_csv(filename, header = None, delimiter=',')

    NSE = param_df[30].values
    NSAE = param_df[31].values
    KGE = param_df[32].values

    param_df['score'] = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) + (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) + (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
    param_df = param_df[param_df['score'] >= np.nanquantile(param_df['score'].values, 1-6000/param_df.shape[0])]
    param_df = param_df.drop(columns = 'score')

    # Only keep the parameters that are used as predictors in the model
    param_df = param_df[param_cols]

    # Compute weights
    NSE = param_df[30].values
    NSAE = param_df[31].values
    KGE = param_df[32].values
    score = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) + (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) + (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
    weights = np.exp(weight_expo * score)
    weights = weights/np.nansum(weights)
    weight_df = pd.DataFrame({'weight': weights})

    # Only keep the parameters that are used as predictors in the model
    param_df = param_df[param_cols]

    stat_tmp = static_df[static_df['gauge_id'] == basin].drop('gauge_id', axis=1)
    stat_tmp = pd.concat([stat_tmp]*param_df.shape[0], ignore_index=True)
    
    data_tmp = pd.concat([param_df.reset_index(drop=True), stat_tmp.reset_index(drop=True), weight_df.reset_index(drop=True)], axis=1)
    data_tmp['basin'] = [basin]*param_df.shape[0]
    model_data = pd.concat([model_data, data_tmp], axis=0)

# Remove basins with NaN parameters
model_data = model_data[~model_data[0].isna()]

# Put three performance metrics (cols: 30, 31, 32) as first three columns
model_data = model_data[[30, 31, 32, 33, 34, 35, 36, 37] + [col for col in model_data.columns if col not in [30, 31, 32, 33, 34, 35, 36, 37]]]

# Training and testing basins
num_basins = len(basin_list)
all_inds = np.arange(num_basins)
np.random.seed(0)
np.random.shuffle(all_inds)

ntrain = int(train_frac * num_basins)
nval = int(val_frac * ntrain)

train_basin_inds = all_inds[:ntrain-nval]
val_basin_inds = all_inds[ntrain-nval:ntrain]

train_basins = [basin_list[ii] for ii in train_basin_inds]
val_basins = [basin_list[ii] for ii in val_basin_inds]

train_data = model_data[model_data['basin'].isin(train_basins)]
val_data = model_data[model_data['basin'].isin(val_basins)]

train_basin_seq = train_data['basin'].values
val_basin_seq = val_data['basin'].values
train_data = train_data.drop('basin', axis=1).values
val_data = val_data.drop('basin', axis=1).values

###############################################################################################################################
###############################################################################################################################
# Code for model training
meanx = np.nanmean(train_data[:, pred_indices], axis=0)
stdx = np.nanstd(train_data[:, pred_indices], axis=0)
stdx[stdx==0] = 10**(-6)
meanx = torch.from_numpy(meanx).float()
stdx = torch.from_numpy(stdx).float()

# Determine the input dimension
input_dim = len(meanx)

# Standard deviation of target variable
ystd = np.std(train_data[:, y_indices], axis=0) + 10**(-6)
ymean = np.mean(train_data[:, y_indices], axis=0)

# LSTM formatting of training and testing data
train_data = torch.from_numpy(train_data).float()
val_data = torch.from_numpy(val_data).float()
ymean = torch.from_numpy(ymean).float()
ystd = torch.from_numpy(ystd).float()

train_dataset = CustomData(train_data, meanx, stdx, pred_indices, y_indices, weight_index, ymean, ystd)
val_dataset = CustomData(val_data, meanx, stdx, pred_indices, y_indices, weight_index, ymean, ystd)


####################################################################################################################
write_data = []
# Model training
for hidden_dims in hidden_dims_list:
    for lrate in lrate_list:
        for N in N_list:
            
            # Define dataloaders
            train_dataloader = DataLoader(train_dataset, batch_size = N, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size = N, shuffle=False)

            # Set model and optimizer
            for seed in range(nseeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # instantiate the model class
                mlp = MLP_metrics(input_dim, hidden_dims, output_dim, dropout_p=dropout)
                mlp.to(device)

                # define loss and optimizer
                optimizer = torch.optim.Adam(mlp.parameters(), lr = lrate)

                # Fix the number of epochs and start model training
                ########################################################################################################################################################################
                loss_tr_list, loss_vl_list, model_state, nse_sv = [], [], [], []
                for t in range(epochs_max):
                    print(f"Epoch {t+1}\n-------------------------------")
                    loss_tr, state = train_mod(train_dataloader, mlp, loss_fn, optimizer)
                    model_state.append(copy.deepcopy(state))
                    loss_tr_list.append(loss_tr)
                    print(loss_tr)

                    if (t+1) % 10 == 0:
                        _, ypred, yobs, _ = test_mod(val_dataloader, mlp)
                        
                        loss_eval = loss_fn_eval(ypred, yobs).item()
                        ypred = ypred.cpu().detach().numpy()
                        yobs = yobs.cpu().detach().numpy()

                        NSE_list, corr_list, loss_list, regret_list, corr_topk_list, corr_sum_list = [], [], [], [], [], []
                        for basin in val_basins:
                            ind = np.nonzero(val_basin_seq == basin)[0]
                            nse_nse = computeNSE(yobs[ind, 0], ypred[ind, 0])
                            nse_nsae = computeNSE(yobs[ind, 1], ypred[ind, 1])
                            nse_kge = computeNSE(yobs[ind, 2], ypred[ind, 2])

                            corr_nse,_ = spearmanr(yobs[ind, 0], ypred[ind,0])
                            corr_nsae,_ = spearmanr(yobs[ind, 1], ypred[ind,1])
                            corr_kge,_ = spearmanr(yobs[ind, 2], ypred[ind,2])

                            regret_nse = compute_regret_at_k(yobs[ind, 0], ypred[ind, 0], k=30)
                            regret_nsae = compute_regret_at_k(yobs[ind, 1], ypred[ind, 1], k=30)
                            regret_kge = compute_regret_at_k(yobs[ind, 2], ypred[ind, 2], k=30)

                            NSE_list.append([nse_nse, nse_nsae, nse_kge])
                            corr_list.append([corr_nse, corr_nsae, corr_kge])
                            loss_list.append(loss_eval)
                            regret_list.append([regret_nse['regret_at_k'], regret_nsae['regret_at_k'], regret_kge['regret_at_k']])
                            corr_topk_list.append([regret_nse['topk_spearman'], regret_nsae['topk_spearman'], regret_kge['topk_spearman']])
                            corr_sum_list.append([corr_nse + regret_nse['topk_spearman'], corr_nsae + regret_nsae['topk_spearman'], corr_kge + regret_kge['topk_spearman']])

                        NSE_list = np.array(NSE_list)
                        corr_list = np.array(corr_list)
                        regret_list = np.array(regret_list)
                        corr_topk_list = np.array(corr_topk_list)
                        corr_sum_list = np.array(corr_sum_list)
                        
                        write_data.append([t+1, hidden_dims, lrate, N, np.percentile(loss_list, 5), np.percentile(loss_list, 10), np.percentile(loss_list, 25), np.median(loss_list), np.percentile(loss_list, 75), np.percentile(loss_list, 90), np.percentile(loss_list, 95),
                                            # Spearman correlation
                                            np.mean(corr_list[:,0]), np.percentile(corr_list[:,0], 5), np.percentile(corr_list[:,0], 10), np.percentile(corr_list[:,0], 25), np.median(corr_list[:,0]), np.percentile(corr_list[:,0], 75), np.percentile(corr_list[:,0], 90), np.percentile(corr_list[:,0], 95),
                                            np.mean(corr_list[:,1]), np.percentile(corr_list[:,1], 5), np.percentile(corr_list[:,1], 10), np.percentile(corr_list[:,1], 25), np.median(corr_list[:,1]), np.percentile(corr_list[:,1], 75), np.percentile(corr_list[:,1], 90), np.percentile(corr_list[:,1], 95),
                                            np.mean(corr_list[:,2]), np.percentile(corr_list[:,2], 5), np.percentile(corr_list[:,2], 10), np.percentile(corr_list[:,2], 25), np.median(corr_list[:,2]), np.percentile(corr_list[:,2], 75), np.percentile(corr_list[:,2], 90), np.percentile(corr_list[:,2], 95),
                                            # top-k-spearman
                                            np.mean(corr_topk_list[:,0]), np.percentile(corr_topk_list[:,0], 5), np.percentile(corr_topk_list[:,0], 10), np.percentile(corr_topk_list[:,0], 25), np.median(corr_topk_list[:,0]), np.percentile(corr_topk_list[:,0], 75), np.percentile(corr_topk_list[:,0], 90), np.percentile(corr_topk_list[:,0], 95),
                                            np.mean(corr_topk_list[:,1]), np.percentile(corr_topk_list[:,1], 5), np.percentile(corr_topk_list[:,1], 10), np.percentile(corr_topk_list[:,1], 25), np.median(corr_topk_list[:,1]), np.percentile(corr_topk_list[:,1], 75), np.percentile(corr_topk_list[:,1], 90), np.percentile(corr_topk_list[:,1], 95),
                                            np.mean(corr_topk_list[:,2]), np.percentile(corr_topk_list[:,2], 5), np.percentile(corr_topk_list[:,2], 10), np.percentile(corr_topk_list[:,2], 25), np.median(corr_topk_list[:,2]), np.percentile(corr_topk_list[:,2], 75), np.percentile(corr_topk_list[:,2], 90), np.percentile(corr_topk_list[:,2], 95),
                                            # regret_at_k
                                            np.mean(regret_list[:,0]), np.percentile(regret_list[:,0], 5), np.percentile(regret_list[:,0], 10), np.percentile(regret_list[:,0], 25), np.median(regret_list[:,0]), np.percentile(regret_list[:,0], 75), np.percentile(regret_list[:,0], 90), np.percentile(regret_list[:,0], 95),
                                            np.mean(regret_list[:,1]), np.percentile(regret_list[:,1], 5), np.percentile(regret_list[:,1], 10), np.percentile(regret_list[:,1], 25), np.median(regret_list[:,1]), np.percentile(regret_list[:,1], 75), np.percentile(regret_list[:,1], 90), np.percentile(regret_list[:,1], 95),
                                            np.mean(regret_list[:,2]), np.percentile(regret_list[:,2], 5), np.percentile(regret_list[:,2], 10), np.percentile(regret_list[:,2], 25), np.median(regret_list[:,2]), np.percentile(regret_list[:,2], 75), np.percentile(regret_list[:,2], 90), np.percentile(regret_list[:,2], 95),
                                            # Sum of overall correlation coefficient and top-k correlation coefficient
                                            np.mean(corr_sum_list[:,0]), np.percentile(corr_sum_list[:,0], 5), np.percentile(corr_sum_list[:,0], 10), np.percentile(corr_sum_list[:,0], 25), np.median(corr_sum_list[:,0]), np.percentile(corr_sum_list[:,0], 75), np.percentile(corr_sum_list[:,0], 90), np.percentile(corr_sum_list[:,0], 95),
                                            np.mean(corr_sum_list[:,1]), np.percentile(corr_sum_list[:,1], 5), np.percentile(corr_sum_list[:,1], 10), np.percentile(corr_sum_list[:,1], 25), np.median(corr_sum_list[:,1]), np.percentile(corr_sum_list[:,1], 75), np.percentile(corr_sum_list[:,1], 90), np.percentile(corr_sum_list[:,1], 95),
                                            np.mean(corr_sum_list[:,2]), np.percentile(corr_sum_list[:,2], 5), np.percentile(corr_sum_list[:,2], 10), np.percentile(corr_sum_list[:,2], 25), np.median(corr_sum_list[:,2]), np.percentile(corr_sum_list[:,2], 75), np.percentile(corr_sum_list[:,2], 90), np.percentile(corr_sum_list[:,2], 95), 
                                            # 
                                            np.mean(NSE_list[:,0]), np.percentile(NSE_list[:,0], 5), np.percentile(NSE_list[:,0], 10), np.percentile(NSE_list[:,0], 25), np.median(NSE_list[:,0]), np.percentile(NSE_list[:,0], 75), np.percentile(NSE_list[:,0], 90), np.percentile(NSE_list[:,0], 95),
                                            np.mean(NSE_list[:,1]), np.percentile(NSE_list[:,1], 5), np.percentile(NSE_list[:,1], 10), np.percentile(NSE_list[:,1], 25), np.median(NSE_list[:,1]), np.percentile(NSE_list[:,1], 75), np.percentile(NSE_list[:,1], 90), np.percentile(NSE_list[:,1], 95),
                                            np.mean(NSE_list[:,2]), np.percentile(NSE_list[:,2], 5), np.percentile(NSE_list[:,2], 10), np.percentile(NSE_list[:,2], 25), np.median(NSE_list[:,2]), np.percentile(NSE_list[:,2], 75), np.percentile(NSE_list[:,2], 90), np.percentile(NSE_list[:,2], 95), 
                                            ])
                                           

    ########################################################################################################################################################################
    del mlp, optimizer, loss_tr_list, loss_vl_list, model_state, nse_sv

# Write the results to a text file
write_df = pd.DataFrame(write_data, columns = 
                        ['epoch', 'hidden_dims', 'learning_rate', 'batch_size', 'loss_eval_5', 'loss_eval_10', 'loss_eval_25', 'loss_eval_50', 'loss_eval_75', 'loss_eval_90', 'loss_eval_95', 

                         'scorr_nse_mean', 'scorr_nse_5', 'scorr_nse_10', 'scorr_nse_25', 'scorr_nse_50', 'scorr_nse_75', 'scorr_nse_90', 'scorr_nse_95', 
                         'scorr_nsae_mean', 'scorr_nsae_5', 'scorr_nsae_10', 'scorr_nsae_25', 'scorr_nsae_50', 'scorr_nsae_75', 'scorr_nsae_90', 'scorr_nsae_95',
                         'scorr_kge_mean', 'scorr_kge_5', 'scorr_kge_10', 'scorr_kge_25', 'scorr_kge_50', 'scorr_kge_75', 'scorr_kge_90', 'scorr_kge_95',

                         'scorrtopk_nse_mean', 'scorrtopk_nse_5', 'scorrtopk_nse_10', 'scorrtopk_nse_25', 'scorrtopk_nse_50', 'scorrtopk_nse_75', 'scorrtopk_nse_90', 'scorrtopk_nse_95', 
                         'scorrtopk_nsae_mean', 'scorrtopk_nsae_5', 'scorrtopk_nsae_10', 'scorrtopk_nsae_25', 'scorrtopk_nsae_50', 'scorrtopk_nsae_75', 'scorrtopk_nsae_90', 'scorrtopk_nsae_95',
                         'scorrtopk_kge_mean', 'scorrtopk_kge_5', 'scorrtopk_kge_10', 'scorrtopk_kge_25', 'scorrtopk_kge_50', 'scorrtopk_kge_75', 'scorrtopk_kge_90', 'scorrtopk_kge_95',

                         'regrettopk_nse_mean', 'regrettopk_nse_5', 'regrettopk_nse_10', 'regrettopk_nse_25', 'regrettopk_nse_50', 'regrettopk_nse_75', 'regrettopk_nse_90', 'regrettopk_nse_95', 
                         'regrettopk_nsae_mean', 'regrettopk_nsae_5', 'regrettopk_nsae_10', 'regrettopk_nsae_25', 'regrettopk_nsae_50', 'regrettopk_nsae_75', 'regrettopk_nsae_90', 'regrettopk_nsae_95',
                         'regrettopk_kge_mean', 'regrettopk_kge_5', 'regrettopk_kge_10', 'regrettopk_kge_25', 'regrettopk_kge_50', 'regrettopk_kge_75', 'regrettopk_kge_90', 'regrettopk_kge_95',

                         'scorrsum_nse_mean', 'scorrsum_nse_5', 'scorrsum_nse_10', 'scorrsum_nse_25', 'scorrsum_nse_50', 'scorrsum_nse_75', 'scorrsum_nse_90', 'scorrsum_nse_95', 
                         'scorrsum_nsae_mean', 'scorrsum_nsae_5', 'scorrsum_nsae_10', 'scorrsum_nsae_25', 'scorrsum_nsae_50', 'scorrsum_nsae_75', 'scorrsum_nsae_90', 'scorrsum_nsae_95',
                         'scorrsum_kge_mean', 'scorrsum_kge_5', 'scorrsum_kge_10', 'scorrsum_kge_25', 'scorrsum_kge_50', 'scorrsum_kge_75', 'scorrsum_kge_90', 'scorrsum_kge_95',

                         'nse_nse_mean', 'nse_nse_5', 'nse_nse_10', 'nse_nse_25', 'nse_nse_50', 'nse_nse_75', 'nse_nse_90', 'nse_nse_95', 
                         'nse_nsae_mean', 'nse_nsae_5', 'nse_nsae_10', 'nse_nsae_25', 'nse_nsae_50', 'nse_nsae_75', 'nse_nsae_90', 'nse_nsae_95',
                         'nse_kge_mean', 'nse_kge_5', 'nse_kge_10', 'nse_kge_25', 'nse_kge_50', 'nse_kge_75', 'nse_kge_90', 'nse_kge_95'])

filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_29_Jul_2026/ranking_model/hyperparameter_optimization', f'MLP_metrics_hyperparameter_tuning_results_phase_{phase}_save_{save_no}.txt')
write_df.to_csv(filename, index=False)