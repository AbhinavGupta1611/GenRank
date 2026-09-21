"""
hyperparameter tuning and model training code for MLP ensemble

Author: Abhinav Gupta (Created: 25 Mar 2026)
"""

import os
from matplotlib.pylab import seed
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from geomloss import SamplesLoss

from NN.MLP_ensemble import MLP_ensemble, ConditionalGenerator, CustomData, SubsetSampler, train_mod, test_mod, train_mod_new, computeNSE, ensemble_test_mod, ensemble_test_mod_new, customLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

###############################################################################################################################
# Code for hyperparameter tuning
"""
max_epochs = 500     # Number of epochs (500 epochs in phase 1 and 2000 epochs in phase 2)
nseeds = 1      # Number of seeds
hidden_dims_list = [[512, 256, 128], [1024, 512, 256],
                    [512, 256, 128, 64], [1024, 512, 256, 128], 
                    [2048, 1024, 512, 256, 128], [4096, 2048, 1024, 512, 256, 128]]       # hidden dimension, number of layers, output dimension
lrate_list = [10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3)]    # Learning rate
N_list = [2**6]     # batch size
blur_list = [0.001, 0.01, 0.1]     # blur parameter for Sinkhorn loss
dropout_p_list = [0.0]     # dropout probability
latent_dim_list = [32, 64, 128, 256, 512]
spread_weight_list  = [0.0, 0.1, 0.5, 1.0] # Weight of the spread loss relative to the geometric loss in the custom loss function
n_ensemble = 600
"""
loss_fn_eval = customLoss(p = 2, sigma = 0.01, spread_weight = 0)
weight_expo = 1.0

save_no = '1'
phase = '100'

pred_indices = range(31,57)   # Indices of predictor variables in the data array    ########################################################################################
y_indices = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12,\
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27, 28, 29])
weights_index = 30
output_dim = len(y_indices)

train_frac = 0.70
val_frac = 2/7

# Parameter bounds
ParBounds = [
    [0.1, 1.74],  # PT alpha (1)
    [0.1, 5],   # %SCF (2)
    [-1, 3],    # %PXTEMP (3)
    [0, 1],     # %TTI (4)
    [0.8, 3],   # %MFMAX (5) 
    [0.01, 0.79], # %MFMIN (6)
    [0.01, 0.40], # %UADJ (7)
    [0.01, 1],    # %TIPM (9)
    [0.01, 0.4],  # %PLWHC (10)
    [0.04, 0.4],  # %NMF (11)
    [0.01, 0.5],  # %DAYGM (12)
    [1, 800],  #uztwm (13)
    [1, 800], #uzfwm (14)
    [1, 800], #lztwm (15)
    [1, 1000], #lzfpm (16)
    [1, 1000], #lzfsm (17)
    [0.1, 0.7], #uzk (18)
    [0.00001, 0.025], #lzpk (19)
    [0.001, 0.25], #lzsk (20)
    [1, 250], #zperc (21)
    [0, 6], #rexp (22)
    [0, 1], #pfree (23)
    [0, 1], #adimp (25)
    [0, 1], # %rserv (28)
    [0.01, 5], # % alpha - IUH parameter (29)
    [0.001, 10]  # % beta  - IUH parameter (30)
]

ParBounds = np.array(ParBounds)
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
####################################################################

# Collate data in one dataframe
model_data = pd.DataFrame()
for basin in basin_list:

    # Read parameter data
    filename = os.path.join(results_dir, 'DREAM_results', 'param_top_10000_{}.txt'.format(basin))
    param_df = pd.read_csv(filename, header = None, delimiter=',')

    NSE = param_df[30].values
    NSAE = param_df[31].values
    KGE = param_df[32].values

    """
    param_df_1 = param_df[param_df[30] >= np.nanquantile(NSE, 1-200/param_df.shape[0])]
    param_df_2 = param_df[param_df[31] >= np.nanquantile(NSAE, 1-200/param_df.shape[0])]
    param_df_3 = param_df[param_df[32] >= np.nanquantile(KGE, 1-200/param_df.shape[0])]

    param_df = pd.concat([param_df_1, param_df_2, param_df_3], axis = 0)
    param_df = param_df.drop_duplicates().reset_index(drop=True)
    """

    param_df['score'] = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) + (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) + (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
    param_df = param_df[param_df['score'] >= np.nanquantile(param_df['score'].values, 1-600/param_df.shape[0])]
    param_df = param_df.drop(columns = 'score')

    NSE = param_df[30].values
    NSAE = param_df[31].values
    KGE = param_df[32].values

    #NSE = (NSE - np.min(NSE)) / (np.max(NSE) - np.min(NSE) + 10**(-6))
    #NSAE = (NSAE - np.min(NSAE)) / (np.max(NSAE) - np.min(NSAE) + 10**(-6))
    #KGE = (KGE - np.min(KGE)) / (np.max(KGE) - np.min(KGE) + 10**(-6))

    param_df = param_df.drop(columns = [ii for ii in range(30, 46)])

    # Compute weights
    score = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) + (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) + (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
    weights = np.exp(weight_expo * score)
    weights = weights / sum(weights)

    param_df['weights'] = weights

    stat_tmp = static_df[static_df['gauge_id'] == basin].drop('gauge_id', axis=1)
    stat_tmp = pd.concat([stat_tmp]*param_df.shape[0], ignore_index=True)
    
    data_tmp = pd.concat([param_df.reset_index(drop=True), stat_tmp.reset_index(drop=True)], axis=1)
    data_tmp['basin'] = [basin]*param_df.shape[0]
    model_data = pd.concat([model_data, data_tmp], axis=0)

# Remove basins with NaN parameters
model_data = model_data[~model_data[0].isna()]

# Training, validation, and testing basins
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

# Craete a 3-D arary of train and test data with dimensions (n_basins, n_samples_per_basin, n_features)
train_data_list, val_data_list = [], []
for basin in train_basins:
    data_tmp = train_data[train_data['basin'] == basin].drop('basin', axis=1).values
# Add padding if number of samples for a basin is less than 200
    if data_tmp.shape[0] < 600:
        padding = np.full((600 - data_tmp.shape[0], data_tmp.shape[1]), np.nan)
        data_tmp = np.vstack([data_tmp, padding])
    train_data_list.append(data_tmp)

for basin in val_basins:
    data_tmp = val_data[val_data['basin'] == basin].drop('basin', axis=1).values
    # Add padding if number of samples for a basin is less than 600
    if data_tmp.shape[0] < 600:
        padding = np.full((600 - data_tmp.shape[0], data_tmp.shape[1]), np.nan)
        data_tmp = np.vstack([data_tmp, padding])
    val_data_list.append(data_tmp)

train_data_list = np.array(train_data_list)
val_data_list = np.array(val_data_list)

###############################################################################################################################
# Code for model training
# Compute mean and standard deviation of each column to be used for standardization
train_data_tmp = np.squeeze(np.concatenate(train_data_list, axis=0))

meanx = np.nanmean(train_data_tmp[:, pred_indices], axis=0)
stdx = np.nanstd(train_data_tmp[:, pred_indices], axis=0)
stdx[stdx==0] = 10**(-6)
meanx = torch.from_numpy(meanx).float()
stdx = torch.from_numpy(stdx).float()

meanx = meanx.to(device)
stdx = stdx.to(device)

# Determine the input dimension
input_dim = len(meanx)

# Standard deviation of target variable
ymean = np.nanmean(train_data_tmp[:, y_indices], axis=0)
ystd = np.nanstd(train_data_tmp[:, y_indices], axis=0) + 10**(-6)

# Scale the parameter bounds
ParBounds_scaled = copy.deepcopy(ParBounds)
ParBounds_scaled[:, 0] = (ParBounds_scaled[:, 0] - ymean) / ystd
ParBounds_scaled[:, 1] = (ParBounds_scaled[:, 1] - ymean) / ystd

# LSTM formatting of training and testing data
train_data_list = torch.from_numpy(train_data_list).float()
val_data_list = torch.from_numpy(val_data_list).float()
ymean = torch.from_numpy(ymean).float()
ystd = torch.from_numpy(ystd).float()
ParBounds_scaled = torch.from_numpy(ParBounds_scaled).float()

train_data_list = train_data_list.to(device)
val_data_list = val_data_list.to(device)
ymean = ymean.to(device)
ystd = ystd.to(device)
ParBounds_scaled = ParBounds_scaled.to(device)

# Instantiate data class
train_dataset = CustomData(train_data_list, meanx, stdx, pred_indices, y_indices, weights_index, ymean, ystd)
val_dataset = CustomData(val_data_list, meanx, stdx, pred_indices, y_indices, weights_index, ymean, ystd)

# These are the optimal hyperparameter found with spread_weight = 0
max_epochs = 600    # Number of epochs (500 epochs in phase 1 and 2000 epochs in phase 2)
nseeds = 1      # Number of seeds
hidden_dims_list = [[1024, 512, 256]]       # hidden dimension, number of layers, output dimension
lrate_list = [10**(-4)]    # Learning rate
N_list = [2**5]     # batch size
blur_list = [0.01]     # blur parameter for Sinkhorn loss
dropout_p_list = [0.0]     # dropout probability
latent_dim_list = [1024]
spread_weight_list  = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] # Weight of the spread loss relative to the geometric loss in the custom loss function
n_ensemble = 600

loss_vl_list, std_ratio_val_list = [], []
for seed in range(nseeds):
    for hidden_dims in hidden_dims_list:
        for lrate in lrate_list:
            for N in N_list:
                for blur in blur_list:
                    for dropout_p in dropout_p_list:
                            for latent_dim in latent_dim_list:
                                for spread_weight in spread_weight_list:

                                    print(f"Hidden dims: {hidden_dims}, Learning rate: {lrate}, batch_size: {N}, Latent dimension: {latent_dim}, spread_weight: {spread_weight}")
                                    torch.manual_seed(seed)
                                    np.random.seed(seed)
                                    random.seed(seed)

                                    # Create data loaders
                                    train_dataloader = DataLoader(train_dataset, batch_size = N, shuffle=True)
                                    val_dataloader = DataLoader(val_dataset, batch_size = N, shuffle=False)

                                    # Instantiate the model class
                                    mlp = ConditionalGenerator(input_dim, latent_dim, hidden_dims, output_dim, ParBounds_scaled, dropout_p=dropout_p)
                                    mlp.to(device)
                                    
                                    # define loss and optimizer
                                    loss_fn = customLoss(p = 2, sigma = blur, spread_weight = spread_weight)
                                    optimizer = torch.optim.Adam(mlp.parameters(), lr = lrate)

                                    # fix the number of epochs and start model training
                                    loss_tr_list = []
                                    for t in range(max_epochs):
                                        loss_tr, state = train_mod_new(train_dataloader, mlp, loss_fn, optimizer, latent_dim, n_ensemble)
                                        loss_tr_list.append(loss_tr)
                                        #print(loss_tr)

                                        if (t+1) % 20 == 0:
                                            print(f"Epochs: {t+1} -- ")
                                            yobs, ypred, weights_yobs = ensemble_test_mod_new(val_dataloader, mlp, latent_dim, n_ensemble)
                                            loss_val, loss_list = loss_fn_eval(ypred, yobs, weights_yobs)
                                            loss_list = torch.tensor(loss_list)
                                            loss_vl_list.append([hidden_dims, lrate, N, blur, t+1, latent_dim, dropout_p, spread_weight, n_ensemble, loss_val.item(), 
                                                                torch.quantile(loss_list, 5/100).item(), torch.quantile(loss_list, 10/100).item(), 
                                                                torch.quantile(loss_list, 25/100).item(), torch.quantile(loss_list, 50/100).item(), 
                                                                torch.quantile(loss_list, 75/100).item(), torch.quantile(loss_list, 90/100).item(), 
                                                                torch.quantile(loss_list, 95/100).item()])

                                            std_ratio_tmp = []
                                            for ii in range(yobs.shape[0]):
                                                std_obs = torch.std(yobs[ii, ~torch.isnan(yobs[ii, :, 0]), :], dim=0)
                                                std_pred = torch.std(ypred[ii,:,:], dim=0)
                                                avg_ratio = torch.mean(std_pred/std_obs)
                                                std_ratio_tmp.append(avg_ratio.item())
                                            std_ratio_val_list.append([hidden_dims, lrate, N, blur, t+1, latent_dim, dropout_p, spread_weight, n_ensemble, np.nanmean(std_ratio_tmp).item(),
                                                                np.nanquantile(std_ratio_tmp, 5/100).item(), np.nanquantile(std_ratio_tmp, 10/100).item(), 
                                                                np.nanquantile(std_ratio_tmp, 25/100).item(), np.nanquantile(std_ratio_tmp, 50/100).item(), 
                                                                np.nanquantile(std_ratio_tmp, 75/100).item(), np.nanquantile(std_ratio_tmp, 90/100).item(), 
                                                                np.nanquantile(std_ratio_tmp, 95/100).item()])
                                    del mlp, optimizer, loss_fn, loss_tr_list

# Write loss data to a textfile
loss_vl_df = pd.DataFrame(loss_vl_list, columns = ['hidden_dims', 'lrate', 'N', 'blur', 'epochs', 'latent_dim', 'dropout_p', 'spread_weight', 'n_ensemble', 'loss_val_combined_mean', 'loss_val_05', 'loss_val_10', 'loss_val_25', 'loss_val_50', 'loss_val_75', 'loss_val_90', 'loss_val_95'])
loss_vl_df.to_csv(os.path.join(results_dir, 'MLP_results/equifinality_quantification_29_Jul_2026/equifinality_cloud/hyperparameter_optimization', f'MLP_hyperparameter_tuning_results_weight_{weight_expo}_phase_{phase}_{save_no}.txt'), index=False, sep=',')

std_ratio_vl_df = pd.DataFrame(std_ratio_val_list, columns = ['hidden_dims', 'lrate', 'N', 'blur', 'epochs', 'latent_dim', 'dropout_p', 'spread_weight', 'n_ensemble', 'std_ratio_val_mean', 'std_ratio_val_05', 'std_ratio_val_10', 'std_ratio_val_25', 'std_ratio_val_50', 'std_ratio_val_75', 'std_ratio_val_90', 'std_ratio_val_95'])
std_ratio_vl_df.to_csv(os.path.join(results_dir, 'MLP_results/equifinality_quantification_29_Jul_2026/equifinality_cloud/hyperparameter_optimization', f'MLP_hyperparameter_tuning_std_results_weight_{weight_expo}_phase_{phase}_{save_no}.txt'), index=False, sep=',')


""""""
