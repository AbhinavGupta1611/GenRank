"""
Script to develop a neural network for regionalization with performance metrics as objective functions

Author: Abhinav Gupta (Created: 16 Mar 2026)
"""

import os
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

from NN.MLP_ensemble import MLP_ensemble, ConditionalGenerator, CustomData, SubsetSampler, train_mod, test_mod, train_mod_new, computeNSE, ensemble_test_mod, ensemble_test_mod_new, ensemble_test_mod_new_chunk, customLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs = 520  # Number of epochs
nseeds = 1      # Number of seeds
hidden_dims = [1024, 512, 256]       # hidden dimension, number of layers, output dimension
lrate = 10**(-4)    # Learning rate
N = 2**5     # batch size
#loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.5)      # Specify the loss function (nn.L2Loss for NMSE; nn.L1Loss for NMAE)
loss_fn = customLoss(p = 2, sigma = 0.01, spread_weight = 3.00)
n_ensemble = 10000
latent_dim = 1024
dropout_p = 0.0 

weight_expo = 1.0

pred_indices = range(31,57)   # Indices of predictor variables in the data array    ########################################################################################
y_indices = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12,\
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27, 28, 29])
weights_index = 30
output_dim = len(y_indices)

train_frac = 0.70
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

# Parameter bounds
ParBounds = [
    [0.1, 2.00],  # PT alpha (1)
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

    param_df_old = pd.concat([param_df_1, param_df_2, param_df_3], axis = 0)
    param_df_old = param_df_old.drop_duplicates().reset_index(drop=True)
    """
    param_df['score'] = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) + (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) + (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
    param_df = param_df[param_df['score'] >= np.nanquantile(param_df['score'].values, 1-600/param_df.shape[0])]
    param_df = param_df.drop(columns = 'score')

    """
    plt.figure(figsize = (18, 6))
    plt.subplot(1,3,1)
    plt.ecdf(param_df_1[32], label = 'NSE')
    plt.ecdf(param_df_2[32], label = 'NSAE')

    plt.subplot(1,3,2)
    plt.scatter(param_df_1[32], param_df_1[30], s=40)
    plt.scatter(param_df_2[32], param_df_2[30], s=10)

    plt.subplot(1,3,3)
    plt.scatter(param_df_1[0], param_df_1[1], s=40)
    plt.scatter(param_df_2[0], param_df_2[1], s=10)
    plt.show()
    """

    NSE = param_df[30].values
    NSAE = param_df[31].values
    KGE = param_df[32].values

    param_df = param_df.drop(columns = [ii for ii in range(30, 46)])

    # Compute weights
    score = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) +\
          (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) +\
              (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
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

# Training and testing basins
num_basins = len(basin_list)
all_inds = np.arange(num_basins)
np.random.seed(0)
np.random.shuffle(all_inds)

ntrain = int(train_frac * num_basins)

train_basin_inds = all_inds[:ntrain]
test_basin_inds  = all_inds[ntrain:]

train_basins = [basin_list[ii] for ii in train_basin_inds]
test_basins = [basin_list[ii] for ii in test_basin_inds]

train_data = model_data[model_data['basin'].isin(train_basins)]
test_data = model_data[model_data['basin'].isin(test_basins)]

# Craete a 3-D arary of train and test data with dimensions (n_basins, n_samples_per_basin, n_features)
train_data_list, test_data_list = [], []
for basin in train_basins:
    data_tmp = train_data[train_data['basin'] == basin].drop('basin', axis=1).values
    # Add padding if number of samples for a basin is less than 600
    if data_tmp.shape[0] < 600:
        padding = np.full((600 - data_tmp.shape[0], data_tmp.shape[1]), np.nan)
        data_tmp = np.vstack([data_tmp, padding])
    train_data_list.append(data_tmp)

for basin in test_basins:
    data_tmp = test_data[test_data['basin'] == basin].drop('basin', axis=1).values
    # Add padding if number of samples for a basin is less than 600
    if data_tmp.shape[0] < 600:
        padding = np.full((600 - data_tmp.shape[0], data_tmp.shape[1]), np.nan)
        data_tmp = np.vstack([data_tmp, padding])
    test_data_list.append(data_tmp)

train_data_list = np.array(train_data_list)
test_data_list = np.array(test_data_list)

#test_performance_metric = test_data[[30, 31, 32]].values

###############################################################################################################################
# Code for model training

# Compute mean and standard deviation of each column to be used for standardization
train_data_tmp = np.squeeze(np.concatenate(train_data_list, axis=0))

meanx = np.nanmean(train_data_tmp[:, pred_indices], axis=0)
stdx = np.nanstd(train_data_tmp[:, pred_indices], axis=0)
stdx[stdx==0] = 10**(-6)
meanx = torch.from_numpy(meanx).float()
stdx = torch.from_numpy(stdx).float()

#meanx = meanx.to(device)
#stdx = stdx.to(device)

# Determine the input dimension
input_dim = len(meanx)

# Standard deviation of target variable
ystd = np.nanstd(train_data_tmp[:, y_indices], axis=0) + 10**(-6)
ymean = np.nanmean(train_data_tmp[:, y_indices], axis=0)

# Scale the parameter bounds
ParBounds_scaled = copy.deepcopy(ParBounds)
ParBounds_scaled[:, 0] = (ParBounds_scaled[:, 0] - ymean) / ystd
ParBounds_scaled[:, 1] = (ParBounds_scaled[:, 1] - ymean) / ystd

# LSTM formatting of training and testing data
train_data_list = torch.from_numpy(train_data_list).float()
test_data_list = torch.from_numpy(test_data_list).float()
ymean = torch.from_numpy(ymean).float()
ystd = torch.from_numpy(ystd).float()
ParBounds_scaled = torch.from_numpy(ParBounds_scaled).float()

"""
train_data_list = train_data_list.to(device)
test_data_list = test_data_list.to(device)
ymean = ymean.to(device)
ystd = ystd.to(device)
ParBounds_scaled = ParBounds_scaled.to(device)
"""
ParBounds_scaled = ParBounds_scaled.to(device)

# Create data loaders
train_dataset = CustomData(train_data_list, meanx, stdx, pred_indices, y_indices, weights_index, ymean, ystd)
test_dataset = CustomData(test_data_list, meanx, stdx, pred_indices, y_indices, weights_index, ymean, ystd)
train_dataloader = DataLoader(train_dataset, batch_size = N, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = N, shuffle=False)
train_dataloader_noshuffle = DataLoader(train_dataset, batch_size = N, shuffle=False)
#########################################################################################################################
# Model training
"""
nse_ts_list, ypred_list, yobs_list = [], [], []
ypred_ensemble_list = []

for seed in range(nseeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # instantiate the model class
    mlp = ConditionalGenerator(input_dim, latent_dim, hidden_dims, output_dim, ParBounds_scaled, dropout_p=dropout_p)
    mlp.to(device)

    # Read model
    filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3', f'final_model_state_{seed}.pth')
    state = torch.load(filename)
    mlp.load_state_dict(state)
    
    # define loss and optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr = lrate)

    # fix the number of epochs and start model training
    loss_tr_list, loss_tst_list, model_state, nse_sv = [], [], [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss_tr, state = train_mod_new(train_dataloader, mlp, loss_fn, optimizer, latent_dim, n_ensemble)
        model_state.append(copy.deepcopy(state))
        loss_tr_list.append(loss_tr)
        print(loss_tr)

        #yobs, ypred, weights_yobs = ensemble_test_mod_new(test_dataloader, mlp, latent_dim, n_ensemble)
        #loss_test, loss_list = loss_fn(ypred, yobs, weights_yobs)
        #loss_tst_list.append([loss_test.item(), np.percentile(loss_list, 10), np.percentile(loss_list, 25), np.percentile(loss_list, 50), np.percentile(loss_list, 75), np.percentile(loss_list, 90)])

    yobs_train, ypred_train, weights_train = ensemble_test_mod_new(train_dataloader_noshuffle, mlp, latent_dim, n_ensemble)
    yobs, ypred, weights_yobs = ensemble_test_mod_new(test_dataloader, mlp, latent_dim, n_ensemble)

    # Remove the standardization to get the predictions in the original scale
    ypred, yobs = ypred.to('cpu'), yobs.to('cpu')
    ypred = torch.mul(ypred, ystd.view(1, 1, -1)) + ymean.view(1, 1, -1)
    yobs = torch.mul(yobs, ystd.view(1, 1, -1)) + ymean.view(1, 1, -1)

    ypred_train, yobs_train = ypred_train.to('cpu'), yobs_train.to('cpu')
    ypred_train = torch.mul(ypred_train, ystd.view(1, 1, -1)) + ymean.view(1, 1, -1)
    yobs_train = torch.mul(yobs_train, ystd.view(1,1,-1)) + ymean.view(1, 1, -1)
    
    # Save the final model state
    filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3', f'final_model_state_{seed}.pth')
    torch.save(mlp.state_dict(), filename)
    del mlp, optimizer, loss_tr_list, loss_tst_list, model_state, nse_sv

    yobs = yobs.cpu().numpy()
    ypred = ypred.cpu().numpy()
    yobs_train = yobs_train.cpu().numpy()
    ypred_train = ypred_train.cpu().numpy()

    # Save the predictions of the ungauged (test) basins
    for ii in range(len(test_basins)):
        
        obs = yobs[ii,:,:]
        pred = ypred[ii,:,:]
        basin = test_basins[ii]

        filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3', f'ungauged_obs_param_ensemble_{basin}.txt')
        np.savetxt(filename, obs)

        filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3', f'ungauged_pred_param_ensemble_{basin}.txt')
        np.savetxt(filename, pred)
        
        plt.scatter(obs[:,24], obs[:,25])
        plt.scatter(pred[:,24], pred[:,25])
        plt.show()
        
    # Save the predictions of the gauged (training) basins
    for ii in range(len(train_basins)):
        
        obs = yobs_train[ii,:,:]
        pred = ypred_train[ii,:,:]
        basin = train_basins[ii]

        filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3', f'gauged_obs_param_ensemble_{basin}.txt')
        np.savetxt(filename, obs)

        filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3', f'gauged_pred_param_ensemble_{basin}.txt')
        np.savetxt(filename, pred)

    del yobs, yobs_train, ypred, ypred_train, weights_yobs, weights_train
"""
#########################################################################################################################

# instantiate the model class

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

mlp = ConditionalGenerator(input_dim, latent_dim, hidden_dims, output_dim, ParBounds_scaled, dropout_p=dropout_p)
mlp.to(device)

# Read model
filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3', f'final_model_state_0.pth')
state = torch.load(filename)
mlp.load_state_dict(state)

yobs, ypred, weights_yobs = ensemble_test_mod_new_chunk(test_dataloader, mlp, latent_dim, n_ensemble)

# Remove the standardization to get the predictions in the original scale
ypred = torch.mul(ypred.cpu(), ystd.view(1, 1, -1)) + ymean.view(1, 1, -1)
yobs = torch.mul(yobs.cpu(), ystd.view(1, 1, -1)) + ymean.view(1, 1, -1)


yobs = yobs.cpu().numpy()
ypred = ypred.cpu().numpy()

# Save the predictions of the ungauged (test) basins
for ii in range(len(test_basins)):
    
    obs = yobs[ii,:,:]
    pred = ypred[ii,:,:]
    basin = test_basins[ii]

    """
    plt.figure(figsize = (14, 6))
    plt.subplot(1,2,1)
    plt.scatter(pred[:,0], pred[:,1], s=10)
    plt.scatter(obs[:,0], obs[:,1])

    plt.subplot(1,2,2)
    plt.scatter(pred[:,24], pred[:,25], s=10)
    plt.scatter(obs[:,24], obs[:,25])
    plt.show()
    """
    
    filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_{weight_expo}_ratio_3/ensemble_10000', 'ungauged_pred_param_ensemble_{}.txt'.format(basin))
    np.savetxt(filename, pred)