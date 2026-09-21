"""

Script to train an MLP to estimate the performance metrics given the model parameters and the static attributes

Author: Abhinav Gupta (Created: 3 Apr 2026)
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
 
from NN.MLP_metrics import MLP_metrics, CustomData, SubsetSampler, WeightedMSELoss, WeightedL1Loss, BasinBatchSampler, CombinedLoss, weights_for_target_ess, train_mod, test_mod, computeNSE, compute_regret_at_k, train_mod_accumulated

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs = 180    # Number of epochs
nseeds = 8      # Number of seeds
hidden_dims = [1024, 512, 256, 128, 64]       # hidden dimension, number of layers, output dimension
lrate = 10**(-3)    # Learning rate
N = 1024     # batch size
loss_fn = nn.L1Loss()

weight_expo = 0
dropout = 0.0

pred_indices = range(8,60)      ###############################################################################
y_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
#y_indices = np.array([2])
param_cols = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12,\
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
output_dim = len(y_indices)
weight_index = 60

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
    # filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_updated/ranking_model/observed_data', 'param_top_1000_{}.txt'.format(basin))
    # param_df = pd.read_csv(filename, header = None, delimiter=',')
    #param_df = param_df.drop(columns = [33, 34, 35])

    # Read parameter data
    filename = os.path.join(results_dir, 'DREAM_results', 'param_top_10000_{}.txt'.format(basin))
    param_df = pd.read_csv(filename, header = None, delimiter=',')

    NSE = param_df[30].values
    NSAE = param_df[31].values
    KGE = param_df[32].values

    """
    param_df_1 = param_df[param_df[30] >= np.quantile(NSE, 1-1000/param_df.shape[0])]
    param_df_2 = param_df[param_df[31] >= np.quantile(NSAE, 1-1000/param_df.shape[0])]
    param_df_3 = param_df[param_df[32] >= np.quantile(KGE, 1-1000/param_df.shape[0])]

    param_df = pd.concat([param_df_1, param_df_2, param_df_3], axis = 0)
    param_df = param_df.drop_duplicates().reset_index(drop=True)
    """

    param_df['score'] = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) + (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) + (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
    param_df = param_df[param_df['score'] >= np.nanquantile(param_df['score'].values, 1-6000/param_df.shape[0])]
    param_df = param_df.drop(columns = 'score')

    # Add the 1000 sets from high-performance region
    filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_28_Aug_2026/ranking_model/high_performance_samples', f'high_performance_sets_1000_{basin}.txt')
    high_param_df = pd.read_csv(filename, header = None, delimiter=',')
    param_df = pd.concat([param_df, high_param_df], axis = 0)

    # Only keep the parameters that are used as predictors in the model
    param_df = param_df[param_cols]

    # Compute weights
    """
    NSE = param_df[30].values
    NSAE = param_df[31].values
    KGE = param_df[32].values
    score = (NSE - np.nanmin(NSE))/(np.nanmax(NSE) - np.nanmin(NSE)) + (NSAE - np.nanmin(NSAE))/(np.nanmax(NSAE) - np.nanmin(NSAE)) + (KGE - np.nanmin(KGE))/(np.nanmax(KGE) - np.nanmin(KGE))
    weights = np.exp(weight_expo * score)
    weights = weights/np.nansum(weights)
    weight_df = pd.DataFrame({'weight': weights})
    """
    weight_df = pd.DataFrame({'weight': [1.0] * param_df.shape[0]})

    stat_tmp = static_df[static_df['gauge_id'] == basin].drop('gauge_id', axis=1)
    stat_tmp = pd.concat([stat_tmp]*param_df.shape[0], ignore_index=True)
    
    data_tmp = pd.concat([param_df.reset_index(drop=True), stat_tmp.reset_index(drop=True), weight_df.reset_index(drop=True)], axis=1)
    data_tmp['basin'] = [basin]*param_df.shape[0]

    model_data = pd.concat([model_data, data_tmp], axis=0)

# Remove basins with NaN parameters
# model_data = model_data[~model_data[0].isna()]
model_data = model_data.dropna(subset=[30, 31, 32, 33, 34, 35, 36, 37])

# Put three performance metrics (cols: 30, 31, 32) as first three columns
model_data = model_data[[30, 31, 32, 33, 34, 35, 36, 37] + [col for col in model_data.columns if col not in [30, 31, 32, 33, 34, 35, 36, 37]]]

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

train_basin_seq = train_data['basin'].values
test_basin_seq = test_data['basin'].values
train_data = train_data.drop('basin', axis=1).values
test_data = test_data.drop('basin', axis=1).values

###############################################################################################################################
# Read the predicted parameters
pred_param_data = pd.DataFrame()
for basin in test_basins:

    # Read parameter data
    filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_1.0_ratio_3/ensemble_10000', f'ungauged_pred_param_ensemble_{basin}.txt')
    param_df = pd.read_csv(filename, header = None, delimiter='\s+', engine='c')

    # Read NSE ensemble
    filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_28_Aug_2026/equifinality_cloud/weight_expo_1.0_ratio_3/ensemble_10000', f'NSE_ensemble_cal_ungauged_{basin}.txt')
    nse_df = pd.read_csv(filename, header = None)
    nse_df = nse_df.rename(columns={0: 'NSE', 1: 'NSAE', 2: 'KGE'})

    stat_tmp = static_df[static_df['gauge_id'] == basin].drop('gauge_id', axis=1)
    stat_tmp = pd.concat([stat_tmp]*param_df.shape[0], ignore_index=True)

    # Add weights equal to 1
    weight_df = pd.DataFrame({'weight': [1.0] * param_df.shape[0]})
    
    data_tmp = pd.concat([nse_df.reset_index(drop=True), param_df.reset_index(drop=True), stat_tmp.reset_index(drop=True), weight_df.reset_index(drop=True)], axis=1)
    data_tmp['basin'] = [basin]*param_df.shape[0]
    pred_param_data = pd.concat([pred_param_data, data_tmp], axis=0)

# Remove basins with NaN parameters
pred_param_data = pred_param_data[~pred_param_data['NSE'].isna()]

test_basin_pred_seq = pred_param_data['basin'].values
pred_param_data = pred_param_data.drop('basin', axis=1).values
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
test_data = torch.from_numpy(test_data).float()
pred_param_data = torch.from_numpy(pred_param_data).float()
ymean = torch.from_numpy(ymean).float()
ystd = torch.from_numpy(ystd).float()

train_dataset = CustomData(train_data, meanx, stdx, pred_indices, y_indices, weight_index, ymean, ystd)
test_dataset = CustomData(test_data, meanx, stdx, pred_indices, y_indices, weight_index, ymean, ystd)
pred_param_dataset = CustomData(pred_param_data, meanx, stdx, pred_indices, y_indices, weight_index, ymean, ystd)


# Data loaders for random shuffling
train_dataloader = DataLoader(train_dataset, batch_size = N, shuffle=True, pin_memory=True)
train_dataloader_noshuffle = DataLoader(train_dataset, batch_size = N, shuffle=False)

# Data loaders for testing
test_dataloader = DataLoader(test_dataset, batch_size = N, shuffle=False)
pred_param_dataloader = DataLoader(pred_param_dataset, batch_size = N, shuffle=False)

############################################################################################################################################################

# Model training
"""
nse_ts_list, ypred_list, yobs_list, ypred_train_list, yobs_train_list, X_train_list, X_list = [], [], [], [], [], [], []

for seed in range(nseeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # instantiate the model class
    mlp = MLP_metrics(input_dim, hidden_dims, output_dim, dropout_p=dropout)
    mlp.to(device)

    # Define loss and optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr = lrate)

    # fix the number of epochs and start model training
    ########################################################################################################################################################################
    loss_tr_list, loss_vl_list, model_state, nse_sv = [], [], [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss_tr, state = train_mod(train_dataloader, mlp, loss_fn, optimizer)
        #loss_tr, state = train_mod_accumulated(train_dataloader, mlp, loss_fn, optimizer)
        #model_state.append(copy.deepcopy(state))
        loss_tr_list.append(loss_tr)
        print(loss_tr)

    # Save the model state
    filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_28_Aug_2026/ranking_model'.format(weight_expo), 'model_state_seed_{}.pth'.format(seed))
    torch.save(mlp.state_dict(), filename)
    
    ########################################################################################################################################################################
    # Predict the performance metrics for the gauged basins using the trained model
    _, ypred_train, yobs_train, X_train = test_mod(train_dataloader_noshuffle, mlp)

    # Test the model on the ungauged basins
    _, ypred, yobs, X_test = test_mod(pred_param_dataloader, mlp)

    # Remove the standardization to get the predictions in the original scale
    yobs, ypred = yobs.cpu(), ypred.cpu()
    yobs_train, ypred_train = yobs_train.cpu(), ypred_train.cpu()
    X_train, X_test = X_train.cpu(), X_test.cpu()

    ypred_train = torch.mul(ypred_train, ystd.view(1, -1)) + ymean.view(1, -1)      
    yobs_train = torch.mul(yobs_train, ystd.view(1, -1)) + ymean.view(1, -1) 
    
    ypred = torch.mul(ypred, ystd.view(1, -1)) + ymean.view(1, -1)      
    yobs = torch.mul(yobs, ystd.view(1, -1)) + ymean.view(1, -1)        
    
    X_train = torch.mul(X_train, stdx.view(1, -1)) + meanx.view(1, -1)
    X_test = torch.mul(X_test, stdx.view(1, -1)) + meanx.view(1, -1)

    # Save the results in a list
    ypred_train_list.append(ypred_train.numpy())
    yobs_train_list.append(yobs_train.numpy())
    X_train_list.append(X_train.numpy())
    ypred_list.append(ypred.numpy())
    yobs_list.append(yobs.numpy())
    X_list.append(X_test.numpy())

    del mlp, optimizer, loss_tr_list, loss_vl_list, model_state, nse_sv

# Get the average predictions
ypred_list = np.array(ypred_list)
ypred_final = np.mean(ypred_list, axis=0)
yobs = yobs_list[0]

ypred_train_list = np.array(ypred_train_list)
ypred_train_final = np.mean(ypred_train_list, axis=0)
yobs_train = yobs_train_list[0]

X_test = X_list[0]
X_train = X_train_list[0]

# Write predicted performance metrics along with the predictor variables for the ungauged basins to a text file

NSE_list = []
corr_list = []
for basin in test_basins:
    ind = np.nonzero(test_basin_pred_seq == basin)[0]
    nse = computeNSE(yobs[ind, 2], ypred_final[ind, 2])
    NSE_list.append(nse)

    corr = np.corrcoef(yobs[ind, 2], ypred_final[ind, 2])[0,1]
    spearman_corr, _ = spearmanr(yobs[ind, 2], ypred_final[ind, 2])
    corr_list.append(corr)

    # Write data to a text file
    write_df = pd.DataFrame({'NSE_actual': yobs[ind, 0], 'NSE_mlp_pred': ypred_final[ind, 0], 
                             'NSAE_actual': yobs[ind, 1], 'NSAE_mlp_pred': ypred_final[ind, 1], 
                             'KGE_actual': yobs[ind, 2], 'KGE_mlp_pred': ypred_final[ind, 2],
                             'corr_actual': yobs[ind, 3], 'corr_mlp_pred': ypred_final[ind, 3],
                             'alpha_actual': yobs[ind, 4], 'alpha_mlp_pred': ypred_final[ind, 4],
                             'beta_actual': yobs[ind, 5], 'beta_mlp_pred': ypred_final[ind, 5],
                             'peak_error_actual': yobs[ind, 6], 'peak_error_mlp_pred': ypred_final[ind, 6],
                             '  h': yobs[ind, 7], 'low_error_mlp_pred': ypred_final[ind, 7],
                             'Predicted_Param': list(X_test[ind, :])})
    filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/ranking_model', 'ungauged_performance_metrics_predictability_{}.txt'.format(basin))
    write_df.to_csv(filename, index=False)

# Write predicted performance metrics along with the predictor variables for the gauged basins to a text file
NSE_list = []
corr_list = []
for basin in train_basins:
    ind = np.nonzero(train_basin_seq == basin)[0]
    nse = computeNSE(yobs_train[ind, 1], ypred_train_final[ind, 1])
    NSE_list.append(nse)

    corr = np.corrcoef(yobs_train[ind, 2], ypred_train_final[ind,2])[0,1]
    spearman_corr, _ = spearmanr(yobs_train[ind, 2], ypred_train_final[ind, 2])
    corr_list.append(spearman_corr)

    # Write data to a text file
    write_df = pd.DataFrame({'NSE_actual': yobs_train[ind, 0], 'NSE_mlp_pred': ypred_train_final[ind, 0], 
                             'NSAE_actual': yobs_train[ind, 1], 'NSAE_mlp_pred': ypred_train_final[ind, 1], 
                             'KGE_actual': yobs_train[ind, 2], 'KGE_mlp_pred': ypred_train_final[ind, 2],
                            'corr_actual': yobs_train[ind, 3], 'corr_mlp_pred': ypred_train_final[ind, 3],
                            'alpha_actual': yobs_train[ind, 4], 'alpha_mlp_pred': ypred_train_final[ind, 4],
                            'beta_actual': yobs_train[ind, 5], 'beta_mlp_pred': ypred_train_final[ind, 5],
                            'peak_error_actual': yobs_train[ind, 6], 'peak_error_mlp_pred': ypred_train_final[ind, 6],
                            'low_error_actual': yobs_train[ind, 7], 'low_error_mlp_pred': ypred_train_final[ind, 7],
                             'Predicted_Param': list(X_train[ind, :])})
    filename = os.path.join(results_dir, f'MLP_results/equifinality_quantification_28_Aug_2026/ranking_model', 'gauged_performance_metrics_predictability_{}.txt'.format(basin))
    write_df.to_csv(filename, index=False)
"""
"""
NSE_list = []
corr_list = []
for basin in test_basins:
    ind = np.nonzero(test_basin_pred_seq == basin)[0]
    nse = computeNSE(yobs[ind, 0], ypred_final[ind, 0])
    NSE_list.append(nse)

    corr = np.corrcoef(yobs[ind, 0], ypred_final[ind,0])[0,1]
    corr_list.append(corr)

    # Write data to a text file
    write_df = pd.DataFrame({'KGE_actual': yobs[ind, 0], 'KGE_mlp_pred': ypred_final[ind, 0],
                             'Predicted_Param': list(X_test[ind, :])})
    filename = os.path.join(results_dir, 'ranking_model_experiments/ranking_model_ess_{}_KGE_MAE'.format(ESS), 'ungauged_performance_metrics_predictability_{}.txt'.format(basin))
    write_df.to_csv(filename, index=False)

# Write predicted performance metrics along with the predictor variables for the gauged basins to a text file
NSE_list = []
corr_list = []
for basin in train_basins:
    ind = np.nonzero(train_basin_seq == basin)[0]
    nse = computeNSE(yobs_train[ind, 0], ypred_train_final[ind, 0])
    NSE_list.append(nse) 

    corr = np.corrcoef(yobs_train[ind, 0], ypred_train_final[ind, 0])[0,1]
    corr_list.append(corr)

    # Write data to a text file
    write_df = pd.DataFrame({'KGE_actual': yobs_train[ind, 0], 'KGE_mlp_pred': ypred_train_final[ind, 0], 
                             'Predicted_Param': list(X_train[ind, :])})
    filename = os.path.join(results_dir, 'ranking_model_experiments/ranking_model_ess_{}_KGE_MAE'.format(ESS), 'gauged_performance_metrics_predictability_{}.txt'.format(basin))
    write_df.to_csv(filename, index=False)
"""
##############################################################################################################################################################################################
# Predict the performance using already trained model

nse_ts_list, ypred_list, yobs_list, ypred_train_list, yobs_train_list, X_train_list, X_list = [], [], [], [], [], [], []

for seed in range(nseeds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # instantiate the model class
    mlp = MLP_metrics(input_dim, hidden_dims, output_dim, dropout_p=dropout)
    mlp.to(device)

    # Read the model state
    filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_28_Aug_2026/ranking_model', 'model_state_seed_{}.pth'.format(seed))
    model_state = torch.load(filename)
    mlp.load_state_dict(model_state)
    
    ########################################################################################################################################################################
    # Predict the performance metrics for the gauged basins using the trained model
    _, ypred_train, yobs_train, X_train = test_mod(train_dataloader_noshuffle, mlp)

    # Test the model on the ungauged basins
    _, ypred, yobs, X_test = test_mod(pred_param_dataloader, mlp)

    # Remove the standardization to get the predictions in the original scale
    yobs, ypred = yobs.cpu(), ypred.cpu()
    yobs_train, ypred_train = yobs_train.cpu(), ypred_train.cpu()
    X_train, X_test = X_train.cpu(), X_test.cpu()

    ypred_train = torch.mul(ypred_train, ystd.view(1, -1)) + ymean.view(1, -1)      
    yobs_train = torch.mul(yobs_train, ystd.view(1, -1)) + ymean.view(1, -1) 
    
    ypred = torch.mul(ypred, ystd.view(1, -1)) + ymean.view(1, -1)      
    yobs = torch.mul(yobs, ystd.view(1, -1)) + ymean.view(1, -1)        
    
    X_train = torch.mul(X_train, stdx.view(1, -1)) + meanx.view(1, -1)
    X_test = torch.mul(X_test, stdx.view(1, -1)) + meanx.view(1, -1)

    # Save the results in a list
    ypred_train_list.append(ypred_train.numpy())
    yobs_train_list.append(yobs_train.numpy())
    X_train_list.append(X_train.numpy())
    ypred_list.append(ypred.numpy())
    yobs_list.append(yobs.numpy())
    X_list.append(X_test.numpy())

    del mlp,  model_state

# Get the average predictions
ypred_list = np.array(ypred_list)
ypred_final = np.mean(ypred_list, axis=0)
yobs = yobs_list[0]

ypred_train_list = np.array(ypred_train_list)
ypred_train_final = np.mean(ypred_train_list, axis=0)
yobs_train = yobs_train_list[0]

X_test = X_list[0]
X_train = X_train_list[0]


# Compute regret
regret_list = []
for basin in test_basins: 
    ind = np.nonzero(test_basin_pred_seq == basin)[0]
    results = compute_regret_at_k(yobs[ind, 0], ypred_final[ind, 0], k=20)

    regret_list.append([results['true_best_kge'], results['regret_at_k'], results['topk_spearman'], results['precision_at_k']])
regret_list = np.array(regret_list)
    
# Write predicted performance metrics along with the predictor variables for the ungauged basins to a text file

NSE_list = []
corr_list = []
for basin in test_basins: 
    ind = np.nonzero(test_basin_pred_seq == basin)[0]
    nse = computeNSE(yobs[ind, 0], ypred_final[ind, 0])
    NSE_list.append(nse)

    corr = np.corrcoef(yobs[ind, 0], ypred_final[ind,0])[0,1]
    spearman_corr, _ = spearmanr(yobs[ind, 0], ypred_final[ind, 0])
    corr_list.append(spearman_corr)

    # Write data to a text file
    
    write_df = pd.DataFrame({'NSE_actual': yobs[ind, 0], 'NSE_mlp_pred': ypred_final[ind, 0], 
                             'NSAE_actual': yobs[ind, 1], 'NSAE_mlp_pred': ypred_final[ind, 1], 
                             'KGE_actual': yobs[ind, 2], 'KGE_mlp_pred': ypred_final[ind, 2],
                             'corr_actual': yobs[ind, 3], 'corr_mlp_pred': ypred_final[ind, 3],
                             'alpha_actual': yobs[ind, 4], 'alpha_mlp_pred': ypred_final[ind, 4],
                             'beta_actual': yobs[ind, 5], 'beta_mlp_pred': ypred_final[ind, 5],
                             'peak_error_actual': yobs[ind, 6], 'peak_error_mlp_pred': ypred_final[ind, 6],
                             '  h': yobs[ind, 7], 'low_error_mlp_pred': ypred_final[ind, 7],
                             'Predicted_Param': list(X_test[ind, :])})
    
    filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_28_Aug_2026/ranking_model/weight_expo_1.0_ratio_3_ensemble_10000', 'ungauged_performance_metrics_predictability_{}.txt'.format(basin))
    write_df.to_csv(filename, index=False)

# Write predicted performance metrics along with the predictor variables for the gauged basins to a text file
NSE_list = []
corr_list = []
for basin in train_basins:
    ind = np.nonzero(train_basin_seq == basin)[0]
    nse = computeNSE(yobs_train[ind, 2], ypred_train_final[ind, 2])
    NSE_list.append(nse)

    corr = np.corrcoef(yobs_train[ind, 0], ypred_train_final[ind,0])[0,1]
    spearman_corr, _ = spearmanr(yobs_train[ind, 2], ypred_train_final[ind, 2])
    corr_list.append(corr)

    # Write data to a text file
    write_df = pd.DataFrame({'NSE_actual': yobs_train[ind, 0], 'NSE_mlp_pred': ypred_train_final[ind, 0], 
                             'NSAE_actual': yobs_train[ind, 1], 'NSAE_mlp_pred': ypred_train_final[ind, 1], 
                             'KGE_actual': yobs_train[ind, 2], 'KGE_mlp_pred': ypred_train_final[ind, 2],
                            'corr_actual': yobs_train[ind, 3], 'corr_mlp_pred': ypred_train_final[ind, 3],
                            'alpha_actual': yobs_train[ind, 4], 'alpha_mlp_pred': ypred_train_final[ind, 4],
                            'beta_actual': yobs_train[ind, 5], 'beta_mlp_pred': ypred_train_final[ind, 5],
                            'peak_error_actual': yobs_train[ind, 6], 'peak_error_mlp_pred': ypred_train_final[ind, 6],
                            'low_error_actual': yobs_train[ind, 7], 'low_error_mlp_pred': ypred_train_final[ind, 7],
                             'Predicted_Param': list(X_train[ind, :])})
    filename = os.path.join(results_dir, 'MLP_results/equifinality_quantification_28_Aug_2026/ranking_model/weight_expo_1.0_ratio_3_ensemble_10000', 'gauged_performance_metrics_predictability_{}.txt'.format(basin))
    write_df.to_csv(filename, index=False)