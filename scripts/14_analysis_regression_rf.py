# Regression analysis with random forests: Training

# %%

import os
if os.getcwd().endswith('scripts'):
    path_root = '../'
else:
    path_root = './' 
import sys
sys.path.insert(0, path_root + 'src/')
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor

import utils
import modeling as mod

# %%

# set paths
path_data = path_root + 'data/'
path_output = path_root + 'output_rf/tmp/'

# create output folder if it does not exist
Path(path_output).mkdir(parents=True, exist_ok=True) 

# %%

# Parameter grids

# Set the challenge, features, splits, and concentration type
# ! for single species challenges: set tax_prop and tax_pdm to 'none'
# ! this script has not been tested for challenges on all taxonomic groups
param_grid = [

    {
     # data
     'challenge': ['t-F2F'],
     # features
     'chem_fp': ['MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred', 'none'], 
#     'chem_fp': ['MACCS'], 
     'chem_prop': ['chemprop'],                 #['none', 'chemprop'],
     'tax_pdm': ['none'],                       #['none', 'pdm', 'pdm-squared'],
     'tax_prop': ['taxprop-migrate2'],          #['none', 'taxprop-migrate2', 'taxprop-migrate5'],
     'exp': ['exp-dropfirst'],                  #['none', 'exp-dropnone', 'exp-dropfirst'],     # use dropfirst
     # splits
     'groupsplit': ['totallyrandom', 'occurrence'],
     # concentration
     'conctype': ['molar', 'mass'] 
    }
]

# Random forest hyperparameter grid
hyperparam_grid = [
    {
    # model hyperparameters     
    'n_estimators': [25, 75],
    'max_depth': [200], 
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_samples': [1.0],  
    'max_features': ['sqrt'],
    }
]
#hyperparam_grid = [
    #{
    ## model hyperparameters     
    #'n_estimators': [50, 100, 150, 300],
    #'max_depth': [50, 100, 200], 
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1],
    #'max_samples': [0.25, 0.5, 1.0],  
    #'max_features': ['sqrt', 1],
    #}
#]

# Evaluation metric (Here, we minimize RMSE.)
metric = 'rmse'

# This parameter is only needed for GP regression. Here, it just needs to be initialized.
lengthscales = -1

# Select columns of prediction datafram to save
list_cols_preds = ['test_id', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']

# %%

for i, param in enumerate(ParameterGrid(param_grid)):

    print("run " + str(i + 1) + " of " + str(len(ParameterGrid(param_grid))))
    param_sorted = {k:param[k] for k in param_grid[0].keys()}
    print(param_sorted)
    print("-------------------------------")

    ## preparation
    ## -----------

    # get parameter grid settings
    challenge = param['challenge']
    chem_fp = param['chem_fp']
    chem_prop = param['chem_prop']
    tax_pdm = param['tax_pdm']
    tax_prop = param['tax_prop']
    exp = param['exp']
    groupsplit = param['groupsplit']
    conctype = param['conctype']

    # set column for concentration type
    if conctype == 'mass':
        col_conc = 'result_conc1_mean_log'
    elif conctype == 'molar':
        col_conc = 'result_conc1_mean_mol_log'

    # Mordred contains chemical properties. Set chemprop to 'none' for Mordred
    if chem_fp == 'Mordred':
        chem_prop = 'none'

    ## load data
    ## ---------

    # load dataset
    df_eco = pd.read_csv(path_data + 'processed/' + challenge + '_mortality.csv', low_memory=False)

    # load phylogenetic distance matrix
    path_pdm = path_data + 'taxonomy/FCA_pdm_species.csv'
    df_pdm = utils.load_phylogenetic_tree(path_pdm)

    # print data loading summary summary
    print("data loading summary")
    print("# entries:", df_eco.shape[0])
    print("# species:", df_eco['tax_all'].nunique())
    print("# chemicals:", df_eco['test_cas'].nunique())

    ## apply encodings
    ## ---------------

    # extract chemical properties from df_eco
    list_cols_chem_prop = ['chem_mw', 'chem_mp', 'chem_ws', 
                           'chem_rdkit_clogp',
                           #'chem_pcp_heavy_atom_count',
                           #'chem_pcp_bonds_count', 'chem_pcp_doublebonds_count', 'chem_pcp_triplebonds_count',
                           #'chem_rings_count', 'chem_OH_count',
                           ]
    df_chem_prop_all = df_eco[list_cols_chem_prop].reset_index(drop=True)

    # encode experimental variables
    df_exp_all = mod.get_encoding_for_experimental_features(df_eco, exp)

    # encode taxonomic pairwise distances
    df_eco, df_pdm, df_enc = mod.get_encoding_for_taxonomic_pdm(df_eco, df_pdm, col_tax='tax_gs')

    # encode taxonomic Add my Pet features 
    df_tax_prop_all = mod.get_encoding_for_taxonomic_addmypet(df_eco)

    ## response variable
    ## -----------------

    # get response variable
    df_response = df_eco[col_conc]
    
    ## prepare train-test-split
    ## ------------------------

    # get train-test-split indices
    col_split = '_'.join(('split', groupsplit))
    df_eco['split'] = df_eco[col_split]
    trainvalid_idx = df_eco[df_eco['split'] != 'test'].index
    test_idx = df_eco[df_eco['split'] == 'test'].index
    
    ## get separate dataframe for each feature set with applied train-test-split
    ## -------------------------------------------------------------------------

    # experimental features
    df_exp, len_exp = mod.get_df_exp(df_exp_all)

    # molecular representation
    df_chem_fp, len_chem_fp, lengthscales_fp = mod.get_df_chem_fp(chem_fp, 
                                                                  df_eco, 
                                                                  lengthscales, 
                                                                  trainvalid_idx, 
                                                                  test_idx)

    # chemical properties
    df_chem_prop, len_chem_prop, lengthscales_prop = mod.get_df_chem_prop(chem_prop, 
                                                                          df_chem_prop_all, 
                                                                          lengthscales, 
                                                                          trainvalid_idx, 
                                                                          test_idx)

    # taxonomic pairwise distances
    df_tax_pdm, len_tax_pdm, squared = mod.get_df_tax_pdm(tax_pdm, df_eco, 'tax_pdm_enc')

    # taxonomic properties
    df_tax_prop, len_tax_prop = mod.get_df_tax_prop(tax_prop, 
                                                    df_tax_prop_all,
                                                    trainvalid_idx, 
                                                    test_idx)

    # concatenate features
    df_features = pd.concat((df_exp, df_chem_fp, df_chem_prop, df_tax_pdm, df_tax_prop), axis=1)
    if len(df_features) == 0:
        print('no features selected')
        continue

    ## apply train-test-split
    ## ----------------------

    # apply train-test-split
    df_trainvalid = df_features.iloc[trainvalid_idx, :].reset_index(drop=True)
    df_test = df_features.iloc[test_idx, :].reset_index(drop=True)
    df_eco_trainvalid = df_eco.iloc[trainvalid_idx, :].reset_index(drop=True)
    df_eco_test = df_eco.iloc[test_idx, :].reset_index(drop=True)
    X_trainvalid = df_trainvalid.to_numpy()
    X_test = df_test.to_numpy()
    y_trainvalid = np.array(df_response[trainvalid_idx]).reshape(-1, 1)
    y_test = np.array(df_response[test_idx]).reshape(-1, 1)

    ## prepare cross-validation

    # initialize
    list_df_error_grid = []
    metric_value = 100
    df_preds_v = pd.DataFrame()
    
    # get cross-validation splits
    if not 'loo' in groupsplit:
        dict_splits = mod.get_precalculated_cv_splits(df_eco_trainvalid)
        n_splits_cv = df_eco_trainvalid['split'].astype('int').max() + 1
    else:
        # leave-one-out (loo) splits are done differently
        dict_splits = {}
        dict_splits['loo'] = (trainvalid_idx, test_idx)
        n_splits_cv = 1

    # grid search over hyperparameter grid
    for idx_hp, hyperparam in enumerate(ParameterGrid(hyperparam_grid)):
        print(idx_hp, hyperparam)

        # initialize
        list_df_pred_v_grid = []
        list_df_pred_t_grid = []

        # run crossvalidation
        for fold, (train_idx, valid_idx) in dict_splits.items():
        
            print('fold:', fold)

            # apply train validation split
            if not 'loo' in groupsplit:
                X_train = df_trainvalid.loc[train_idx].to_numpy()
                X_valid = df_trainvalid.loc[valid_idx].to_numpy()
                df_eco_train = df_eco_trainvalid.loc[train_idx]
                df_eco_valid = df_eco_trainvalid.loc[valid_idx]
                y_train = y_trainvalid[train_idx]
                y_valid = y_trainvalid[valid_idx]
            else:
                # leave-one-out (loo) splits are done differently
                X_train = df_trainvalid.to_numpy()
                X_valid = df_test.to_numpy()
                df_eco_train = df_eco_trainvalid.copy()
                df_eco_valid = df_eco_test.copy()
                y_train = y_trainvalid
                y_valid = y_test
            print("train and validation", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
        
            # train random forest model
            model = RandomForestRegressor(**hyperparam)
            model.fit(X_train, y_train.ravel())

            # predict for validation data
            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)
        
            # generate output
            df_pred_train = df_eco_train.copy()
            df_pred_train['conc_pred'] = y_train_pred
            df_pred_train = mod._add_params_fold_to_df(df_pred_train, 
                                                       hyperparam, 
                                                       fold)
            list_df_pred_t_grid.append(df_pred_train)
            df_pred_valid = df_eco_valid.copy()
            df_pred_valid['conc_pred'] = y_valid_pred
            df_pred_valid = mod._add_params_fold_to_df(df_pred_valid, 
                                                       hyperparam, 
                                                       fold)
            list_df_pred_v_grid.append(df_pred_valid)
        
        # calculate evaulation metrics
        col_true = col_conc
        col_pred = 'conc_pred'
        df_preds_t_grid = pd.concat(list_df_pred_t_grid)
        df_preds_v_grid = pd.concat(list_df_pred_v_grid)
        df_preds_v_grid['idx_hp'] = idx_hp
        df_error_grid = mod.calculate_evaluation_metrics(df_preds_t_grid, 
                                                         df_preds_v_grid,
                                                         col_true, 
                                                         col_pred, 
                                                         n_splits_cv)
        df_error_grid = mod._add_params_fold_to_df(df_error_grid, hyperparam)
        df_error_grid['idx_hp'] = idx_hp
        list_df_error_grid.append(df_error_grid)

        # determine whether error improved (based on validation error)
        metric_grid = df_error_grid[(df_error_grid['fold'] == 'mean')
                                    & (df_error_grid['set'] == 'valid')][metric].iloc[0]
        if metric_grid < metric_value:
            df_preds_v = df_preds_v_grid.copy()
            metric_value = metric_grid

    # concatenate errors for hyperparameter grid
    df_errors_grid = pd.concat(list_df_error_grid).reset_index(drop=True)

    # find best hyperparameters based on validation error
    df_e_v = df_errors_grid[(df_errors_grid['fold'] == 'mean') &
                            (df_errors_grid['set'] == 'valid')]
    df_e_v_best = df_e_v.loc[df_e_v[metric].idxmin()]    # minimize rmse
    idx_hp_best = df_e_v_best['idx_hp']
    df_errors_grid['best_hp'] = False
    df_errors_grid.loc[df_errors_grid['idx_hp'] == idx_hp_best, 'best_hp'] = True

    # store errors for all hyperparameters
    df_errors_grid = mod._add_params_fold_to_df(df_errors_grid, param_sorted)
    str_file = '_'.join([str(i) for i in param_sorted.values()])
    df_errors_grid.round(5).to_csv(path_output + 'errors_' + str_file + '.csv', index=False)

    # store predictions for best hyperparameter
    list_cols_conc = ['idx_hp', 'fold', col_conc, 'conc_pred']
    df_store = df_preds_v[list_cols_preds + list_cols_conc].copy()
    df_store = mod._add_params_fold_to_df(df_store, param_sorted)
    df_store = mod._add_params_fold_to_df(df_store, hyperparam)
    df_store.round(5).to_csv(path_output + 'preds_' + str_file + '.csv', index=False)

print('done')
# %%