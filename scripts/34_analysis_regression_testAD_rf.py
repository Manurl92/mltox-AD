# Regression analysis with random forests: Testing

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
path_output = path_root + 'output/regression/'

# %%

# load cross-validation errors
# the dataframe includes the best hyperparameters
modeltype = 'rf'
df_cv = pd.read_csv(path_output + modeltype + '_CV-errors.csv')

# get df with one entry only per chem_fp x groupsplit combination
df_cv = df_cv[df_cv['set'] == 'valid'].copy()

# %%

# load test output
path_file = Path(path_output + modeltype + '_test-errors.csv')
if path_file.is_file():
    df_e_test = pd.read_csv(path_file)
else: 
    df_e_test = pd.DataFrame()
path_file = Path(path_output + modeltype + '_predictions.csv')
if path_file.is_file():
    df_pr_test = pd.read_csv(path_file)
else: 
    df_pr_test = pd.DataFrame()

# %%

# Parameter grids

# Set the challenge, molecular representation, splits, and concentration type
# The other features are selected automatically
param_grid = [
    {
     # data
     'challenge': ['t-F2F'],
     # features
     'chem_fp': ['none', 'MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec'], #['Mordred']
     # splits
     'groupsplit': ['totallyrandom', 'occurrence'],
     # concentration
     'conctype': ['molar'],   #['molar', 'mass']
    }
]

# This parameter is only needed for GP regression. Here, it just needs to be initialized.
lengthscales = -1

# Select columns of prediction datafram to save
list_cols_preds = ['test_id', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']

# %%

# initialize
list_df_errors = []
list_df_preds = []

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
    groupsplit = param['groupsplit']
    conctype = param['conctype']

    # check whether this test run is already done
    if len(df_e_test) > 0:
        df_tmp = df_e_test[(df_e_test['challenge'] == challenge)
                           & (df_e_test['chem_fp'] == chem_fp)
                           & (df_e_test['conctype'] == conctype)
                           & (df_e_test['groupsplit'] == groupsplit)]
        if len(df_tmp) > 0:
            continue

    # get other parameters
    df_e_sel = df_cv[(df_cv['challenge'] == challenge)
                     & (df_cv['chem_fp'] == chem_fp)
                     & (df_cv['groupsplit'] == groupsplit)
                     & (df_cv['conctype'] == conctype)]
    if len(df_e_sel) == 0:   # continue if this cv run is not done
        continue
    chem_prop = df_e_sel['chem_prop'].iloc[0]
    tax_pdm = df_e_sel['tax_pdm'].iloc[0]
    tax_prop = df_e_sel['tax_prop'].iloc[0]
    exp = df_e_sel['exp'].iloc[0]

    # get hyperparameters as a dictionary
    hyperparam = {}
    list_cols_hp = ['max_depth', 'max_features', 'max_samples', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
    for col in list_cols_hp:
        if col == 'max_features' and df_e_sel[col].iloc[0] == '1':
            hyperparam[col] = int(df_e_sel[col].iloc[0])
        else:
            hyperparam[col] = df_e_sel[col].iloc[0]

    # set concentration type
    if conctype == 'mass':
        col_conc = 'result_conc1_mean_log'
    elif conctype == 'molar':
        col_conc = 'result_conc1_mean_mol_log'

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
    if challenge not in ['s-A2A', 't-A2A']:
        df_tax_prop_all = mod.get_encoding_for_taxonomic_addmypet(df_eco)
    else:
        df_tax_prop_all = pd.DataFrame()
        tax_prop = 'none'

    # print summary
    print("# entries:", df_eco.shape[0])
    print("# species:", df_eco['tax_all'].nunique())
    print("# chemicals:", df_eco['test_cas'].nunique())

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

    ## modeling
    ## --------

    # train random forest model on entire training data
    model = RandomForestRegressor(**hyperparam)
    model.fit(X_trainvalid, y_trainvalid.ravel())

    # predict for training and test data
    y_tv_pred = model.predict(X_trainvalid)
    y_test_pred = model.predict(X_test)
        
    # generate output
    df_pred_tv = df_eco_trainvalid.copy()
    df_pred_tv['conc_pred'] = y_tv_pred
    df_pred_tv = mod._add_params_fold_to_df(df_pred_tv, 
                                            hyperparam, 
                                            'trainvalid')
    df_pred_test = df_eco_test.copy()
    df_pred_test['conc_pred'] = y_test_pred
    df_pred_test = mod._add_params_fold_to_df(df_pred_test, 
                                              hyperparam, 
                                              'test')

    # calculate evaulation metrics
    col_true = col_conc
    col_pred = 'conc_pred'
    df_error = mod.calculate_evaluation_metrics(df_pred_tv, 
                                                df_pred_test,
                                                col_true, 
                                                col_pred, 
                                                -1)
    df_error['set'] = df_error['fold']
    df_error['chem_prop'] = chem_prop
    df_error['tax_pdm'] = tax_pdm 
    df_error['tax_prop'] = tax_prop
    df_error['exp'] = exp 
    df_error = mod._add_params_fold_to_df(df_error, param)
    df_error = mod._add_params_fold_to_df(df_error, hyperparam)
    list_df_errors.append(df_error)

    # store predictions
    df_pred = pd.concat([df_pred_tv, df_pred_test])
    list_cols_conc = ['fold', col_conc, 'conc_pred']
    df_pred = df_pred[list_cols_preds + list_cols_conc].copy()
    df_pred = mod._add_params_fold_to_df(df_pred, param_sorted)
    df_pred = mod._add_params_fold_to_df(df_pred, hyperparam)
    list_df_preds.append(df_pred)

# concatenate and store
if len(list_df_errors) > 0:
    df_errors = pd.concat(list_df_errors)
    df_errors = pd.concat((df_e_test, df_errors))
    df_errors.round(5).to_csv(path_output + modeltype + '_test-errors.csv', index=False)

if len(list_df_preds) > 0:
    df_preds = pd.concat(list_df_preds)
    df_preds = pd.concat((df_pr_test, df_preds))
    df_preds.round(5).to_csv(path_output + modeltype + '_predictions.csv', index=False)

print('done')
# %%