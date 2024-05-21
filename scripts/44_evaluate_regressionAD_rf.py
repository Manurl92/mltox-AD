
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

import utils

# %%

path_output = path_root + 'output/regression/'

# %%

def compile_errors(modeltype):

    # load error files
    df_cv = pd.read_csv(path_output + modeltype + '_CV-errors.csv')
    df_test = pd.read_csv(path_output + modeltype + '_test-errors.csv')

    # concatenate
    df = pd.concat((df_cv, df_test)).reset_index(drop=True)
    df['chem_fp'] = df['chem_fp'].str.replace('pcp', 'PubChem')

    # add modeltype
    if modeltype == 'lasso':
        df['model'] = 'LASSO'
    elif modeltype == 'rf':
        df['model'] = 'RF'
    elif modeltype == 'xgboost':
        df['model'] = 'XGBoost'
    elif modeltype == 'gp':
        df['model'] = 'GP'

    return df

# %%

# load results
modeltype = 'rf'
df_errors = compile_errors(modeltype)
df_errors

# %%

# only two group splits
list_cols = ['totallyrandom', 'occurrence']
df_errors = df_errors[df_errors['groupsplit'].isin(list_cols)].copy()

# categorical variables
# the fingerprint 'none' corresponds to the top 3 features models
list_cols_fps = ['MACCS', 'PubChem', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred', 'none']
df_errors = utils._transform_to_categorical(df_errors, 'groupsplit', ['totallyrandom', 'occurrence'])
df_errors = utils._transform_to_categorical(df_errors, 'chem_fp', list_cols_fps)
df_errors = utils._transform_to_categorical(df_errors, 'model', ['LASSO', 'RF', 'XGBoost', 'GP'])
df_errors = utils._transform_to_categorical(df_errors, 'set', ['train', 'valid', 'trainvalid', 'test'])
df_errors = utils._transform_to_categorical(df_errors, 'conctype', ['molar', 'mass'])

# %%

# best hyperparameters for RF
df_errors = utils._transform_to_categorical(df_errors, 'conctype', ['molar', 'mass'])
df_errors = utils._transform_to_categorical(df_errors, 'groupsplit', ['totallyrandom', 'occurrence'])
df_errors = utils._transform_to_categorical(df_errors, 'chem_fp', list_cols_fps)

# 'max_features', 'min_samples_leaf', 
list_cols_hp = ['n_estimators', 'max_depth', 'max_samples', 'min_samples_split', 'max_features']
list_cols = ['conctype', 'groupsplit', 'chem_fp']  #, 'rmse', 'mae', 'r2']
list_cols += list_cols_hp
list_cols_sort = ['conctype', 'groupsplit', 'chem_fp']
df_l = df_errors[df_errors['set'] == 'train'][list_cols].sort_values(list_cols_sort).copy()
print(df_l.to_latex(index=False))

# %%
