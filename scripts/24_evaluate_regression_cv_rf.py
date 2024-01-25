# Regression analysis with random forests: Evaluate cross-validation

# %%

import os
if os.getcwd().endswith('scripts'):
    path_root = '../'
else:
    path_root = './' 
import sys
sys.path.insert(0, path_root + 'src/')
from pathlib import Path

import utils

# %%

path_cvoutput = path_root + 'output_rf/'
path_output = path_root + 'output/regression/'

# create output folder if it does not exist
Path(path_output).mkdir(parents=True, exist_ok=True) 

# %%

# Random Forest

path_output_dir = path_cvoutput + 'tmp/'
df_errors = utils.read_result_files(path_output_dir, file_type='error')

# %%

# categorical variables for fingerprints
col = 'chem_fp'
list_categories = ['none', 'MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# categorical variable for folds
col = 'fold'
list_categories = ['mean', '0', '1', '2', '3', '4']
df_errors[col] = df_errors[col].astype('str')
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# categorical variable for groupsplit
col = 'groupsplit'
list_categories = ['totallyrandom', 'occurrence']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# categorical variable for conctype
col = 'conctype'
list_categories = ['molar', 'mass']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# sort
df_errors = df_errors.sort_values(['chem_fp', 'groupsplit'])

# check loading
if df_errors.isna().sum().sum() > 0:
    print("warning: check whether loading was correct")
else:
    print("loading seems correct")

# %%

# only look at results with best hyperparameters
df_oi = df_errors[df_errors['best_hp'] == True].copy()

# get data frame with best hyperparmeters only
df_e_oi = df_errors[(df_errors['best_hp'] == True) &
                    (df_errors['set'] == 'valid') &
                    (df_errors['fold'] == 'mean')]

# mean errors (train and valid of 5-fold CV)
df_e_v = df_oi[(df_oi['fold'] == 'mean')].copy()

list_cols = ['challenge', 'chem_fp', 'groupsplit', 'conctype', 'set', 'fold']
list_cols += ['chem_prop', 'tax_pdm', 'tax_prop', 'exp']
list_cols += ['best_hp', 'idx_hp', 'n_estimators', 'max_depth', 'max_features', 'max_samples', 'min_samples_leaf', 'min_samples_split']
list_cols += ['r2', 'rmse', 'mae', 'pearson']
df_e_v[df_e_v['best_hp'] == True][list_cols].round(2)

# %%

# store 
df_e_v[df_e_v['best_hp'] == True][list_cols].round(5).to_csv(path_output + 'rf_CV-errors.csv', index=False)

# %%
