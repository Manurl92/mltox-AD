# View a dataset

# %%

import os
if os.getcwd().endswith('scripts'):
    path_root = '../'
else:
    path_root = './' 
import sys
sys.path.insert(0, path_root + 'src/')

import numpy as np
import pandas as pd

import utils

# %%

# set paths
path_data = path_root + 'data/'

# %%

# set a challenge
challenge = 't-F2F'

# load dataset
df_eco = pd.read_csv(path_data + 'processed/' + challenge + '_mortality.csv', low_memory=False)

# load phylogenetic distance matrix
path_pdm = path_data + 'taxonomy/FCA_pdm_species.csv'
df_pdm = utils.load_phylogenetic_tree(path_pdm)

# print data loading summary
print("data loading summary")
print("# entries:", df_eco.shape[0])
print("# species:", df_eco['tax_all'].nunique())
print("# chemicals:", df_eco['test_cas'].nunique())

# %%