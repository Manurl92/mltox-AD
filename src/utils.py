import os

import numpy as np
import pandas as pd

# -------------------------------------
def load_phylogenetic_tree(path_pdm):
    '''
    load phylogenetic tree and return pairwise distance matrix pdm.
    If pdm has been stored already, load it directly.
    
    '''

    df_pdm = pd.read_csv(path_pdm).set_index('Unnamed: 0')
    
    return df_pdm

# -------------------------------
def get_necessary_bits(df, 
                       col_fp,
                       name_fp,
                       **kwargs):
    
    df_fp = get_fingerprint_as_dataframe(df, 
                                         col_fp=col_fp,
                                         name_fp=name_fp,
                                         do_drop_duplicates=False)
    
    df_fp_nodupl = get_fingerprint_as_dataframe(df, 
                                                col_fp=col_fp,
                                                name_fp=name_fp,
                                                do_drop_duplicates=True)

    # get uninformative bits from fp without (chemical) duplicates
    list_uninformative = get_uninformative_bits(df_fp_nodupl, 
                                               **kwargs)
    df_fp_subset = df_fp.iloc[:, ~df_fp.columns.isin(list_uninformative)]

    # get duplicated bits from remaining bits
    df = df_fp_subset.sum().reset_index().rename(columns = {0: "n_pos"})
    df_agg = df.groupby(['n_pos'])['index'].agg(list).to_frame().reset_index().rename(columns = {'index': 'bits'})
    df_agg['n_bits'] = df_agg['bits'].apply(lambda x: len(x))
    df_agg = df_agg[df_agg['n_bits'] > 1]
    list_duplicated = []
    for bits in df_agg['bits']:
        list_dupl_ = get_duplicated_bits(df_fp_subset[bits])
        list_duplicated.append(list_dupl_)
        
    # remove uninformative and duplicated bits
    df_fp_subset = df_fp.iloc[:, ~df_fp.columns.isin(list_uninformative + list_duplicated)]
    
    return df_fp_subset

def get_fingerprint_as_dataframe(df, 
                                 col_fp,
                                 name_fp,
                                 do_drop_duplicates=True):
    '''
    get fingerprint from string as a dataframe
    
    '''

    if do_drop_duplicates:
        _df = df[['test_cas', col_fp]].drop_duplicates().reset_index(drop=True)
    else:
        _df = df[['test_cas', col_fp]].reset_index(drop=True)
        
    df_fp = _df[col_fp].apply(lambda x: list(x)).reset_index()
    df_fp = pd.DataFrame(df_fp[col_fp].to_list()).astype(int)
    n_digits = len(str(np.max(df_fp.columns)))
    df_fp.columns = [name_fp + str(col).zfill(n_digits) for col in df_fp.columns]

    return df_fp

def get_uninformative_bits(df_fp, std_threshold=0.1):
    '''
    helper function to get uniformative bits, i.e. those which vary less than specified standard deviation threshold
    (in Anlehnung an Lovric https://doi.org/10.3390/ph14080758)
    
    '''
    
    list_uninformative = list(df_fp.loc[:, df_fp.std() < std_threshold].columns)

    return list_uninformative

def get_duplicated_bits(df_fp, 
                        do_print=False):
    '''
    function to get duplicated bits
    '''

    # initialize
    list_dupl = []
    
    # for all combinations
    for i, col1 in enumerate(df_fp.columns):
        
        for col2 in list(df_fp.columns)[i+1:]:
            
            # they are the same if diff is 0
            diff = np.abs((df_fp[col1] - df_fp[col2])).mean()
            
            if diff == 0:
                
                if do_print:
                    print(col1, col2, df_fp[col1].sum())
                
                # update list
                if col2 not in list_dupl:
                    list_dupl.append(col2)

    return list_dupl

def get_fingerprint(df, chem_fp, trainvalid_idx, test_idx):

    col_fp = 'chem_' + chem_fp + '_fp'
    df_fp_tv = get_necessary_bits(df.iloc[trainvalid_idx], 
                                  col_fp, 
                                  chem_fp,
                                  std_threshold=0.1)
    
    if len(test_idx) > 0:
        df_fp_test = get_fingerprint_as_dataframe(df.iloc[test_idx], 
                                                  col_fp=col_fp,
                                                  name_fp=chem_fp,
                                                  do_drop_duplicates=False)
        df_fp_test = df_fp_test[list(df_fp_tv.columns)].copy()

    df_output = pd.DataFrame(index=df.index, columns=df_fp_tv.columns, dtype='float64')
    df_output.iloc[trainvalid_idx] = df_fp_tv.to_numpy()
    if len(test_idx) > 0:
        df_output.iloc[test_idx] = df_fp_test.to_numpy()

    return df_output

def get_mol2vec(df_eco):

    # get mol2vec dataframe
    list_cols_mol2vec = [col for col in df_eco.columns if 'mol2vec' in col and 'allowed' not in col]
    df_mol2vec = df_eco[list_cols_mol2vec].copy()

    return df_mol2vec

def get_mordred(df_eco):

    # get mordred dataframe
    list_cols_mordred = [col for col in df_eco.columns if 'mordred' in col]
    df_mordred = df_eco[list_cols_mordred].copy()

    # remove constant columns
    df_mordred = df_mordred.loc[:, df_mordred.std() != 0].copy()

    # remove columns with some very high values
    df_mordred = df_mordred.loc[:, df_mordred.std() < 1e4].copy()

    # remove integer columns with very low standard deviations
    list_cols = list(df_mordred.loc[:, (df_mordred.dtypes == 'int') & (df_mordred.std() < 0.05)].columns)
    if 'chem_mordred_SsssP' in df_mordred.columns:
        list_cols += ['chem_mordred_SsssP']   # a float column with all zeros but one other value
    df_mordred = df_mordred.drop(list_cols, axis=1)

    return df_mordred

# -------------------------------
def read_result_files(path, file_type='errors'):
    '''
    
    function to read result files from specified directory
    file needs to start with type

    '''

    # list of files
    list_files = [f for f in os.listdir(path) if f.startswith(file_type)]

    # list of data frames
    list_dfs = []
    for f in list_files:
        _df = pd.read_csv(path + f)
        list_dfs.append(_df)

    # concatenate and preprocess
    df = pd.concat(list_dfs)
    df = df.reset_index(drop=True)

    return df

# ------------------------------
def _transform_to_categorical(df, col, categories):
    ''' helper function to transform a pandas column to a categorical '''
    df[col] = pd.Categorical(df[col],
                             categories=categories,
                             ordered=True)

    return df
