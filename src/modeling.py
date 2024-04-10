import os

import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

import gpflow
from gpflow import set_trainable
from gpflow.config import default_float
import tensorflow as tf

from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

import utils

# -------------------------------
def get_encoding_for_features(df, list_onehot, list_ordinal, drop='none'):
    '''
    function to get onehot and ordinal encoding for features

    written for experimental features
    
    '''

    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

    # initialize
    list_dfs = []

    # onehot encoding
    if drop == 'none':
        drop = None
    onehot_enc = OneHotEncoder(drop=drop)
    for col in list_onehot:
        arr = onehot_enc.fit_transform(df[col].to_numpy().reshape(-1, 1))
        list_categories = list([col + '_' + cat for cat in onehot_enc.categories_][0])
        if drop=='first':
            list_categories = list_categories[1:]
        _df = pd.DataFrame(arr.toarray(), columns = list_categories)
        list_dfs.append(_df)
    
    # ordinal encoding
    ordinal_enc = OrdinalEncoder()
    scaler = MinMaxScaler()
    for col in list_ordinal:
        arr = ordinal_enc.fit_transform(df[col].to_numpy().reshape(-1,1))
        arr = scaler.fit_transform(arr)
        _df = pd.DataFrame(arr, columns=[col])
        list_dfs.append(_df)
    
    # concatenate
    df_exp = pd.concat(list_dfs, axis=1)

    return df_exp

def get_encoding_for_experimental_features(df_eco, exp):

    if exp in ['exp-dropnone', 'exp-dropfirst']:
        if exp == 'exp-dropnone':
            drop = 'none'
        elif exp == 'exp-dropfirst':
            drop = 'first'
        list_exp_onehot = ['result_conc1_type', 'test_exposure_type', 'test_media_type']
        list_exp_ordinal = ['result_obs_duration_mean']
        df_exp_all = get_encoding_for_features(df_eco, 
                                               list_exp_onehot, 
                                               list_exp_ordinal,
                                               drop=drop)

    elif exp == 'none':
        df_exp_all = pd.DataFrame()

    return df_exp_all

def get_encoding_for_taxonomic_pdm(df, 
                                   df_pdm, 
                                   col_tax='tax_gs_replaced'):
    
    from sklearn.preprocessing import OrdinalEncoder
    
    # get encoding
    ordinal_enc = OrdinalEncoder(dtype=np.int64)
    arr = ordinal_enc.fit_transform(df[col_tax].to_numpy().reshape(-1, 1))
    
    # add to input data frame
    df['tax_pdm_enc'] = arr
    
    # create data frame of encoding
    list_fish = list(ordinal_enc.categories_[0])
    df_enc = pd.DataFrame(list_fish, columns=[col_tax])
    df_enc['tax_pdm_enc'] = range(len(list_fish))
    
    # get subset of pairwise distances
    df_pdm = df_pdm.loc[df_pdm.index.isin(list_fish), df_pdm.columns.isin(list_fish)]

    # rename columns and index
    df_pdm.columns = ordinal_enc.transform(np.array(df_pdm.columns).reshape(-1, 1)).ravel()
    df_pdm.index = ordinal_enc.transform(np.array(df_pdm.index).reshape(-1, 1)).ravel()

    # sort by columns and index
    df_pdm = df_pdm.sort_index(axis=0).sort_index(axis=1)
    
    return df, df_pdm, df_enc

def _multihotencoding(df, col_var):
    '''
    
    helper function to get 'multihotencoding' for one variable

    '''

    def _get_new_colname(col_var, col):
        return '_'.join((col_var, col))

    # get unique entries
    df['split'] = df[col_var].str.split('_')
    unique_entries = sorted(set([item for sublist in df['split'] for item in sublist]))

    # create new data frame
    df_new = df['tax_gs'].to_frame()
    for col in unique_entries:
        col_new = _get_new_colname(col_var, col)
        df_new[col_new] = 0

    # fill new data frame
    for idx, list_split in enumerate(df['split']):
        for col in list_split:
            col_new = _get_new_colname(col_var, col)
            df_new.loc[idx, col_new] = 1

    return df_new

def get_encoding_for_taxonomic_addmypet(df_eco):

    # get multihotencoding for ecology data
    list_cols = ['tax_gs'] + [col for col in df_eco.columns if 'tax_eco' in col]
    df_tax_eco = df_eco[list_cols].drop_duplicates().reset_index(drop=True)
    df_tax_eco_enc = get_multihotencoding_for_tax_ecology(df_tax_eco)

    # get life history columns with no NAs
    df_tmp = df_eco[[col for col in df_eco.columns if 'tax_lh' in col]].isna().sum().to_frame()
    list_cols_lh = list(df_tmp[df_tmp[0] == 0].index)

    # merge it with life history and pseudo data from df_eco
    list_cols = ['tax_gs'] + [col for col in df_eco.columns if 'tax_ps' in col] + list_cols_lh
    df_tax_prop_all = pd.merge(df_eco[list_cols],
                               df_tax_eco_enc,
                               left_on=['tax_gs'],
                               right_on=['tax_gs'],
                               how='left')
    df_tax_prop_all = df_tax_prop_all.drop(['tax_gs'], axis=1)
    df_tax_prop_all = df_tax_prop_all.astype('float64')

    return df_tax_prop_all

def get_multihotencoding_for_tax_ecology(df):
    '''
    get multihotencoding for taxonomic ecology variables
    
    '''

    list_cols = ['tax_eco_food', 'tax_eco_climate', 'tax_eco_ecozone']
    list_cols += [col for col in df.columns if col.startswith('tax_eco_migrate')]

    list_new = []
    for i, var in enumerate(list_cols):
        df_new = _multihotencoding(df, var)
        if i != 0:
            df_new = df_new.drop('tax_gs', axis=1)
        list_new.append(df_new)

    df_tax_eco_new = pd.concat(list_new, axis=1)
    
    return df_tax_eco_new

def _get_tax_prop_migrate(df_tax_prop_all, col_migrate):
    '''
    helper function to get subset of tax_prop_all columns
    
    '''

    list_cols_nomigrate = [col for col in df_tax_prop_all.columns if 'migrate' not in col]
    list_cols_migrate = [col for col in df_tax_prop_all.columns if col.startswith(col_migrate)]
    list_cols = list_cols_nomigrate + list_cols_migrate
   
    return df_tax_prop_all[list_cols]

# -------------------------------
# train, validation and test splits
def get_precalculated_cv_splits(df):

    dict_splits = {}

    for fold in sorted(df['split'].unique()):
    
        dict_splits[fold] = (df[df['split'] != fold].index, df[df['split'] == fold].index)

    return dict_splits 

# -------------------------------
def standardscale_variables(df, 
                            trainvalid_idx, 
                            test_idx):

    from sklearn.preprocessing import StandardScaler

    # get trainvalid and test set
    X = df.to_numpy(dtype='float64')
    X_trainvalid = X[trainvalid_idx]
    X_test = X[test_idx]

    # scale features based on training set
    scaler = StandardScaler().fit(X_trainvalid)
    X_trainvalid = scaler.transform(X_trainvalid)
    X_test = scaler.transform(X_test)
    df_output = pd.DataFrame(index=df.index, columns=df.columns, dtype='float64')
    df_output.iloc[trainvalid_idx] = X_trainvalid
    df_output.iloc[test_idx] = X_test
    
    return df_output

def minmaxscale_variables(df, 
                          trainvalid_idx, 
                          test_idx):

    from sklearn.preprocessing import MinMaxScaler

    # get trainvalid and test set
    X = df.to_numpy(dtype='int64')
    X_trainvalid = X[trainvalid_idx]
    X_test = X[test_idx]

    # scale features based on training set
    scaler = MinMaxScaler().fit(X_trainvalid)
    X_trainvalid = scaler.transform(X_trainvalid)
    X_test = scaler.transform(X_test)
    df_output = pd.DataFrame(index=df.index, columns=df.columns, dtype='float64')
    df_output.iloc[trainvalid_idx] = X_trainvalid
    df_output.iloc[test_idx] = X_test
    
    return df_output

def get_df_exp(df_exp_all):

    df_exp = df_exp_all.copy()
    len_exp = df_exp.shape[1]

    return df_exp, len_exp

def get_df_chem_fp(chem_fp, 
                   df_eco, 
                   lengthscales, 
                   trainvalid_idx, 
                   test_idx):

    if chem_fp in ['pcp', 'MACCS', 'ToxPrint', 'Morgan']:
        df_chem_fp = utils.get_fingerprint(df_eco, chem_fp, trainvalid_idx, test_idx) 
        lengthscales_fp = lengthscales

    elif chem_fp == 'mol2vec':
        df_mol2vec = utils.get_mol2vec(df_eco)
        # standardscale (mol2vec features should be on a comparable scale to rest of the features)
        df_chem_fp = standardscale_variables(df_mol2vec, 
                                             trainvalid_idx, 
                                             test_idx)
        lengthscales_fp = np.round(lengthscales * 3.3, 0)    # 10

    elif chem_fp == 'Mordred':
        df_mordred = utils.get_mordred(df_eco)
        # minmax scale integer values
        df_m_int = df_mordred.loc[:, df_mordred.dtypes == 'int']
        df_m_int_scaled = minmaxscale_variables(df_m_int, 
                                                trainvalid_idx, 
                                                test_idx)
        # standardscale float values
        df_m_float = df_mordred.loc[:, df_mordred.dtypes == 'float']
        df_m_float_scaled = standardscale_variables(df_m_float, 
                                                    trainvalid_idx, 
                                                    test_idx)
        df_chem_fp = pd.concat((df_m_int_scaled, df_m_float_scaled), axis=1)
        lengthscales_fp = lengthscales
        
    elif chem_fp == 'none':
        df_chem_fp = pd.DataFrame()
        lengthscales_fp = lengthscales

    len_chem_fp = df_chem_fp.shape[1]

    return df_chem_fp, len_chem_fp, lengthscales_fp

def get_df_chem_prop(chem_prop, df_chem_prop_all, lengthscales, trainvalid_idx, test_idx):

    if chem_prop == 'chemprop':
        df_chem_prop = standardscale_variables(df_chem_prop_all,
                                               trainvalid_idx,
                                               test_idx)
        lengthscales_prop = np.round(lengthscales * 3.3, 0)    # 10
    if chem_prop == 'none':
        df_chem_prop = pd.DataFrame()
        lengthscales_prop = lengthscales

    len_chem_prop = df_chem_prop.shape[1]

    return df_chem_prop, len_chem_prop, lengthscales_prop

def get_df_tax_pdm(tax_pdm, df_eco, col_tax_pdm):

    if tax_pdm == 'pdm':
        df = df_eco[col_tax_pdm].to_frame()
        squared = False
    elif tax_pdm == 'pdm-squared':
        df = df_eco[col_tax_pdm].to_frame()
        squared = True
    elif tax_pdm == 'none':
        df = pd.DataFrame()
        squared = False

    len_tax = df.shape[1]

    return df, len_tax, squared

def get_df_tax_prop(tax_prop, 
                    df_tax_prop_all, 
                    trainvalid_idx, 
                    test_idx, 
                    list_cols_lh=None):

    if tax_prop.startswith('taxprop'):
        list_cols_eco = [col for col in df_tax_prop_all.columns if 'tax_eco' in col]
        col_migrate = tax_prop.split('-')[1]
        df_tax_prop_eco = _get_tax_prop_migrate(df_tax_prop_all[list_cols_eco], col_migrate)

        # standardscale pseudodata and lifehistory
        if list_cols_lh is None:
            list_cols_lh = [col for col in df_tax_prop_all.columns if 'tax_lh' in col]
        list_cols_ps = [col for col in df_tax_prop_all.columns if 'tax_ps' in col]
        list_cols_pslh = list_cols_ps + list_cols_lh 
        df_tax_prop_pslh = standardscale_variables(df_tax_prop_all[list_cols_pslh], 
                                                   trainvalid_idx, 
                                                   test_idx)

        df_tax_prop = pd.concat((df_tax_prop_eco, df_tax_prop_pslh), axis=1)

    elif tax_prop == 'none':
        df_tax_prop = pd.DataFrame()

    len_tax_prop = df_tax_prop.shape[1]

    return df_tax_prop, len_tax_prop

def _update_lol_cols_ARD(lol_cols_ARD, feat, do_ARD, df):

    if feat != 'none':
        if do_ARD:
            lol_cols_ARD.append(list(df.columns))
        else:
            lol_cols_ARD.append([])

    return lol_cols_ARD

# LASSO
# ------------------------------
def get_model_weights(model, list_cols):
    '''
    function to get LASSO model weights
    
    '''

    # intercept
    list_cols_i = ['intercept']
    df_i = pd.DataFrame([list_cols_i, model.intercept_], index=['feature', 'value']).transpose()

    # coefficients
    list_coefs = list(model.coef_[0])
    df_c = pd.DataFrame([list_cols, list_coefs], index=['feature', 'value']).transpose()

    # concatenate
    df = pd.concat((df_i, df_c)).reset_index(drop=True)
    df['value'] = df['value'].astype('float')

    return df

# GP
# ------------------------------
# code from PhotoSwitch paper 
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, active_dims=None):
        super().__init__(active_dims=active_dims)
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(variance, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

def get_kernel(which_kernel, variance, lengthscales, len_feat, len_tot, do_ARD):

    if which_kernel == 'RBF':
        kernel = gpflow.kernels.RBF(variance=variance,
                                    lengthscales=[lengthscales] * len_feat if do_ARD else lengthscales,
                                    active_dims=slice(len_tot, len_tot + len_feat))

    elif which_kernel == 'Linear':

        kernel = gpflow.kernels.Linear(variance=variance,
                                       active_dims=slice(len_tot, len_tot + len_feat))

    elif which_kernel == 'Tanimoto':

        kernel = Tanimoto(variance=variance,
                          active_dims=slice(len_tot, len_tot + len_feat))

    return kernel

def _update_kernel(kernel, new_kernel):

    if kernel is None:
        kernel = new_kernel
    else:
        kernel = kernel + new_kernel

    return kernel

def _update_len_tot(len_tot, new_len):

    len_tot = len_tot + new_len

    return len_tot

class PairwiseDistance(gpflow.kernels.IsotropicStationary):
    '''
    pairwise distance kernel for phylogenetic distances
    
    '''

    
    def __init__(self, variance=1.0, lengthscales=1.0, active_dims=None, t_pdm=None, squared=False):
        super().__init__(variance=variance, lengthscales=lengthscales, active_dims=active_dims)
        self.t_pdm = t_pdm
        self.squared = squared
     
    def K(self, X, X2=None):

        pd = self.get_pd_for_samples(X, self.t_pdm, X2)

        if self.squared:
            pd = tf.math.square(pd)
        
        return self.variance * tf.exp(-0.5 * pd / self.lengthscales)      
        
    def get_pd_for_samples(self, X, t_pdm, X2=None):
        '''
        X (and X2) contain the unique encoding for each fish which corresponds 
        to the column and row indices in tf_pdm. 
        These indices get extracted with meshgrid and properly combined with stack
        before the subset corresponding to X (and X2) is gathered.
        '''

        from gpflow.config import default_int
        
        if X2 is None:
            t_indices = tf.stack(tf.meshgrid(X, X), axis=-1)
            t_indices = tf.cast(t_indices, dtype=default_int())
            t_pdm_subset = tf.gather_nd(t_pdm,
                                        indices=t_indices)
            
            return t_pdm_subset
        
        # cast data types (they need to be the same)
        if X.dtype != X2.dtype:
            if X.dtype == 'int64':
                X2 = tf.cast(X2, dtype=X.dtype)
            elif X2.dtype == 'int64':
                X = tf.cast(X, dtype=X2.dtype)

        t_indices = tf.stack(tf.meshgrid(X, X2), axis=-1)
        t_indices = tf.cast(t_indices, dtype=default_int())
        t_pdm_subset = tf.gather_nd(t_pdm,
                                    indices=t_indices)

        return tf.transpose(t_pdm_subset)

def run_GP(X_train, y_train, 
           kernel, mean_function, noise_variance,
           maxiter,
           GP_type, ind_type, n_inducing):
           

    if GP_type == 'sparse':

        if ind_type == 'random':
            # select inducing points randomly
            idx_inducing = np.random.choice(range(len(X_train)), 
                                            size=n_inducing, 
                                            replace=False)
            inducing_variable = tf.convert_to_tensor(X_train[idx_inducing], 
                                                     dtype=default_float())

        elif ind_type == 'kmeans':
            # select inducing points through k-means clustering
            from sklearn.cluster import KMeans
            km_clustering = KMeans(n_clusters=n_inducing)
            km_clustering.fit(X_train)
            # the cluster centers, i.e., not data points, are selected!
            inducing_variable = tf.convert_to_tensor(km_clustering.cluster_centers_,
                                                     dtype=default_float())

        # SGPR
        model = gpflow.models.SGPR(data=(X_train, y_train),
                                   kernel=kernel,
                                   mean_function=mean_function,
                                   noise_variance=noise_variance,
                                   inducing_variable=inducing_variable         
                                   )
        set_trainable(model.inducing_variable, False)

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model.training_loss, 
                                model.trainable_variables, 
                                options=dict(maxiter=maxiter),
                                )
                                        
    elif GP_type == 'full':

        model = gpflow.models.GPR(data=(X_train, y_train),
                                  kernel=kernel,
                                  mean_function=mean_function,
                                  noise_variance=noise_variance)


        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(model.training_loss, 
                                model.trainable_variables, 
                                options=dict(maxiter=maxiter),
                                )

    else:

        print("specify a valid GP type")

    return opt_logs, model

def get_paramvalues_from_module(model, 
                                column_name='value', 
                                lol_cols_ARD=None,
                                list_rows_ind=None,
                                list_cols_ind=None):
    '''
    get pandas dataframe of module parameters
    '''
    from gpflow.utilities import read_values

    # dictionary of values
    dict_values = read_values(model)

    # iterate over copy
    for key, value in dict(dict_values).items():

        # if there is more than one entry, lengthscales was initiliazed as list with an entry for each X variable 
        # (i.e., using automatic relevance detection ARD)
        if len(value.shape) == 1:
            dict_temp = {}
            if len(lol_cols_ARD) == 1:
                kernel_int = 0
            else:
                kernel_int = int(''.join([i for i in key if i.isdigit()]))
            list_cols_ARD = lol_cols_ARD[kernel_int]
            for var, val in zip(list_cols_ARD, value):
                dict_temp['.'.join((key, var))] = val
            dict_values.pop(key)
            dict_values.update(dict_temp)
        elif len(value.shape) == 2:
            # create data frame for inducing variables
            df_ind = pd.DataFrame(dict_values[key], 
                                  index=list_rows_ind,
                                  columns=list_cols_ind)

            # don't store inducing variables in dictionary
#            dict_values[key] = np.nan
            dict_values.pop(key)

        else:
            dict_values[key] = np.squeeze(value)

    # data frame
    df_params = pd.DataFrame.from_dict(dict_values, 
                                orient='index', 
                                columns=[column_name],
                                dtype='object')

    # round to 5 digits
    df_params[column_name] = df_params[column_name].apply(lambda x: np.round(x, 5))

    df_ind = df_ind.apply(lambda x: np.round(x, 5))

    return df_params, df_ind

def get_df_opt(opt_logs, do_print=True):

    if do_print:
        for key in ['fun', 'hess_inv', 'message', 'nfev', 'nit', 'njev', 'status', 'success']:  # 'jac', 'x'
            print(key, opt_logs[key])
    else:
        pass

    dict_opt = {k:v for (k,v) in opt_logs.items() if k in ['status', 'success', 'message', 'nit', 'fun']}
    df_opt = pd.DataFrame.from_dict(dict_opt, orient='index').transpose()

    return df_opt

def _add_params_fold_to_df(df, param, fold=None):
    '''
    helper function to add parameters of grid and fold to a data frame    
    '''

    for key, value in param.items():
        df[key] = value
    if fold is not None:
        df['fold'] = str(fold)

    return df

def get_complete_kernel(len_exp, len_chem_fp, len_chem_prop, len_tax_pdm, len_tax_prop,
                        which_kernel_fp, which_kernel_other, 
                        variance, 
                        lengthscales, lengthscales_fp, lengthscales_prop, lengthscales_tax_pdm,
                        do_ARD_fp, do_ARD_other,
                        df_pdm, squared):

    '''
    function to get complete kernel and number of features

    '''

    # initialize
    kernel = None
    len_tot = 0
            
    # experimental
    if len_exp > 0:
        print("exp:", len_tot, slice(len_tot, len_tot + len_exp))
        k_exp = get_kernel(which_kernel_other, 
                           variance, 
                           lengthscales, 
                           len_exp, 
                           len_tot, 
                           do_ARD_other)

        kernel = k_exp
        len_tot = _update_len_tot(len_tot, len_exp)
              
    # fingerprint
    if len_chem_fp > 0:
        print("chem_fp:", len_tot, slice(len_tot, len_tot + len_chem_fp))
        print("lengthsales for fingerprint:", lengthscales_fp)

        k_chem_fp = get_kernel(which_kernel_fp, 
                               variance, 
                               lengthscales_fp, 
                               len_chem_fp, 
                               len_tot, 
                               do_ARD_fp)

        kernel = _update_kernel(kernel, k_chem_fp)
        len_tot = _update_len_tot(len_tot, len_chem_fp)
            
    # chemical properties
    if len_chem_prop > 0:
        print("chem_prop:", len_tot, slice(len_tot, len_tot + len_chem_prop))

        k_chem_prop = get_kernel(which_kernel_other, 
                                 variance, 
                                 lengthscales_prop, 
                                 len_chem_prop, 
                                 len_tot, 
                                 do_ARD_other)

        kernel = _update_kernel(kernel, k_chem_prop)
        len_tot = _update_len_tot(len_tot, len_chem_prop)
                
    # taxonomic pairwise distances
    if len_tax_pdm > 0:
        print("tax", len_tot, slice(len_tot, len_tot + len_tax_pdm))

        t_pdm = tf.convert_to_tensor(df_pdm)
        k_tax_pdm = PairwiseDistance(variance=variance,
                                     lengthscales=lengthscales_tax_pdm,
                                     active_dims=slice(len_tot, len_tot + len_tax_pdm),
                                     t_pdm=t_pdm,
                                     squared=squared)

        kernel = _update_kernel(kernel, k_tax_pdm)
        len_tot = _update_len_tot(len_tot, len_tax_pdm)

    # taxonomic features
    if len_tax_prop > 0:
        print("tax_prop", len_tot, slice(len_tot, len_tot + len_tax_prop))

        k_tax_prop = get_kernel(which_kernel_other, 
                                variance, 
                                lengthscales, 
                                len_tax_prop, 
                                len_tot, 
                                do_ARD_other)

        kernel = _update_kernel(kernel, k_tax_prop)
        len_tot = _update_len_tot(len_tot, len_tax_prop)

    return kernel, len_tot

# calculate evaluation metrics
# ------------------------------
def _calculate_rmse(df, col_true, col_pred):
    
    return root_mean_squared_error(df[col_true], df[col_pred])

def _calculate_mae(df, col_true, col_pred):
    
    return mean_absolute_error(df[col_true], df[col_pred])

def _calculate_r2(df, col_true, col_pred):
    
    return r2_score(df[col_true], df[col_pred])

def _calculate_pearson(df, col_true, col_pred):

    y_true = np.array(df[col_true])
    y_pred = np.array(df[col_pred])
    return stats.pearsonr(y_true, y_pred)[0]

def calculate_evaluation_metrics(df_preds_train,
                                 df_preds_valid,
                                 col_true, 
                                 col_pred, 
                                 n_splits):

    # initialize
    list_df_trains = []
    list_df_valids = []

    # overall error is only calculated if the runs for all folds were successful
    calculate_overall = df_preds_valid['fold'].nunique() == n_splits

    # RMSE
    metric = 'rmse'
    df_train = df_preds_train.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_rmse, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_train = df_train.to_frame().rename(columns = {col_true: metric})
    df_valid = df_preds_valid.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_rmse, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_valid = df_valid.to_frame().rename(columns = {col_true: metric})
    if calculate_overall:
        df_train.loc['mean', metric] = df_train[metric].mean()
        df_valid.loc['mean', metric] = df_valid[metric].mean()
    list_df_trains.append(df_train)
    list_df_valids.append(df_valid)

    # MAE
    metric = 'mae'
    df_train = df_preds_train.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_mae, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_train = df_train.to_frame().rename(columns = {col_true: metric})
    df_valid = df_preds_valid.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_mae, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_valid = df_valid.to_frame().rename(columns = {col_true: metric})
    if calculate_overall:
        df_train.loc['mean', metric] = df_train[metric].mean()
        df_valid.loc['mean', metric] = df_valid[metric].mean()
    list_df_trains.append(df_train)
    list_df_valids.append(df_valid)

    # R2
    metric = 'r2'
    df_train = df_preds_train.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_r2, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_train = df_train.to_frame().rename(columns = {col_true: metric})
    df_valid = df_preds_valid.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_r2, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_valid = df_valid.to_frame().rename(columns = {col_true: metric})
    if calculate_overall:
        df_train.loc['mean', metric] = df_train[metric].mean()
        df_valid.loc['mean', metric] = df_valid[metric].mean()
    list_df_trains.append(df_train)
    list_df_valids.append(df_valid)

    # Pearson correlation
    metric = 'pearson'
    df_train = df_preds_train.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_pearson, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_train = df_train.to_frame().rename(columns = {col_true: metric})
    df_valid = df_preds_valid.groupby(['fold'])[[col_true, col_pred]].agg(_calculate_pearson, 
                                                                          col_true=col_true, 
                                                                          col_pred=col_pred)[col_true]
    df_valid = df_valid.to_frame().rename(columns = {col_true: metric})
    if calculate_overall:
        df_train.loc['mean', metric] = df_train[metric].mean()
        df_valid.loc['mean', metric] = df_valid[metric].mean()
    list_df_trains.append(df_train)
    list_df_valids.append(df_valid)

    # concatenate
    df_trains = pd.concat(list_df_trains, axis=1)
    df_trains['set'] = 'train'
    df_valids = pd.concat(list_df_valids, axis=1)
    df_valids['set'] = 'valid'

    df_error = pd.concat((df_trains, df_valids), axis=0).reset_index()
    df_error = df_error.rename(columns={'index': 'fold'})
    
    return df_error
