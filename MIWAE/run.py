"""
Use the MIWAE and not-MIWAE on UCI data
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from MIWAE import MIWAE
from notMIWAE import notMIWAE
import trainer
import utils

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
from missing_process.block_rules import *

from missing_util import preprocessing,missing_by_range,OT_missing,load_data_index
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor





# ---- data settings
name = '/tmp/uci/task01/best'
n_hidden = 128
n_samples = 20
max_iter = 100000
batch_size = 16
L = 10000

# ---- choose the missing model
# mprocess = 'linear'
# mprocess = 'selfmasking'
mprocess = 'selfmasking_known'



url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

url3 = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

url4= "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


# ---- number of runs
runs = 3



dataset = sys.argv[1]
# dataset = ["banknote","concrete_compression","wine_quality_white","wine_quality_red"]
mechanism = sys.argv[2]
# mechanism = ["quantile" "MAR","MCAR","OTquantile","OTlogistic","OTselfmask"]





def run_one(multiple_block=None,rule_name = None):

    RMSE_miwae = []
    RMSE_notmiwae = []
    RMSE_notmiwae_selfmasking = []
    RMSE_notmiwae_linear = []
    RMSE_mean = []

    # ML_miwae = []
    # ML_notmiwae = []
    # ML_notmiwae_selfmasking = []
    # ML_notmiwae_linear = []
    # ML_mean = []
    # ML_original = []


    for _ in range(runs):

        rule_name = "Q4_complete"
        #data_shape, Xtrain, Xval_org, Xtest, Ytrain, Yval_org , Ytest, dl = preprocessing(dataset)
        data_shape, dl, train_idx, test_idx, valid_idx,full_data, mask = load_data_index(dataset, mechanism, rule_name)

        print(len(train_idx), len(test_idx), len(valid_idx))
        print(full_data.shape)


        Xtrain = full_data[train_idx]
        Xtest = full_data[test_idx]
        Xval_org = full_data[valid_idx]

        Xtrain_mask = mask[train_idx]
        Xtest_mask = mask[test_idx]
        Xval_org_mask = mask[valid_idx]

        Xnan = Xtrain.copy()
        Xz = Xtrain.copy()
        Xnan[Xtrain_mask == 0] = np.nan
        Xz[Xtrain_mask == 0] = 0
        S = np.array(~np.isnan(Xnan), dtype=np.float)
        

        Xval  = Xval_org.copy()
        Xvalz = Xval_org.copy()
        Xval[Xval_org_mask == 0] = np.nan
        Xvalz[Xval_org_mask == 0] = 0

        X_test_nan = Xtest.copy()
        X_test_z = Xtest.copy()
        X_test_nan[Xtest_mask == 0] = np.nan
        X_test_z[Xtest_mask == 0] = 0
        S_test = np.array(~np.isnan(X_test_nan), dtype=np.float)

        print(Xtrain)
        exit()
        # # ---- introduce missing process
        # if mechanism == "quantile":
        #     Xnan, Xz = missing_by_range(Xtrain, multiple_block)

            
        #     # mask
        #     S = np.array(~np.isnan(Xnan), dtype=np.float)
        #     #Xval, Xvalz = missing_by_range(Xval_org, multiple_block)
        #     Xval, Xvalz = missing_by_range(Xtrain, multiple_block)
        
        # else:
            
        #     Xnan, Xz = OT_missing(Xtrain, 0.5, mechanism, p_obs=0.2, q=0.2)
        #     S = np.array(~np.isnan(Xnan), dtype=np.float)
        #     Xval, Xvalz = OT_missing(Xval_org, 0.5, mechanism, p_obs=0.2, q=0.2)
        

        # ------------------- #
        # ---- fit MIWAE ---- #
        # ------------------- #
        miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, name=name)

        # ---- do the training
        trainer.train(miwae, batch_size=batch_size, max_iter=max_iter, name=name + 'miwae')

        # ---- find imputation RMSE
        #RMSE_miwae.append(utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)[0])
        #rmse, Ximp = utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)
        rmse, Ximp = utils.imputationRMSE(miwae, Xtest, X_test_z, X_test_nan, S_test, L)
        RMSE_miwae.append(rmse)
        # ---- find imputation ML util
        #ML_miwae.append(utils.downstream(Ximp, Ytrain, Xtest, Ytest, dataset))




        # ---------------------- #
        # ---- fit not-MIWAE---- #
        # ---------------------- #
        notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking_known", name=name)

        # ---- do the training
        trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

        # ---- find imputation RMSE
        #rmse,Ximp = utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)
        rmse, Ximp = utils.imputationRMSE(notmiwae, Xtest, X_test_z, X_test_nan, S_test, L)
        RMSE_notmiwae.append(rmse)
        #ML_notmiwae.append(utils.downstream(Ximp, Ytrain, Xtest, Ytest, dataset))


        # ---------------------- #
        # ---- fit not-MIWAE---- #
        # ---------------------- #
        notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking", name=name)

        # ---- do the training
        trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

        # ---- find imputation RMSE
        #rmse,Ximp = utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)
        rmse, Ximp = utils.imputationRMSE(notmiwae, Xtest, X_test_z, X_test_nan, S_test, L)
        RMSE_notmiwae_selfmasking.append(rmse)
        #ML_notmiwae_selfmasking.append(utils.downstream(Ximp, Ytrain, Xtest, Ytest, dataset))

        # ---------------------- #
        # ---- fit not-MIWAE---- #
        # ---------------------- #
        notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="linear", name=name)

        # ---- do the training
        trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

        # ---- find imputation RMSE
        #rmse,Ximp = utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)
        rmse, Ximp = utils.imputationRMSE(notmiwae, Xtest, X_test_z, X_test_nan, S_test, L)
        RMSE_notmiwae_linear.append(rmse)
        #ML_notmiwae_linear.append(utils.downstream(Ximp, Ytrain, Xtest, Ytest, dataset))




        # ------------------------- #
        # ---- mean imputation ---- #
        # ------------------------- #
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(Xnan)
        Ximp = imp.transform(X_test_nan)
        RMSE_mean.append(np.sqrt(np.sum((Xtest - Ximp) ** 2 * (1 - S_test)) / np.sum(1 - S_test)))
        
        #ML_mean.append(utils.downstream(Ximp, Ytrain, Xtest, Ytest, dataset))

        # ------------------------- #
        # ----- original data ----- #
        # ------------------------- #

        #ML_original.append(utils.downstream(Xtrain, Ytrain, Xtest, Ytest, dataset))


    print("Data Set:",dataset, mechanism)
    print("Data Shape: ", data_shape)
    if mechanism == "quantile":
        print("Missing Block", multiple_block)

    print("RMSE_mean = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_mean), np.std(RMSE_mean)))
    print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae), np.std(RMSE_miwae)))
    print("RMSE_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae), np.std(RMSE_notmiwae)))
    print("RMSE_notmiwae selfmasking = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_selfmasking), np.std(RMSE_notmiwae_selfmasking)))
    print("RMSE_notmiwae linear = {0:.5f} +- {1:.5f}\n\n".format(np.mean(RMSE_notmiwae_linear), np.std(RMSE_notmiwae_linear)))



    # print("ML_original = {0:.5f} +- {1:.5f}".format(np.mean(ML_original), np.std(ML_original)))
    # print("ML_mean = {0:.5f} +- {1:.5f}".format(np.mean(ML_mean), np.std(ML_mean)))
    # print("ML_miwae = {0:.5f} +- {1:.5f}".format(np.mean(ML_miwae), np.std(ML_miwae)))
    # print("ML_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(ML_notmiwae), np.std(ML_notmiwae)))
    # print("ML_notmiwae selfmasking = {0:.5f} +- {1:.5f}".format(np.mean(ML_notmiwae_selfmasking), np.std(ML_notmiwae_selfmasking)))
    # print("ML_notmiwae linear = {0:.5f} +- {1:.5f}".format(np.mean(ML_notmiwae_linear), np.std(ML_notmiwae_linear)))


if mechanism == "quantile":
    json_name = sys.argv[3]
    # json_name = ["double_quantile_1","double_quantile_2","single_quantile"]
    missing_rule = load_json_file(json_name + ".json")


    for rule_name in missing_rule:
        multiple_block = missing_rule[rule_name]
        
        run_one(multiple_block,rule_name)


else:
    run_one()
