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
from missing_util import preprocessing,introduce_mising_advanced,missing_by_range
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
# dataset = ["banknote","concrete","white","red"]
mechanism = sys.argv[2]
json_name = sys.argv[3]
# mechanism = ["MAR","MCAR","MNAR"]


# multiple_block = {1:{"lower": 0.5,"upper":1,"partial_missing":0.}
#                   #,2:{"lower": 0.5,"upper":0.6,"partial_missing":0.}
#                   #,3:{"lower": 0.8,"upper":0.9,"partial_missing":0.}
#                   }

with open('missing_mech/{}.json'.format(json_name)) as f:
    multiple_blocks = json.load(f)


def run_one(multiple_block):

    RMSE_miwae = []
    RMSE_notmiwae = []
    RMSE_notmiwae_selfmasking = []
    RMSE_notmiwae_linear = []
    RMSE_mean = []


    # for _ in range(runs):

    #     data_shape, Xtrain,Xval_org, dl = preprocessing(dataset)

    #     # ---- introduce missing process
    #     Xnan, Xz = missing_by_range(Xtrain, multiple_block)
    #     #Xnan, Xz =introduce_mising_advanced(Xtrain, 0.75 , "MNAR")
    #     # mask
    #     S = np.array(~np.isnan(Xnan), dtype=np.float)
    #     Xval, Xvalz = missing_by_range(Xtrain, multiple_block)
    #     #Xval, Xvalz = introduce_mising_advanced(Xtrain, 0.8 , "MNAR")

    #     # ------------------- #
    #     # ---- fit MIWAE ---- #
    #     # ------------------- #
    #     miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, name=name)

    #     # ---- do the training
    #     trainer.train(miwae, batch_size=batch_size, max_iter=max_iter, name=name + 'miwae')

    #     # ---- find imputation RMSE
    #     RMSE_miwae.append(utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)[0])

    #     # ---------------------- #
    #     # ---- fit not-MIWAE---- #
    #     # ---------------------- #
    #     notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking_known", name=name)

    #     # ---- do the training
    #     trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    #     # ---- find imputation RMSE
    #     RMSE_notmiwae.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    #     # ---------------------- #
    #     # ---- fit not-MIWAE---- #
    #     # ---------------------- #
    #     notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking", name=name)

    #     # ---- do the training
    #     trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    #     # ---- find imputation RMSE
    #     RMSE_notmiwae_selfmasking.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    #     # ---------------------- #
    #     # ---- fit not-MIWAE---- #
    #     # ---------------------- #
    #     notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="linear", name=name)

    #     # ---- do the training
    #     trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    #     # ---- find imputation RMSE
    #     RMSE_notmiwae_linear.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])



    #     # ------------------------- #
    #     # ---- mean imputation ---- #
    #     # ------------------------- #
    #     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #     imp.fit(Xnan)
    #     Xrec = imp.transform(Xnan)
    #     RMSE_mean.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))


    print("Data Set:",dataset, mechanism)
    #print("Data Shape: ", data_shape)
    print("Missing Block", multiple_block)

    print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae), np.std(RMSE_miwae)))
    print("RMSE_mean = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_mean), np.std(RMSE_mean)))
    print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae), np.std(RMSE_miwae)))
    print("RMSE_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae), np.std(RMSE_notmiwae)))
    print("RMSE_notmiwae selfmasking = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_selfmasking), np.std(RMSE_notmiwae_selfmasking)))
    print("RMSE_notmiwae linear = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_linear), np.std(RMSE_notmiwae_linear)))



for block in multiple_blocks:
    multiple_block = multiple_blocks[block]
    run_one(multiple_block)
