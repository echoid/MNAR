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
from missing_util import introduce_mising, preprocessing,introduce_mising_advanced
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
runs = 1
RMSE_miwae1 = []
RMSE_notmiwae1 = []
RMSE_notmiwae_selfmasking1 = []
RMSE_notmiwae_linear1 = []
RMSE_mean1 = []
RMSE_mice1 = []
RMSE_RF1 = []

for _ in range(runs):

    data_shape, Xtrain,Xval_org, dl = preprocessing("url1")

    # ---- introduce missing process
    #Xnan, Xz = introduce_mising(Xtrain)
    Xnan, Xz =introduce_mising_advanced(Xtrain, 0.75 , "MNAR","logistic")
    # mask
    S = np.array(~np.isnan(Xnan), dtype=np.float)
    Xval, Xvalz = introduce_mising(Xval_org)
    #Xval, Xvalz = introduce_mising_advanced(Xtrain, 0.8 , "MNAR")

    # ------------------- #
    # ---- fit MIWAE ---- #
    # ------------------- #
    miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, name=name)

    # ---- do the training
    trainer.train(miwae, batch_size=batch_size, max_iter=max_iter, name=name + 'miwae')

    # ---- find imputation RMSE
    RMSE_miwae1.append(utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)[0])

    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking_known", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae1.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_selfmasking1.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="linear", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_linear1.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])



    # # ------------------------- #
    # # ---- mean imputation ---- #
    # # ------------------------- #
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(Xnan)
    # Xrec = imp.transform(Xnan)
    # RMSE_mean1.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))

print("URL1")
#print("RMSE_mean = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_mean1), np.std(RMSE_mean)))
print("Data Shape: ", data_shape)
print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae1), np.std(RMSE_miwae1)))
print("RMSE_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae1), np.std(RMSE_notmiwae1)))
print("RMSE_notmiwae selfmasking = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_selfmasking1), np.std(RMSE_notmiwae_selfmasking1)))
print("RMSE_notmiwae linear = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_linear1), np.std(RMSE_notmiwae_linear1)))


# ---- number of runs
runs = 1
RMSE_miwae2 = []
RMSE_notmiwae2 = []
RMSE_notmiwae_selfmasking2 = []
RMSE_notmiwae_linear2 = []
RMSE_mean2 = []
RMSE_mice2 = []
RMSE_RF2 = []

for _ in range(runs):

    data_shape, Xtrain,Xval_org, dl = preprocessing("url2")

    # ---- introduce missing process
    #Xnan, Xz = introduce_mising(Xtrain)
    Xnan, Xz =introduce_mising_advanced(Xtrain, 0.75 , "MNAR","logistic")
    # mask
    S = np.array(~np.isnan(Xnan), dtype=np.float)
    Xval, Xvalz = introduce_mising(Xval_org)
    #Xval, Xvalz = introduce_mising_advanced(Xtrain, 0.8 , "MNAR")

    # ------------------- #
    # ---- fit MIWAE ---- #
    # ------------------- #
    miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, name=name)

    # ---- do the training
    trainer.train(miwae, batch_size=batch_size, max_iter=max_iter, name=name + 'miwae')

    # ---- find imputation RMSE
    RMSE_miwae2.append(utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)[0])

    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking_known", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae2.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_selfmasking2.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="linear", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_linear2.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])



print("URL2")
print("Data Shape: ", data_shape)
print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae2), np.std(RMSE_miwae2)))
print("RMSE_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae2), np.std(RMSE_notmiwae2)))
print("RMSE_notmiwae selfmasking = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_selfmasking2), np.std(RMSE_notmiwae_selfmasking2)))
print("RMSE_notmiwae linear = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_linear2), np.std(RMSE_notmiwae_linear2)))



# ---- number of runs
runs = 1
RMSE_miwae3 = []
RMSE_notmiwae3 = []
RMSE_notmiwae_selfmasking3 = []
RMSE_notmiwae_linear3 = []
RMSE_mean3 = []
RMSE_mice3 = []
RMSE_RF3 = []

for _ in range(runs):

    data_shape, Xtrain,Xval_org, dl = preprocessing("url3")

    # ---- introduce missing process
    #Xnan, Xz = introduce_mising(Xtrain)
    Xnan, Xz =introduce_mising_advanced(Xtrain, 0.75 , "MNAR","logistic")
    # mask
    S = np.array(~np.isnan(Xnan), dtype=np.float)
    Xval, Xvalz = introduce_mising(Xval_org)
    #Xval, Xvalz = introduce_mising_advanced(Xtrain, 0.8 , "MNAR")

    # ------------------- #
    # ---- fit MIWAE ---- #
    # ------------------- #
    miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, name=name)

    # ---- do the training
    trainer.train(miwae, batch_size=batch_size, max_iter=max_iter, name=name + 'miwae')

    # ---- find imputation RMSE
    RMSE_miwae3.append(utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)[0])

    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking_known", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae3.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_selfmasking3.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="linear", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_linear3.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])




print("URL3")
print("Data Shape: ", data_shape)
print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae3), np.std(RMSE_miwa3e)))
print("RMSE_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae), np.std(RMSE_notmiwae)))
print("RMSE_notmiwae selfmasking = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_selfmasking), np.std(RMSE_notmiwae_selfmasking)))
print("RMSE_notmiwae linear = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_linear), np.std(RMSE_notmiwae_linear)))



# ---- number of runs
runs = 1
RMSE_miwae = []
RMSE_notmiwae = []
RMSE_notmiwae_selfmasking = []
RMSE_notmiwae_linear = []
RMSE_mean = []
RMSE_mice = []
RMSE_RF = []

for _ in range(runs):

    data_shape, Xtrain,Xval_org, dl = preprocessing("url4")

    # ---- introduce missing process
    #Xnan, Xz = introduce_mising(Xtrain)
    Xnan, Xz =introduce_mising_advanced(Xtrain, 0.75 , "MNAR","logistic")
    # mask
    S = np.array(~np.isnan(Xnan), dtype=np.float)
    Xval, Xvalz = introduce_mising(Xval_org)
    #Xval, Xvalz = introduce_mising_advanced(Xtrain, 0.8 , "MNAR")

    # ------------------- #
    # ---- fit MIWAE ---- #
    # ------------------- #
    miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, name=name)

    # ---- do the training
    trainer.train(miwae, batch_size=batch_size, max_iter=max_iter, name=name + 'miwae')

    # ---- find imputation RMSE
    RMSE_miwae.append(utils.imputationRMSE(miwae, Xtrain, Xz, Xnan, S, L)[0])

    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking_known", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="selfmasking", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_selfmasking.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])


    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process="linear", name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    RMSE_notmiwae_linear.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])



    # ------------------------- #
    # ---- mean imputation ---- #
    # ------------------------- #
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(Xnan)
    Xrec = imp.transform(Xnan)
    RMSE_mean.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))





print("URL4")
print("RMSE_mean = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_mean), np.std(RMSE_mean)))
print("Data Shape: ", data_shape)
print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae), np.std(RMSE_miwae)))
print("RMSE_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae), np.std(RMSE_notmiwae)))
print("RMSE_notmiwae selfmasking = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_selfmasking), np.std(RMSE_notmiwae_selfmasking)))
print("RMSE_notmiwae linear = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae_linear), np.std(RMSE_notmiwae_linear)))