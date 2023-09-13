"""
Use the MIWAE and not-MIWAE on UCI data
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
sys.path.append(os.getcwd())
from MIWAE import MIWAE
from notMIWAE import notMIWAE
import trainer
import utils
import json

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
#from missing_process.block_rules import *
from missing_util import load_data_index

os.environ["CUDA_VISIBLE_DEVICES"] = "3"




def run_one(args, rule_name):

    RMSE_miwae = []
    RMSE_notmiwae = []



    for run in range(args.runs):
        print("Before I go to",os.getcwd())
        data_shape, dl, train_idx, test_idx, valid_idx,full_data, mask = load_data_index(args,rule_name)


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


        

        # ------------------- #
        # ---- fit MIWAE ---- #
        # ------------------- #
        miwae = MIWAE(Xnan, Xval, n_latent=dl, n_samples=args.n_samples, n_hidden=args.n_hidden, name="miwae")
        # ---- do the training
        trainer.train(miwae, batch_size=args.batch_size, max_iter=args.max_iter, name='miwae')
        # ---- find imputation RMSE
        rmse, Ximp = utils.imputationRMSE(miwae, Xtest, X_test_z, X_test_nan, S_test, args.L)
        pd.DataFrame(Ximp).to_csv("results/miwae/Imputation_{}_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name,run),index=False)
        RMSE_miwae.append(rmse)
        print("First RMSE",rmse)

        mse, Ximp = utils.imputationRMSE(miwae, Xtest, X_test_z, X_test_nan, S_test, args.L)

        print("Second RMSE",rmse)



        # ---------------------- #
        # ---- fit not-MIWAE---- #
        # ---------------------- #
        notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=args.n_samples, n_hidden=args.n_hidden, missing_process="selfmasking_known", name="notmiwae")

        # ---- do the training
        trainer.train(notmiwae, batch_size=args.batch_size, max_iter=args.max_iter, name='notmiwae')

        # ---- find imputation RMSE
        rmse,Ximp = utils.not_imputationRMSE(notmiwae, Xtest, X_test_z, X_test_nan, S_test, args.L)
        pd.DataFrame(Ximp).to_csv("results/notmiwae/Imputation_{}_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name,run),index=False)
        RMSE_notmiwae.append(rmse)
        print("NOTMIWAE - First RMSE",rmse)

        mse, Ximp = utils.imputationRMSE(miwae, Xtest, X_test_z, X_test_nan, S_test, args.L)

        print("NOT-MIWAE - Second RMSE",rmse)





    #print("RMSE_mean = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_mean), np.std(RMSE_mean)))
    print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae), np.std(RMSE_miwae)))
    print("RMSE_notmiwae selfmasking_known = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae), np.std(RMSE_notmiwae)))


    return np.mean(RMSE_miwae), np.std(RMSE_miwae), np.mean(RMSE_notmiwae), np.std(RMSE_notmiwae)





parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--L', type=int, default=10000)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--n_samples', type=float, default=20)
parser.add_argument('--n_hidden', type=float, default=128)
parser.add_argument('--runs', type=int, default = 5)


parser.add_argument('--dataset', type = str, default ="wine_quality_red" )
parser.add_argument("--missingtype", type=str, default="diffuse")
parser.add_argument("--missingpara", type=str, default="diffuse_ratio")
args = parser.parse_args()



def load_json_file(filename):
    json_path = os.path.join("missing_process/block_rules", filename)
    with open(json_path) as f:
        return json.load(f)
print(os.getcwd())
missing_rule = load_json_file(args.missingpara + ".json")
rule_list = []
notmiwae_rmse_list =  []
notmiwae_rmse_std_list =  []
miwae_rmse_list =  []
miwae_rmse_std_list =  []


for rule_name in missing_rule:
    RMSE_miwae, RMSE_miwae_std, RMSE_notmiwae,RMSE_notmiwae_std = run_one(args, rule_name)


    miwae_rmse_list.append(RMSE_miwae)
    miwae_rmse_std_list.append(RMSE_miwae_std)
    notmiwae_rmse_list.append(RMSE_notmiwae)
    notmiwae_rmse_std_list.append(RMSE_notmiwae_std)


    rule_list.append(rule_name)

    exit()

result = pd.DataFrame({"Missing_Rule":rule_list,"MIWAE_RMSE":miwae_rmse_list,"MIWAE_STD":miwae_rmse_std_list})
result.to_csv("results/miwae/RMSE_{}_{}.csv".format(args.dataset,args.missingpara),index=False)


result = pd.DataFrame({"Missing_Rule":rule_list,"NotMIWAE_RMSE":notmiwae_rmse_list,"NotMIWAE_STD":notmiwae_rmse_std_list})
result.to_csv("results/notmiwae/RMSE_{}_{}.csv".format(args.dataset,args.missingpara),index=False)
print("Experiment completed")