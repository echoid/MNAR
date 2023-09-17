"""
Use the MIWAE and not-MIWAE on UCI data
"""
import numpy as np
import pandas as pd
import os
import sys
import torch.nn as nn
import argparse
sys.path.append(os.getcwd())
from imputers import OTimputer, RRimputer
from utils import *
import json

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
#from missing_process.block_rules import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"




def run_one(args, rule_name):


    
    print("Before I go to",os.getcwd())
    data_shape, dl, train_idx, test_idx, valid_idx,full_data, mask = load_data_index(args,rule_name)


    n, d = data_shape
        # If the batch size is larger than half the dataset's size,
                    # it will be redefined in the imputation methods.
    


    Xtrain = full_data[train_idx]
    Xtest = full_data[test_idx]
    Xval_org = full_data[valid_idx]

    Xtrain_mask = mask[train_idx]
    Xtest_mask = mask[test_idx]
    Xval_org_mask = mask[valid_idx]

    # Xnan = Xtrain.copy()
    # Xz = Xtrain.copy()
    # Xnan[Xtrain_mask == 0] = np.nan
    # Xz[Xtrain_mask == 0] = 0
    # S = np.array(~np.isnan(Xnan), dtype=np.float)
    

    # Xval  = Xval_org.copy()
    # Xvalz = Xval_org.copy()
    # Xval[Xval_org_mask == 0] = np.nan
    # Xvalz[Xval_org_mask == 0] = 0

    X_test_nan = Xtest.copy()
    X_test_z = Xtest.copy()
    X_test_nan[Xtest_mask == 0] = np.nan
    X_test_z[Xtest_mask == 0] = 0
    S_test = np.array(~np.isnan(X_test_nan), dtype=np.float)

    Xtest = torch.from_numpy(Xtest)
    X_test_nan = torch.from_numpy(X_test_nan)

    epsilon = pick_epsilon(X_test_nan)
    

    # ------------------- #
    # ---- Sinkhorn ---- #
    # ------------------- #

    sk_imputer = OTimputer(eps=epsilon, batchsize=args.batchsize, lr=args.lr, niter=args.max_iter)
    sk_imp, sk_maes, sk_rmses = sk_imputer.fit_transform(X_test_nan, verbose=True, report_interval=500, X_true=Xtest)
    pd.DataFrame(sk_imp.detach().numpy()).to_csv("../results/ot_sk/Imputation_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)





    # --------------------------- #
    # ----Linear round-robin---- #
    # -------------------------- #
    #Create the imputation models
    # d_ = d - 1
    # models = {}

    # for i in range(d):
    #     models[i] = nn.Linear(d_, 1)

    # #Create the imputer
    # lin_rr_imputer = RRimputer(models, eps=epsilon, lr=args.lr)
    # lin_imp, lin_maes, lin_rmses = lin_rr_imputer.fit_transform(X_test_nan, verbose=True, X_true=Xtest)
    # pd.DataFrame(lin_imp.detach().numpy()).to_csv("../results/ot_linear/Imputation_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)
    lin_rmses = [0]



    # # --------------------------- #
    # # ----MLP round-robin   ---- #
    # # -------------------------- #

    # #Create the imputation models
    # d_ = d - 1
    # models = {}

    # for i in range(d):
    #     models[i] = nn.Sequential(nn.Linear(d_, 2 * d_),
    #                             nn.ReLU(),
    #                             nn.Linear(2 * d_, d_),
    #                             nn.ReLU(),
    #                             nn.Linear(d_, 1))

    # #Create the imputer
    # mlp_rr_imputer = RRimputer(models, eps=epsilon, lr=args.lr)
    # mlp_imp, mlp_maes, mlp_rmses = mlp_rr_imputer.fit_transform(X_test_nan, verbose=True, X_true=Xtest)
    # pd.DataFrame(mlp_imp.detach().numpy()).to_csv("../results/ot_mlp/Imputation_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)

    mlp_rmses = [0]



    return sk_rmses[-1],lin_rmses[-1],mlp_rmses[-1]





parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=int, default=1e-2)
parser.add_argument('--max_iter', type=int, default=5000)
# parser.add_argument('--n_samples', type=float, default=20)
# parser.add_argument('--n_hidden', type=float, default=128)
# parser.add_argument('--runs', type=int, default = 5)


parser.add_argument('--dataset', type = str, default ="wine_quality_red" )
parser.add_argument("--missingtype", type=str, default="diffuse")
parser.add_argument("--missingpara", type=str, default="diffuse_ratio")
args = parser.parse_args()



def load_json_file(filename):
    json_path = os.path.join("../missing_process/block_rules", filename)
    with open(json_path) as f:
        return json.load(f)
    

print(os.getcwd())
missing_rule = load_json_file(args.missingpara + ".json")
rule_list = []
sk_rmse_list =  []
lin_rmse_list =  []
mlp_rmselist =  []


for rule_name in missing_rule:
    sk_rmses,lin_rmses,mlp_rmses = run_one(args, rule_name)


    sk_rmse_list.append(sk_rmses)
    lin_rmse_list.append(lin_rmses)
    mlp_rmselist.append(mlp_rmses)
   

    rule_list.append(rule_name)

result = pd.DataFrame({"Missing_Rule":rule_list,"OT_SK_RMSE":sk_rmse_list,"OT_linear_RMSE":lin_rmses,"OT_mlp_RMSE":mlp_rmses})
result.to_csv("../results/ot_sk/RMSE_{}_{}.csv".format(args.dataset,args.missingpara),index=False)
print("Experiment completed")

