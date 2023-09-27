"""
Use the MIWAE and not-MIWAE on UCI data
"""
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import argparse
from tqdm import tqdm
sys.path.append(os.getcwd())
from hyperimpute.plugins.imputers import Imputers, ImputerPlugin
from hyperimpute.plugins.utils.metrics import RMSE
from util import load_data_index

import json

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
#from missing_process.block_rules import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"




def run_one(args, rule_name):

    imputers = Imputers()

    data_shape, dl, train_idx, test_idx, valid_idx,full_data, mask = load_data_index(args,rule_name)



    n, d = data_shape
    print(n,d)
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


    # Xtest = torch.from_numpy(Xtest)
    # X_test_nan = torch.from_numpy(X_test_nan)

    # nan_columns = np.all(np.isnan(Xnan), axis=0)
    # if np.any(nan_columns):
    #     random_value = np.random.uniform(0, 1)
    #     Xnan[:, nan_columns] = random_value 


    nan_columns = np.all(np.isnan(X_test_nan), axis=0)
    if np.any(nan_columns):
        random_value = np.random.uniform(0, 1)
        X_test_nan[:, nan_columns] = random_value 

    # ------------------- #
    # ---- Sinkhorn ---- #
    # ------------------- #
    ctx = imputers.get("hyperimpute")
    x_imp = ctx.fit_transform(X_test_nan)

 

    print(rule_name,"Hyper Impute done")
    hyper_rmse =  np.sqrt(np.sum((Xtest - x_imp.values) ** 2 * (1 - Xtest_mask)) / np.sum(1 - Xtest_mask))

    x_imp.to_csv("../results/hyperimputer/Imputation_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)




    # --------------------------- #
    # ----miracle---- #
    # -------------------------- #
    #Create the imputation models
    ctx = imputers.get("miracle")
    x_imp = ctx.fit_transform(X_test_nan)
    
    print(rule_name,"miracle Impute done")
    miracle_rmse = np.sqrt(np.sum((Xtest - x_imp.values) ** 2 * (1 - Xtest_mask)) / np.sum(1 - Xtest_mask))
    x_imp.to_csv("../results/miracle/Imputation_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)






    return hyper_rmse, miracle_rmse





parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type = str, default ="wine_quality_red" )
parser.add_argument("--missingtype", type=str, default="diffuse")
parser.add_argument("--missingpara", type=str, default="diffuse_ratio")
args = parser.parse_args()



def load_json_file(filename):
    json_path = os.path.join("../missing_process/block_rules", filename)
    with open(json_path) as f:
        return json.load(f)
    

datalist = ["banknote","concrete_compression",
            "wine_quality_white","wine_quality_red",
            "california","climate_model_crashes",
            "connectionist_bench_sonar","qsar_biodegradation",
            "yeast","yacht_hydrodynamics"
            ]


datalist = ["climate_model_crashes",
            "connectionist_bench_sonar","qsar_biodegradation",
            "yeast","yacht_hydrodynamics"
            ]
d = {"logistic":"missing_rate","diffuse":"diffuse_ratio","quantile":"quantile_full"}

missingtypelist = ["quantile"]

for dataset in tqdm(datalist):
        if dataset == "california":
            missingtypelist = ["quantile","diffuse"]

        for missingtype in missingtypelist:
            if missingtype == "logistic":
                missing_rule = load_json_file("missing_rate.json")
            elif missingtype == "diffuse":
                missing_rule = load_json_file("diffuse_ratio.json")
            elif missingtype == "quantile":
                missing_rule = load_json_file("quantile_full.json")

            rule_list = []
            hyper_rmse_list =  []
            miracle_rmse_list =  []


            for rule_name in tqdm(missing_rule):


                args.dataset = dataset
                args.missingtype = missingtype  
                args.missingpara = d[missingtype]

                hyper_rmse, miracle_rmse = run_one(args, rule_name)


                hyper_rmse_list.append(hyper_rmse)
                miracle_rmse_list.append(miracle_rmse)


                rule_list.append(rule_name)

                

            result = pd.DataFrame({"Missing_Rule":rule_list,"hyper_RMSE":hyper_rmse_list})
            result.to_csv("../results/hyperimputer/RMSE_{}_{}.csv".format(args.dataset,args.missingpara),index=False)


            result = pd.DataFrame({"Missing_Rule":rule_list,"miracle_RMSE":miracle_rmse_list})
            result.to_csv("../results/miracle/RMSE_{}_{}.csv".format(args.dataset,args.missingpara),index=False)


            print("Experiment completed")

