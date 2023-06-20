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

from missing_util import load_data_index
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer







dataset = sys.argv[1] # dataset = ["banknote","concrete_compression","wine_quality_white","wine_quality_red"]

mechanism = sys.argv[2]



def run_baseline():
    RMSE_mean = []
    RMSE_MICE = []
    data_shape, dl, train_idx, test_idx, valid_idx, full_data, mask = load_data_index(dataset, mechanism, rule_name)

    print(train_idx, test_idx, valid_idx)

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


    # ------------------------- #
    # ---- mean imputation ---- #
    # ------------------------- #
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(Xnan)
    Ximp = imp.transform(X_test_nan)
    RMSE_mean.append(np.sqrt(np.sum((Xtest - Ximp) ** 2 * (1 - S_test)) / np.sum(1 - S_test)))

    # ------------------------- #
    # ---- mean imputation ---- #
    # ------------------------- #
    imp_MICE = IterativeImputer(random_state=0)
    imp_MICE.fit(Xnan)
    imp_MICE.transform(X_test_nan)
    RMSE_MICE.append(np.sqrt(np.sum((Xtest - Ximp) ** 2 * (1 - S_test)) / np.sum(1 - S_test)))





if mechanism == "quantile":
    json_name = sys.argv[3]
    # json_name = ["q1_quantile","q2_quantile","quantile","three_block_quantile", "complete"]
    missing_rule = load_json_file(json_name + ".json")


    for rule_name in missing_rule:
        multiple_block = missing_rule[rule_name]
        
        run_baseline(multiple_block,rule_name)


