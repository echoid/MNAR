import argparse
import torch
import datetime
import json
import yaml
import os
import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
import pickle
from missing_process.block_rules import *

#dataname = "wine_quality_white"


from src.main_model_table import TabCSDI
from src.utils_table_new import train, evaluate
from mean_loader import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--dataset",type = str, default ="wine_quality_red" )
parser.add_argument("--config", type=str, default="test") # test
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

parser.add_argument("--missingtype", type=str, default="MCAR")
parser.add_argument("--missingpara", type=str, default="simple_rule")
#parser.add_argument("--testmissingratio", type=float, default=0.1)


args = parser.parse_args()



#print(json.dumps(config, indent=4))




missing_rule = load_json_file(args.missingpara + ".json")


for rule_name in missing_rule:
    rule = missing_rule[rule_name]
    print("========")
    print("Current Rule",rule )
    # Create folder



    

    #Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
    train_loader, valid_loader, test_loader = get_dataloader(
        dataname=args.dataset,
        seed=args.seed,
        nfold=args.nfold,
        batch_size=128,
        missing_type = args.missingtype,
        missing_para = rule,
        missing_name = rule_name

    )

    # #----------------------------------------------------------------------------------------------
    # processed_data_path_norm = "data_census_onehot/missing_ratio-0.2_seed-1_max-min_norm.pk"
    # with open(processed_data_path_norm, "rb") as f:
    #     observed_values, observed_masks, gt_masks = pickle.load(
    #                 f
    #             )
    # seed = 1
    # nfold = 5


    # observed_values = observed_values
    # values_nan = observed_values.copy()
    # values_nan[gt_masks == 0] = np.nan
    # indlist = np.arange(observed_values.shape[0])

    # np.random.seed(seed + 1)
    # np.random.shuffle(indlist)
    # tmp_ratio = 1 / nfold
    # start = (int)((nfold - 1) * observed_values.shape[0] * tmp_ratio)
    # end = (int)(nfold * observed_values.shape[0] * tmp_ratio)
    # test_index = indlist[start:end]
    # remain_index = np.delete(indlist, np.arange(start, end))

    # np.random.shuffle(remain_index)

    #     # Modify here to change train,valid ratio
    # num_train = (int)(len(remain_index) * 1)
    # train_index = remain_index[:num_train]
    # valid_index = remain_index[num_train:]


    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(values_nan[train_index,:])
    # train_imp = imp.transform(values_nan[train_index,:])
    # test_imp = imp.transform(values_nan[test_index,:])
    # S_train = np.array(~np.isnan(values_nan[train_index,:]), dtype=float)
    # S_test = np.array(~np.isnan(values_nan[test_index,:]), dtype=float)

    # test_rmse = np.sqrt(np.sum((observed_values[test_index,:] - test_imp) ** 2 * (1 - S_test)) / np.sum(1 - S_test))
    # print("Mean Imputation Test RMSE: " , test_rmse)


    # imp_mean = IterativeImputer(random_state=0)
    # imp_mean.fit(values_nan[train_index,:])

    # train_imp = imp_mean.transform(values_nan[train_index,:])
    # test_imp = imp_mean.transform(values_nan[test_index,:])

    # test_rmse = np.sqrt(np.sum((observed_values[test_index,:] - test_imp) ** 2 * (1 - S_test)) / np.sum(1 - S_test))
    # print("Mice Imputation Test RMSE: " , test_rmse)

# #---------------------------------------------------------------------------------------


    if os.getcwd().endswith('MNAR'):
        os.chdir("TabCSDI")


    train_dataset = train_loader
    test_dataset = test_loader


    train_obs = train_dataset["observed_data"]
    test_obs = test_dataset["observed_data"]

    
    # print(train_obs.shape)
    # print(test_obs.shape)

    train_mask = train_dataset["gt_mask"]
    test_mask = test_dataset["gt_mask"]

    # print(train_mask.shape)
    # print(test_mask.shape)

    train_nan = train_obs.copy()
    test_nan = test_obs.copy()
    train_nan[train_mask == 0] = np.nan
    test_nan[test_mask == 0] = np.nan

    # print(train_nan.shape)
    # print(test_nan.shape)

    S_train = np.array(~np.isnan(train_nan), dtype=float)
    S_test = np.array(~np.isnan(test_nan), dtype=float)



    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train_nan)

    


    train_imp = imp.transform(train_nan)
    test_imp = imp.transform(test_nan)

    train_rmse = np.sqrt(np.sum((train_obs - train_imp) ** 2 * (1 - S_train)) / np.sum(1 - S_train))
    test_rmse = np.sqrt(np.sum((test_obs - test_imp) ** 2 * (1 - S_test)) / np.sum(1 - S_test))

    #print("Train RMSE: ", train_rmse)
    print("Test RMSE: " , test_rmse)
    print("========")





