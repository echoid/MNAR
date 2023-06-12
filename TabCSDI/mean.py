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
from sklearn.metrics import mean_squared_error
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)

from missing_process.block_rules import *

#dataname = "wine_quality_white"

from src.main_model_table import TabCSDI
from src.utils_table_new import train, evaluate
from mean_loader import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--dataset",type = str, default ="wine_quality_white" )
parser.add_argument("--config", type=str, default="test") # test
parser.add_argument("--device", default="cpu", help="Device")
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

parser.add_argument("--missingtype", type=str, default="MCAR")
parser.add_argument("--missingpara", type=str, default="missing_rate")
#parser.add_argument("--testmissingratio", type=float, default=0.1)


args = parser.parse_args()

args.config = "{}.yaml".format(args.config)

print(args.dataset, args.config, args.missingtype)

print("Before load config:",os.getcwd())
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path = "config/" + args.config

with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
#config["model"]["test_missing_ratio"] = args.testmissingratio

#print(json.dumps(config, indent=4))




missing_rule = load_json_file(args.missingpara + ".json")


for rule_name in missing_rule:
    rule = missing_rule[rule_name]
    print("Current Rule",rule )
    # Create folder
    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # foldername = "./save/{}_fold".format(args.dataset) + str(args.nfold) + "_" + current_time + "/"
    

    # os.makedirs(foldername, exist_ok=True)
    # with open(foldername + "config.json", "w") as f:
    #     json.dump(config, f, indent=4)


 

    # Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
    train_dataset, valid_dataset,test_dataset = get_dataloader(
        dataname=args.dataset,
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_type = args.missingtype,
        missing_para = rule,
        missing_name = rule_name

    )

    print(train_dataset)


    # Compute the mean of the train observed data along axis 0
    train_mean = np.mean(train_dataset["observed_data"], axis=0)


    # Perform train mean imputation on the test observed data using the mask
    test_imputed_data = test_dataset["observed_data"].copy()




    # Remove data from test_observed_data using the mask
    test_imputed_data = np.where(test_dataset["gt_mask"], 0, test_dataset["observed_data"])

    # Calculate the RMSE between the imputed data and the ground truth (test_observed_data)
    rmse = np.sqrt(mean_squared_error(test_dataset["observed_data"], test_imputed_data, squared=False))

    print("Mean RMSE:", rmse)







    # Convert the mask to contain NaN values
    test_gt_mask_nan = np.where(test_dataset["gt_mask"] == 0, np.nan, 1)

    # Create a missing data version of the observed data
    test_observed_data_missing = test_dataset["observed_data"] * test_gt_mask_nan


    # Convert the mask to contain NaN values
    train_gt_mask_nan = np.where(train_dataset["gt_mask"] == 0, np.nan, 1)

    # Create a missing data version of the observed data
    train_observed_data_missing = train_dataset["observed_data"] * train_gt_mask_nan



    # Create an instance of the IterativeImputer with a linear regression model
    imputer = IterativeImputer(estimator=LinearRegression())

    # Fit the imputer on the observed data
    imputer.fit(train_observed_data_missing)

    # Impute the missing values in the test_data
    imputed_data = imputer.transform(test_observed_data_missing)


    rmse = mean_squared_error(test_dataset["observed_data"], imputed_data, squared=False)

    print("MICE RMSE:", rmse)


    exit()

