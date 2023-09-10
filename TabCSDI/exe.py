import argparse
import torch
import datetime
import json
import yaml
import os
import sys
import pandas as pd

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)

from missing_process.block_rules import *

#dataname = "wine_quality_white"


from src.main_model_table import TabCSDI
from src.utils_table_new import train, evaluate
from dataset_loader import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--dataset",type = str, default ="wine_quality_red" )
parser.add_argument("--config", type=str, default="common") # test
parser.add_argument("--device", default="cpu", help="Device")
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

parser.add_argument("--missingtype", type=str, default="MCAR")
parser.add_argument("--missingpara", type=str, default="simple_rule")
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
rule_list = []
rmse_list =  []





for rule_name in missing_rule:
    rule = missing_rule[rule_name]
    print("Current Rule",rule )
    # Create folder
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/{}_fold".format(args.dataset) + str(args.nfold) + "_" + current_time + "/"
    

    os.makedirs(foldername, exist_ok=True)
    # with open(foldername + "config.json", "w") as f:
    #     json.dump(config, f, indent=4)


 

    # Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
    train_loader, valid_loader, test_loader = get_dataloader(
        dataname=args.dataset,
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_type = args.missingtype,
        missing_para = rule,
        missing_name = rule_name

    )
    


    if os.getcwd().endswith('MNAR'):
        os.chdir("TabCSDI")



    model = TabCSDI(config, args.device).to(args.device)



    if args.modelfolder == "":
        print("model train")
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    print("---------------Start testing---------------")
    rmse, samples, original,mask = evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

    pd.DataFrame(torch.cat(samples, 0).detach().numpy()).to_csv("../results/tabcsdi/Imputation_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)
    # pd.DataFrame(torch.cat(original, 0).detach().numpy()).to_csv("../results/tabcsdi/Origin_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)
    # pd.DataFrame(torch.cat(mask, 0).detach().numpy()).to_csv("../results/tabcsdi/Mask_{}_{}_{}.csv".format(args.dataset,args.missingtype,rule_name),index=False)

    rule_list.append(rule_name)
    rmse_list.append(rmse)


result = pd.DataFrame({"Missing_Rule":rule_list,"Imputer RMSE":rmse_list})
result.to_csv("../results/tabcsdi/RMSE_{}_{}.csv".format(args.dataset,args.missingpara),index=False)