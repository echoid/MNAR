from missing_process.block_rules import *
import argparse
from data_loaders import *
import numpy as np



parser = argparse.ArgumentParser(description="Baseline")
parser.add_argument("--dataset",type = str, default ="wine_quality_white" )

parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--missingtype", type=str, default="MCAR")
parser.add_argument("--missingpara", type=str, default="missing_rate")



args = parser.parse_args()


missing_rule = load_json_file(args.missingpara + ".json")
np.random.seed(args.seed)


data = dataset_loader('wine_quality_white')



for rule_name in missing_rule:
    rule = missing_rule[rule_name]
    print("Current Rule:",rule )