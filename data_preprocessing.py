import numpy as np
import pandas as pd
import os
import data_loaders
import json
from tqdm import  tqdm

from missing_process.missing_method import *

missing_method = "logistic"
missing_dim = 0.5
missing_rate = 0.5

dataname = [
            #"connectionist_bench_sonar","qsar_biodegradation",
            "wine_quality_white",
            #"yeast","california","concrete_compression","yacht_hydrodynamics","airfoil_self_noise"
            ]
#print(len(data_loaders.DATASETS))

DATASAVE = False


for name in dataname:
    print(name)
    data = data_loaders.dataset_loader(name) 

    data_split,X = data_loaders.normal_split(data)

    if DATASAVE:
        data_loaders.save(name,data,data_split)


if missing_method == "quantile":
    for json_name in os.listdir("missing_process/block_rules"):
        with open('missing_process/block_rules/{}'.format(json_name)) as f:
            block_rules = json.load(f)

        for rule_name in block_rules:
            block_rule = block_rules[rule_name]
            Xnan, Xz = missing_by_range(data_split["train_X"], block_rule, missing_dim)

            #print("-----{}-----".format(rule_name))
            



elif missing_method == "logistic":
    mask = MNAR_mask_logistic(data_split["train_X"],missing_rate)
    Xnan = data_split["train_X"][mask] = np.nan
    rule_name = "logistic"

elif missing_method == "self_mask":
    mask = MNAR_self_mask_logistic(data_split["train_X"], missing_rate)
    Xnan = data_split["train_X"][mask] = np.nan
    rule_name = "self_mask"

if DATASAVE:
    if not os.path.isdir('datasets/{}/{}'.format(name,rule_name)):
        os.mkdir('datasets/{}/{}'.format(name,rule_name))
        np.save("datasets/{}/{}/Xnan.npy".format(name,rule_name), Xnan)