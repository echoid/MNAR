import numpy as np
import pandas as pd
#from data_loaders import *

import missing_process.missing_method as missing_method
#from missing_process.block_rules import double_quantile_1,double_quantile_2,single_quantile
import missing_process.all_rules as ar
import os

from missing_process.block_rules import *


print("import missing process")

#print(double_quantile_1)
data = dataset_loader("wine_quality_red")
# for rule in double_quantile_1:

# #from missing_process.missing_method import * 
#     print(rule)
#     missing_method.missing_by_range(data["data"],double_quantile_1[rule])



def MCAR(observed_values, missing_ratio, masks):
    for col in range(observed_values.shape[1]):  # col #

        obs_indices = np.where(observed_values[:, col])[0]
        miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False

    return masks










# observed_values = data["data"].astype("float32")

# observed_masks = ~np.isnan(observed_values)
# masks = observed_masks.copy()

# "Need input origin dataset and parameters"
# masks = MCAR(observed_values,0.2,masks)


