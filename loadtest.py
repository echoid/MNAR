import numpy as np
import pandas as pd

import data_loaders


print(len(data_loaders.DATASETS))

for dataname in data_loaders.DATASETS:
    data_loaders.dataset_loader(dataname)