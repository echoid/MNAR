import sys
sys.path.append("..")
import pickle
import yaml
import os
import re
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from data_loaders import *
import missing_process.missing_method as missing_method



def MCAR(observed_values, missing_ratio, masks):
    for col in range(observed_values.shape[1]):  # col #

        obs_indices = np.where(observed_values[:, col])[0]
        miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False

    return masks

def process_func(dataname,path: str, aug_rate=1,missing_type = "MCAR",
                  missing_para = ""):
 
    data = dataset_loader(dataname)
    # print(data)
    # data.replace("?", np.nan, inplace=True)
    # Don't apply data argument (use n*dataset)
    # data_aug = pd.concat([data] * aug_rate)

    observed_values = data["data"].astype("float32")

    observed_masks = ~np.isnan(observed_values)
    masks = observed_masks.copy()

    "Need input origin dataset and parameters"
    if missing_type == "MCAR":
        masks = MCAR(observed_values,missing_para,masks)

    elif missing_type == "quantile":
        Xnan, Xz = missing_method.missing_by_range(observed_values, missing_para)
        masks = np.array(~np.isnan(Xnan), dtype=np.float)
        

    elif missing_type == "logistic":
        masks = missing_method.MNAR_mask_logistic(observed_values, missing_para)
 

    elif missing_type == "self_mask":
        masks = missing_method.MNAR_self_mask_logistic(observed_values, missing_para)


    # gt_mask: 0 for missing elements and manully maksed elements
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype(int)
    gt_masks = gt_masks.astype(int)

    return observed_values, observed_masks, gt_masks, data["data"].shape[1]


class tabular_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(
        self, dataname, use_index_list=None, 
        aug_rate=1, seed=0,
        missing_type = "MCAR", missing_para = "",missing_name = "MCAR"
        ):
        #self.eval_length = eval_length
        np.random.seed(seed)
        
        dataset_path = f"datasets/{dataname}/data.csv"
        processed_data_path = (
            f"datasets/{dataname}/{missing_type}-{missing_name}_seed-{seed}.pk"
        )
        processed_data_path_norm = (
            f"datasets/{dataname}/{missing_type}-{missing_name}_seed-{seed}_max-min_norm.pk"
        )
        # If no dataset created
        if not os.path.isfile(processed_data_path):
            self.observed_values, self.observed_masks, self.gt_masks, self.eval_length = process_func(
                dataname, dataset_path, aug_rate=aug_rate,
                missing_type = missing_type, missing_para = missing_para
            )
            print("self.eval_length",self.eval_length)
            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks, self.eval_length], f
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, self.eval_length = pickle.load(
                    f
                )

                            # Calculate the percentage of zeros
            # zero_percentage = (self.gt_masks == 0).mean() * 100

            # print(f"0的占比: {zero_percentage}%")

            # # Check for columns with all zeros
            # all_zero_columns = np.all(self.gt_masks == 1, axis=0)

            # if any(all_zero_columns):
            #     print("存在一列全部为1的情况")
            #     # If you want to print the indices of the all-zero columns:
            #     zero_column_indices = np.where(all_zero_columns)[0]
            #     print("全部为1的列的索引:", zero_column_indices)
            # else:
            #     print("没有一列全部为1的情况")
                    
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(dataname, seed=1, nfold=5, batch_size=16,
                   missing_type = "MCAR", missing_para = "", missing_name = "MCAR"):

    dataset = tabular_dataset(dataname = dataname,seed=seed,
                              missing_type = missing_type, missing_para = missing_para,
                                missing_name = missing_name)
    print(f"Dataset size:{len(dataset)} entries")
    
    
    indlist = np.arange(len(dataset))

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / nfold
    start = (int)((nfold - 1) * len(dataset) * tmp_ratio)
    
    end = (int)(nfold * len(dataset) * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)

    # Modify here to change train,valid ratio
    num_train = (int)(len(remain_index) * 0.9)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]



    # Here we perform max-min normalization.
    processed_data_path_norm = (
        f"datasets/{dataname}/{missing_type}-{missing_name}_seed-{seed}_max-min_norm.pk"
    )
    if not os.path.isfile(processed_data_path_norm):
        print(
            "--------------Dataset has not been normalized yet. Perform data normalization and store the mean value of each column.--------------"
        )
        # data transformation after train-test split.
        col_num = dataset.observed_values.shape[1]
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        for k in range(col_num):
            # Using observed_mask to avoid counting missing values.
            obs_ind = dataset.observed_masks[train_index, k].astype(bool)
            temp = dataset.observed_values[train_index, k]
            max_arr[k] = max(temp[obs_ind])
            min_arr[k] = min(temp[obs_ind])
        print(f"--------------Max-value for each column {max_arr}--------------")
        print(f"--------------Min-value for each column {min_arr}--------------")

        dataset.observed_values = (
            (dataset.observed_values - 0 + 1) / (max_arr - 0 + 1)
        ) * dataset.observed_masks

        # with open(processed_data_path_norm, "wb") as f:
        #     pickle.dump(
        #         [dataset.observed_values, dataset.observed_masks, dataset.gt_masks, dataset.eval_length], f
        #     )

    # Create datasets and corresponding data loaders objects.
    train_dataset = tabular_dataset(dataname = dataname,
        use_index_list=train_index, seed=seed,
        missing_type = missing_type, missing_para = missing_para, missing_name = missing_name
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = tabular_dataset(dataname = dataname,
        use_index_list=valid_index, seed=seed,
        missing_type = missing_type, missing_para = missing_para, missing_name = missing_name
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = tabular_dataset(dataname = dataname,
        use_index_list=test_index, seed=seed,
        missing_type = missing_type, missing_para = missing_para, missing_name = missing_name
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader
