import numpy as np
import os
import pickle

def load_data_index(args,rule_name):


    processed_data_path_norm = (
            f"../datasets/{args.dataset}/{args.missingtype}-{rule_name}_seed-{args.seed}_max-min_norm.pk"
        )

    with open(processed_data_path_norm, "rb") as f:
            observed_values, observed_masks, gt_masks, eval_length = pickle.load(
                    f
            )

    N, D = observed_values.shape

    dl = D - 1
    
    indlist = np.arange(N)

    np.random.seed(args.seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / 5
    start = (int)((5 - 1) * N * tmp_ratio)
    
    end = (int)(5 * N * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)

    # Modify here to change train,valid ratio
    num_train = (int)(len(remain_index) * 0.9)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]


    return observed_values.shape,dl, train_index, test_index, valid_index,observed_values, gt_masks