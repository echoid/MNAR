import numpy as np
import torch
from torch import nn
import sys
import util
import pickle


def load_data_index(dataname,missing_type,missing_name,seed = 1):
    processed_data_path_norm = (
            f"../datasets/{dataname}/{missing_type}-{missing_name}_seed-{seed}_max-min_norm.pk"
        )
    with open(processed_data_path_norm, "rb") as f:
            observed_values, observed_masks, gt_masks, eval_length = pickle.load(
                    f
            )

    N, D = observed_values.shape

    dl = D - 1
    
    indlist = np.arange(N)

    np.random.seed(seed + 1)
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


    return train_index, test_index, valid_index, observed_values, gt_masks

def get_split(train_index, test_index, valid_index, matrix, mask):

    # print("matrix",matrix)
    # print("mask",mask)

    Xtrain = matrix[train_index]
    Xtest = matrix[test_index]
    Xval = matrix[valid_index]

    Xtrain_mask = mask[train_index]
    Xtest_mask = mask[test_index]
    Xval_mask = mask[valid_index]


    Xtrainz = Xtrain.copy()
    Xtrainz[Xtrain_mask == 0] = 0


    Xvalz = Xval.copy()
    Xvalz[Xval_mask == 0] = 0


    Xtestz = Xtest.copy()
    Xtestz[Xtest_mask == 0] = 0

    return Xtrainz,Xtestz,Xvalz, Xtrain_mask, Xtest_mask, Xval_mask, Xtrain, Xtest, Xval


class DataLoader(nn.Module):
    """Data loader module for training MCFlow
    Args:
        mode (int): Determines if we are loading training or testing data
        seed (int): Used to determine the fold for cross validation experiments or reproducibility consideration if not
        path (str): Determines the dataset to load
        drp_percent (float): Determines the binomial success rate for observing a feature
    """
    MODE_TRAIN = 0
    MODE_TEST = 1

    def __init__(self, mode=MODE_TRAIN, seed=0, path='news', drp_percent=0.5,
                missing_type = "diffuse",
                missing_para = [0.25,0.25],
                missing_name = "diffuse"):
        
        self.path = path

        train_index, test_index, valid_index, self.matrix, self.mask = load_data_index(path,missing_type,missing_name,seed = 1)
        
        self.train, self.test,self.val, self.mask_tr,self.mask_te,self.mask_val,self.original_tr, self.original_te , self.original_val = get_split(train_index, 
                                                                                        test_index, 
                                                                                                         valid_index, 
                                                                                                         self.matrix, 
                                                                                                         self.mask)


        # if path == 'mnist':
        #     self.original_tr, self.original_te, img_shape = util.path_to_matrix(path)
        # else:
        #     # load data and get max and min
        #     matrix = util.path_to_matrix(path)
        #     self.matrix, self.maxs, self.mins = util.preprocess(matrix) #Preprocess according to the paper cited above
        #     #print("dataloader")
        #     print(self.matrix.shape)
        # if path == 'mnist':
        #     self.mask_tr, self.mask_te = util.create_img_dropout_masks(drp_percent, path, img_shape, len(self.original_tr), len(self.original_te))
        #     self.train, self.test = util.fill_img_missingness(self.original_tr, self.original_te, self.mask_tr, self.mask_te, img_shape, 0) #For now 0 represents nearest neighbor calc
        # else:
            
        #     np.random.shuffle(self.matrix)
        #     #print("shuffle",self.matrix.shape)
        #     np.random.seed(seed)
        #     self.mask = util.make_static_mask(drp_percent, seed, path, self.matrix) #check if the mask is there or not in this function
        #     #print("mask shape",self.mask.shape)

        #     # get train and test
        #     self.original_tr, self.original_te = util.create_k_fold(self.matrix, seed)
        #     self.unique_values = []
        #     self.mask_tr, self.mask_te = util.create_k_fold_mask(seed, self.mask)
        #     trans = np.transpose(self.matrix)
        #     for r_idx, rows in enumerate(trans):
        #         row = []
        #         for c_idx, element in enumerate(rows):
        #             if self.mask[c_idx][r_idx] == 0:
        #                 row.append(element)
        #         self.unique_values.append(np.asarray(row))
        #     # with 0
        #     #print(self.mask)
        #     self.train, self.test = util.fill_missingness(self.matrix, self.mask, self.unique_values, self.path, seed)
        self.mode = mode


    def reset_imputed_values(self, nn_model, nf_model, seed, args):

        random_mat = np.clip(util.inference_imputation_networks(nn_model, nf_model, self.train, args), 0, 1)
        self.train = (1-self.mask_tr) * self.original_tr + self.mask_tr * random_mat
        random_mat = np.clip(util.inference_imputation_networks(nn_model, nf_model, self.test, args), 0, 1)
        self.test = (1-self.mask_te) * self.original_te + self.mask_te * random_mat

        random_mat = np.clip(util.inference_imputation_networks(nn_model, nf_model, self.val, args), 0, 1)
        self.val = (1-self.mask_val) * self.original_val + self.mask_val * random_mat

    def reset_img_imputed_values(self, nn_model, nf_model, seed, args):

        util.inference_img_imputation_networks(nn_model, nf_model, self.train, self.mask_tr, self.original_tr, args)
        util.inference_img_imputation_networks(nn_model, nf_model, self.test, self.mask_te, self.original_te, args)
        util.inference_img_imputation_networks(nn_model, nf_model, self.val, self.mask_val, self.original_val, args)
    def __len__(self):
        if self.mode==0:
            return len(self.train)
        elif self.mode==1:
            return len(self.test)
        elif self.mode == 2:
            return len(self.val)
        else:
            print("Data loader mode error -- acceptable modes are 0,1,2")
            sys.exit()

    def __getitem__(self, idx):
        if self.mode==0:
            return torch.Tensor(self.train[idx]) , (torch.Tensor(self.original_tr[idx]), torch.Tensor(self.mask_tr[idx]))
        elif self.mode==1:
            return torch.Tensor(self.test[idx]) , (torch.Tensor(self.original_te[idx]), torch.Tensor(self.mask_te[idx]))
        elif self.mode==2:
            return torch.Tensor(self.val[idx]) , (torch.Tensor(self.original_val[idx]), torch.Tensor(self.mask_val[idx]))
        else:
            print("Data loader mode error -- acceptable modes are 0,1,2")
            sys.exit()
