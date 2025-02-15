import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import torch
import pickle
import os

# https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values
#import wget
#wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')

# from utils_generation import *
# import torch

def introduce_mising(X):
    N, D = X.shape
    Xnan = X.copy()

    # using mean as indicator
    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz


# def introduce_mising_middle(X):
#     N, D = X.shape
#     Xnan = X.copy()

#     # using mean as indicator
#     # ---- MNAR in D/2 dimensions

#     q3 = np.quantile(Xnan[:, :D],0.75, axis=0)
#     q1 = np.quantile(Xnan[:, :D],0.25, axis=0)
#     ix_q3 = Xnan[:, :D] > q3
#     ix_q1 = Xnan[:, :D] < q1
#     #ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
#     Xnan[:, :D][ix_q3] = np.nan
#     Xnan[:, :D][ix_q1] = np.nan

#     Xz = Xnan.copy()
#     Xz[np.isnan(Xnan)] = 0

#     return Xnan, Xz


def OT_missing(X, p, missing_mecha, p_obs=0.2, q=0.25):
    N, D = X.shape
    Xnan = X.copy()


    X_miss_mcar = produce_NA(X, p_miss=p, mecha= missing_mecha, p_obs = p_obs, q=q)

    Xnan = X_miss_mcar['X_incomp'].numpy()
    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz



def read_data(url):
    
    url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

    url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

    url3 = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    url4= "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    if url == "banknote":
        data = np.array(pd.read_csv(url1, low_memory=False, sep=','))
        return data

    elif url == "concrete":
        data = np.array(pd.read_excel(url2))
        return data

    elif url == "white":
        data = np.array(pd.read_csv(url3, low_memory=False, sep=';'))
        return data

    elif url == "red":
        data = np.array(pd.read_csv(url4, low_memory=False, sep=';'))
        return data
    
    else:
    
        print("Please input valid dataset name: 'banknote', 'concrete','white','red'.")
        exit()

    



# Function produce_NA for generating missing values ------------------------------------------------------

def produce_NA(X, p_miss, mecha="OTselfmask", p_obs=0.2, q=0.25):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        print("MAR")
        mask = MAR_mask(X, p_miss, p_obs).double()

    elif mecha == "OTlogistic":
        print("OTlogistic")
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()

    elif mecha == "OTquantile":
        print("OTquantile")
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()

    elif mecha == "OTselfmask":
        print("OTselfmask")
        mask = MNAR_self_mask_logistic(X, p_miss).double()

    else:
        print("MCAR")
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}



def preprocessing(url):
    # ---- load data

    data = read_data(url)
    np.random.shuffle(data)
    label = data.copy()[:,-1]
    # ---- drop the classification attribute
    data = data[:, :-1]

    # ----

    N, D = data.shape

    dl = D - 1

    # ---- standardize data
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)

    # ---- random permutation
    p = np.random.permutation(N)
    data = data[p, :]
    label = label[p]

    # Xtrain = data.copy()
    # Xval_org = data.copy()
    # Ytrain = label.copy()
    # Yval_org = label.copy()

    Xtrain = data.copy()[:int(N*0.8),:]
    Xval_org = data.copy()[int(N*0.8):int(N*0.9),:]
    Xtest = data.copy()[int(N*0.9):,:]

    Ytrain = label.copy()[:int(N*0.8)]
    Yval_org = label.copy()[int(N*0.8):int(N*0.9)]
    Ytest = label.copy()[int(N*0.9):]

    return data.shape, Xtrain, Xval_org, Xtest, Ytrain, Yval_org, Ytest,dl


def load_data_index(args,rule_name):
    print(os.getcwd())

    processed_data_path_norm = (
            f"datasets/{args.dataset}/{args.missingtype}-{rule_name}_seed-{args.seed}_max-min_norm.pk"
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


def compare_distributions(complete_data, missing_data,title,density = True
                          ,bins=20
                          ):
    """
    Plots histograms and kernel density estimates for each dimension of a complete dataset and a dataset with missing data.
    
    Parameters:
        complete_data (np.ndarray): A complete dataset with an arbitrary number of dimensions.
        missing_data (np.ndarray): A dataset with missing data. Must have the same shape as `complete_data`.
        bins (int): The number of bins to use for the histograms.
    
    Returns:
        None
    """
    num_dims = complete_data.shape[1]
    fig, axs = plt.subplots(nrows=num_dims, ncols=1, figsize=(10, 2*num_dims))
    plt.suptitle(title)

    for i in range(num_dims):
        axs[i].hist(complete_data[:, i], alpha=0.5, 
                    bins=bins, 
                    density=density, label='Complete Data')
        axs[i].hist(missing_data[:, i], alpha=0.5, 
                    bins=bins, 
                    density=density, label='Missing Data')
        sns.kdeplot(complete_data[:, i], ax=axs[i], color='blue', label='Complete Data KDE')
        sns.kdeplot(missing_data[:, i], ax=axs[i], color='orange', label='Missing Data KDE')
        #axs[i].set_xlabel(f'Dimension {i+1}')
        if not density:
            axs[i].set_ylabel('Frequency')
        else:
            axs[i].set_ylabel('Density')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("plots/{}.png".format(title))
    plt.show()



def random_missing(array, fractions_to_change):
    """
    Randomly changes a fraction of the True values in each column of a boolean array to False.

    Args:
        array (numpy.ndarray): The input boolean array.
        fractions_to_change (list or numpy.ndarray): The fractions of True values to change to False in each column.

    Returns:
        result (numpy.ndarray): The boolean array with the specified fractions of True values changed to False in each column.
    """

    result = array.copy()
    for col in range(result.shape[1]):
        col_array = result[:, col]
        # if need to change to multiple conversion rate
        #n_to_change = int(np.sum(col_array) * fractions_to_change[col])
        n_to_change = int(np.sum(col_array) * fractions_to_change)
        ix_to_change = np.random.choice(np.flatnonzero(col_array), size=n_to_change, replace=False)

        col_array[ix_to_change] = False

    return result

def generate_middle(lower,upper,partial_missing,dataset,missing_dim):
    if lower == 0:
        lower_quantile = np.min(dataset[:, :missing_dim], axis=0)
    else:
        lower_quantile = np.quantile(dataset[:, :missing_dim],lower, axis=0)
    if upper == 1:
        upper_quantile = np.max(dataset[:, :missing_dim], axis=0)
    else:
        upper_quantile = np.quantile(dataset[:, :missing_dim],upper, axis=0)



    ix_larger_than = dataset[:, :missing_dim] >= lower_quantile
    ix_smaller_than = dataset[:, :missing_dim] <= upper_quantile
    combined_ix = np.equal(ix_larger_than, ix_smaller_than)
    combined_ix = random_missing(combined_ix,partial_missing)
    
    return combined_ix

def missing_by_range(X,multiple_block,missing_dim=1):
    """    
    Missing_quantile: value is larger than quantile will be missing
    Missing_dim: how many columns have missing data
    Partial_missing: if partially or completely missing (default=0, partial_missing rate = 0), 
                    if larger means left more data
    Missing_type: middle, outside, multiple
    """

    N, D = X.shape
    Xnan = X.copy()

    # ---- Missing Dimention
    missing_dim = int(missing_dim * D)
    
    ix_list = []
    for key in multiple_block.keys():
        info = multiple_block[key]
        combined_ix = generate_middle(info["lower"],info["upper"],info["partial_missing"], X, missing_dim)
        ix_list.append(combined_ix)
    combined_ix = np.logical_or.reduce(ix_list)
    

    Xnan[:, :missing_dim][combined_ix] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz