#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.preprocessing import scale
import os
import pandas as pd

import numpy as np

import wget


import sys
sys.path.append("..")


DATASETS = ['iris', 'wine', 
            #'boston',
             'california',
             #  'parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel',
             # 'concrete_slump', \
            'wine_quality_red', 'wine_quality_white',"banknote"]



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
        
        print(data.shape)
        return data
    
    else:
    
        print("Please input valid dataset name: 'banknote', 'concrete','white','red'.")
        exit()



def dataset_loader(dataset):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes 
    datasets are located in './datasets'. If the called for dataset is not in 
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.
        
    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS , f"Dataset not supported: {dataset}"
    if not "datasets" in os.listdir():
        os.chdir("../")

    if dataset in DATASETS:
        if dataset == 'iris':
            data = load_iris()
        elif dataset == 'wine':
            data = load_wine()
        # elif dataset == 'boston':
        #     X = load_boston()['data']
        elif dataset == 'california':
            data = fetch_california_housing()
            try:
                os.mkdir('datasets/california')
            except:
                pass
        # elif dataset == 'parkinsons':
        #     data = fetch_parkinsons()
        elif dataset == 'climate_model_crashes':
            data = fetch_climate_model_crashes()
        elif dataset == 'concrete_compression':
            data = fetch_concrete_compression()
        elif dataset == 'yacht_hydrodynamics':
            data = fetch_yacht_hydrodynamics()
        elif dataset == 'airfoil_self_noise':
            data = fetch_airfoil_self_noise()
        elif dataset == 'connectionist_bench_sonar':
            data = fetch_connectionist_bench_sonar()
        elif dataset == 'ionosphere':
            data = fetch_ionosphere()
        elif dataset == 'qsar_biodegradation':
            data = fetch_qsar_biodegradation()
        elif dataset == 'seeds':
            data = fetch_seeds()
        elif dataset == 'glass':
            data = fetch_glass()
        elif dataset == 'ecoli':
            data = fetch_ecoli()
        elif dataset == 'yeast':
            data = fetch_yeast()
        elif dataset == 'libras':
            data = fetch_libras()
        elif dataset == 'planning_relax':
            data = fetch_planning_relax()
        elif dataset == 'blood_transfusion':
            data = fetch_blood_transfusion()
        elif dataset == 'breast_cancer_diagnostic':
            data = fetch_breast_cancer_diagnostic()
        elif dataset == 'connectionist_bench_vowel':
            data = fetch_connectionist_bench_vowel()
        # elif dataset == 'concrete_slump':
        #     data = fetch_concrete_slump()
        elif dataset == 'wine_quality_red':
            data = fetch_wine_quality_red()
        elif dataset == 'wine_quality_white':
            data = fetch_wine_quality_white()
        elif dataset == 'banknote':
            data = fetch_banknote()
        #print(dataset,data["data"].shape,len(np.unique(data["target"])))
        return data




# def fetch_parkinsons():
#     if not os.path.isdir('datasets/parkinsons'):
#         os.mkdir('datasets/parkinsons')
#         url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
#         wget.download(url, out='datasets/parkinsons/')

#     with open('datasets/parkinsons/parkinsons.data', 'rb') as f:
#         df = pd.read_csv(f, delimiter=',', header = 0)
#         Xy = {}
#         Xy['data'] = df.values[:, 1:].astype('float')
#         Xy['target'] =  df.values[:, 0]

#     return Xy


def fetch_climate_model_crashes():
    if not os.path.isdir('datasets/climate_model_crashes'):
        os.mkdir('datasets/climate_model_crashes')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat'
        wget.download(url, out='datasets/climate_model_crashes/')

    with open('datasets/climate_model_crashes/pop_failures.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = 0)
        Xy = {}
        # Ignore the two blocking factor
        Xy['data'] = df.values[:, 2:-1]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_banknote():
    if not os.path.isdir('datasets/banknote'):
        os.mkdir('datasets/banknote')
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        wget.download(url, out='datasets/banknote/')

    with open('datasets/banknote/data_banknote_authentication.txt', 'rb') as f:
        df = pd.read_csv(f, low_memory=False, sep=',')
        Xy = {}
        # Ignore the two blocking factor
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy


    

def fetch_concrete_compression():
    if not os.path.isdir('datasets/concrete_compression'):
        os.mkdir('datasets/concrete_compression')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out='datasets/concrete_compression/')

    with open('datasets/concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_yacht_hydrodynamics():
    if not os.path.isdir('datasets/yacht_hydrodynamics'):
        os.mkdir('datasets/yacht_hydrodynamics')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        wget.download(url, out='datasets/yacht_hydrodynamics/')

    with open('datasets/yacht_hydrodynamics/yacht_hydrodynamics.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_airfoil_self_noise():
    if not os.path.isdir('datasets/airfoil_self_noise'):
        os.mkdir('datasets/airfoil_self_noise')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        wget.download(url, out='datasets/airfoil_self_noise/')

    with open('datasets/airfoil_self_noise/airfoil_self_noise.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_connectionist_bench_sonar():
    if not os.path.isdir('datasets/connectionist_bench_sonar'):
        os.mkdir('datasets/connectionist_bench_sonar')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        wget.download(url, out='datasets/connectionist_bench_sonar/')

    with open('datasets/connectionist_bench_sonar/sonar.all-data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_ionosphere():
    if not os.path.isdir('datasets/ionosphere'):
        os.mkdir('datasets/ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out='datasets/ionosphere/')

    with open('datasets/ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_qsar_biodegradation():
    if not os.path.isdir('datasets/qsar_biodegradation'):
        os.mkdir('datasets/qsar_biodegradation')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out='datasets/qsar_biodegradation/')

    with open('datasets/qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_seeds():
    if not os.path.isdir('datasets/seeds'):
        os.mkdir('datasets/seeds')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
        wget.download(url, out='datasets/seeds/')

    with open('datasets/seeds/seeds_dataset.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_glass():
    if not os.path.isdir('datasets/glass'):
        os.mkdir('datasets/glass')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        wget.download(url, out='datasets/glass/')

    with open('datasets/glass/glass.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        # remove index
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_ecoli():
    if not os.path.isdir('datasets/ecoli'):
        os.mkdir('datasets/ecoli')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        wget.download(url, out='datasets/ecoli/')

    with open('datasets/ecoli/ecoli.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        # remove index
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_yeast():
    if not os.path.isdir('datasets/yeast'):
        os.mkdir('datasets/yeast')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        wget.download(url, out='datasets/yeast/')

    with open('datasets/yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        # remove index
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_libras():
    if not os.path.isdir('datasets/libras'):
        os.mkdir('datasets/libras')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        wget.download(url, out='datasets/libras/')

    with open('datasets/libras/movement_libras.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_planning_relax():
    if not os.path.isdir('datasets/planning_relax'):
        os.mkdir('datasets/planning_relax')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt'
        wget.download(url, out='datasets/planning_relax/')

    with open('datasets/planning_relax/plrx.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_blood_transfusion():
    if not os.path.isdir('datasets/blood_transfusion'):
        os.mkdir('datasets/blood_transfusion')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out='datasets/blood_transfusion/')

    with open('datasets/blood_transfusion/transfusion.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_breast_cancer_diagnostic():
    if not os.path.isdir('datasets/breast_cancer_diagnostic'):
        os.mkdir('datasets/breast_cancer_diagnostic')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out='datasets/breast_cancer_diagnostic/')

    with open('datasets/breast_cancer_diagnostic/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] =  df.values[:, 1]

    return Xy


def fetch_connectionist_bench_vowel():
    if not os.path.isdir('datasets/connectionist_bench_vowel'):
        os.mkdir('datasets/connectionist_bench_vowel')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out='datasets/connectionist_bench_vowel/')

    with open('datasets/connectionist_bench_vowel/vowel-context.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


# def fetch_concrete_slump():
#     if not os.path.isdir('datasets/concrete_slump'):
#         os.mkdir('datasets/concrete_slump')
#         url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
#         wget.download(url, out='datasets/concrete_slump/')

#     with open('datasets/concrete_slump/slump_test.data', 'rb') as f:
#         df = pd.read_csv(f, delimiter=',')
#         Xy = {}
#         Xy['data'] = df.values[:, 1:-3].astype('float')
#         Xy['target'] =  df.values[:, -3:]

#     return Xy


def fetch_wine_quality_red():
    if not os.path.isdir('datasets/wine_quality_red'):
        os.mkdir('datasets/wine_quality_red')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        wget.download(url, out='datasets/wine_quality_red/')

    with open('datasets/wine_quality_red/winequality-red.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

# Dpne!
def fetch_wine_quality_white():
    if not os.path.isdir('datasets/wine_quality_white'):
        os.mkdir('datasets/wine_quality_white')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out='datasets/wine_quality_white/data.csv')
    with open('datasets/wine_quality_white/data.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy



def split(n, nfold=5, seed = 1):

    indlist = np.arange((n))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / nfold
    start = (int)((nfold - 1) * n * tmp_ratio)

    end = (int)(nfold * n * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)

    # Modify here to change train,valid ratio
    num_train = (int)(len(remain_index) * 0.8)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    return train_index,valid_index,test_index

def normal_split(data):
    # ---- load data

    y = data["target"]
    # ---- drop the classification attribute
    X = scale(data["data"]).astype(np.float32)

    # ----

    N, D = X.shape

    dl = D - 1

    # # ---- standardize data
    # X = X - np.mean(X, axis=0)
    # X = X / np.std(X, axis=0)


    # ---- random permutation
    p = np.random.permutation(N)
    X = X[p, :]
    y = y[p]

    # Xtrain = data.copy()
    # Xval_org = data.copy()
    # Ytrain = label.copy()
    # Yval_org = label.copy()

    # Xtrain = X.copy()[:int(N*0.8),:]
    # Xval_org = X.copy()[int(N*0.8):int(N*0.9),:]
    # Xtest = X.copy()[int(N*0.9):,:]

    # Ytrain = y.copy()[:int(N*0.8)]
    # Yval_org = y.copy()[int(N*0.8):int(N*0.9)]
    # Ytest = y.copy()[int(N*0.9):]


    train_X = X.copy()[:int(N*0.8),:]
    #dex_X = X.copy()[int(N*0.8):int(N*0.9),:]
    test_X = X.copy()[int(N*0.8):,:]

    train_y = y.copy()[:int(N*0.8)]
    #dex_y = y.copy()[int(N*0.8):int(N*0.9)]
    test_y = y.copy()[int(N*0.8):]

    data = {}
    data["train_X"] = train_X
    data["test_X"] = test_X
    data["train_y"] = train_y
    data["test_y"] = test_y
    # data["dex_X"] = dex_X
    # data["dex_y"] = dex_y

    return data,X #, Xval_org, Yval_org, 



# dataname = ["connectionist_bench_sonar","qsar_biodegradation","wine_quality_white","yeast",
#             "california","concrete_compression","yacht_hydrodynamics","airfoil_self_noise"]
# #print(len(data_loaders.DATASETS))

# for name in dataname:
#     data = dataset_loader(name) 
#     np.save("datasets/{}/origin_X.npy".format(name), data["data"])
#     np.save("datasets/{}/origin_y.npy".format(name), data["target"])


def save(name,data,data_split):
    np.save("datasets/{}/origin_X.npy".format(name), data["data"])
    np.save("datasets/{}/origin_y.npy".format(name), data["target"])
    # np.save("datasets/{}/train_X.npy".format(name), data_split["train_X"])
    # np.save("datasets/{}/train_y.npy".format(name), data_split["train_y"])
    # # np.save("datasets/{}/val_X.npy".format(name), data_split["dex_X"])
    # # np.save("datasets/{}/val_y.npy".format(name), data_split["dex_y"])
    # np.save("datasets/{}/test_X.npy".format(name), data_split["test_X"])
    # np.save("datasets/{}/test_y.npy".format(name), data_split["test_y"])