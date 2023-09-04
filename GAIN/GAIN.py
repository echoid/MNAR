import torch
import os
import sys
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from GAIN_imputer_utility import xavier_init,MyDataset,preprocess,load_dataloader
from sklearn.impute import SimpleImputer
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
os.chdir("../")
from missing_process.block_rules import *


#dataset_file = "banknote"#'concrete_compression', "wine_quality_red","wine_quality_white"
  # 'Letter.csv' for Letter dataset an 'Spam.csv' for Spam dataset
# missing_type = "quantile"
missing_type = sys.argv[2]

dataset_file = sys.argv[1]


if missing_type == "quantile":

    missing_rule = ["Q1_complete","Q1_partial","Q2_complete","Q2_partial","Q3_complete","Q3_partial","Q4_complete","Q4_partial",
    "Q1_Q2_complete","Q1_Q2_partial","Q1_Q3_complete","Q1_Q3_partial","Q1_Q4_complete","Q1_Q4_partial","Q2_Q3_complete","Q2_Q3_partial",
    "Q2_Q4_complete","Q2_Q4_partial","Q3_Q4_complete","Q3_Q4_partial"]

elif missing_type == "diffuse":
    missing_rule = [0.5,0.75,0.25]

elif missing_type =="logistic":
    missing_rule = [0.5,0.75,0.25]


#%% System Parameters
batch_size = 64
epoch = 10000



class Simple_imputer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = nn.Linear(dim, dim)

    def forward(self, data, m):
        imputed_data = torch.sigmoid(self.linear(data))

        return imputed_data


class Imputation_model(nn.Module):
    def __init__(self, dim, hidden_dim1, hidden_dim2):
        super(Imputation_model, self).__init__()
    
        self.G_W1 = nn.Parameter(torch.tensor(xavier_init([dim * 2, hidden_dim1]), dtype=torch.float32), requires_grad=True)
        self.G_b1 = nn.Parameter(torch.zeros(hidden_dim1, dtype=torch.float32), requires_grad=True)
        #self.G_bn1 = nn.BatchNorm1d(hidden_dim1)

        self.G_W2 = nn.Parameter(torch.tensor(xavier_init([hidden_dim1, hidden_dim2]), dtype=torch.float32), requires_grad=True)
        self.G_b2 = nn.Parameter(torch.zeros(hidden_dim2, dtype=torch.float32), requires_grad=True)
        #self.G_bn2 = nn.BatchNorm1d(hidden_dim2)

        self.G_W3 = nn.Parameter(torch.tensor(xavier_init([hidden_dim2, dim]), dtype=torch.float32), requires_grad=True)
        self.G_b3 = nn.Parameter(torch.zeros(dim, dtype=torch.float32), requires_grad=True)


    def forward(self, data, mask):
        inputs = torch.cat(dim=1, tensors=[data, mask])  # Mask + Data Concatenate
        inputs = inputs.float()  
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1.float()) + self.G_b1.float())
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2.float()) + self.G_b2.float())
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3.float()) + self.G_b3.float())  # [0,1] normalized Output

        return G_prob
    

class Discriminator_model(nn.Module):
    def __init__(self, dim, hidden_dim1, hidden_dim2):
        super(Discriminator_model, self).__init__()
    
        self.D_W1 = nn.Parameter(torch.tensor(xavier_init([dim*2, hidden_dim1]), dtype=torch.float32), requires_grad=True)
        self.D_b1 = nn.Parameter(torch.zeros(hidden_dim1, dtype=torch.float32), requires_grad=True)


        self.D_W2 = nn.Parameter(torch.tensor(xavier_init([hidden_dim1, hidden_dim2]), dtype=torch.float32), requires_grad=True)
        self.D_b2 = nn.Parameter(torch.zeros(hidden_dim2, dtype=torch.float32), requires_grad=True)
        #self.G_bn2 = nn.BatchNorm1d(hidden_dim2)

        self.D_W3 = nn.Parameter(torch.tensor(xavier_init([hidden_dim2, dim]), dtype=torch.float32), requires_grad=True)
        self.D_b3 = nn.Parameter(torch.zeros(dim, dtype=torch.float32), requires_grad=True)


    def forward(self, data, h):
        inputs = torch.cat(dim=1, tensors=[data, h])  # Mask + Data Concatenate
        inputs = inputs.float()  
        D_h1 = F.relu(torch.matmul(inputs, self.D_W1.float()) + self.D_b1.float())
        D_h2 = F.relu(torch.matmul(D_h1, self.D_W2.float()) + self.D_b2.float())
        D_prob = torch.sigmoid(torch.matmul(D_h2, self.D_W3.float()) + self.D_b3.float())  # [0,1] normalized Output

        return D_prob





def get_dataset_loaders(trainX, train_Mask,train_input,testX, test_Mask, test_input,train_H, test_H):

    train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input,train_H), MyDataset(testX, test_Mask, test_input,test_H)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader , test_loader

def train_loss(truth, mask, data,imputer):
    generated = imputer(data, mask)
    
    RMSE = torch.sqrt(torch.sum((truth - generated) ** 2 * (1 - mask))/torch.sum(1 - mask))

    return  RMSE, generated


def G_loss(mask, data, imputer, h, discriminator):
    generated = imputer(data, mask)

    New_data = data * mask + generated * (1-mask)

    D_prob = discriminator(New_data,h)

    G_loss1 = -torch.mean((1- mask) * torch.log(D_prob + 1e-8))

    MSE_train_loss =torch.mean((mask * data - mask * generated)**2) / torch.mean(mask)

    G_loss = G_loss1 + 10 * MSE_train_loss

    return G_loss 

def D_loss(mask, data, imputer, h, discriminator):
    generated = imputer(data, mask)

    New_data = data * mask + generated * (1-mask)

    D_prob = discriminator(New_data,h)

    
    d_loss = -torch.mean(mask * torch.log(D_prob + 1e-8) + (1-mask) * torch.log(1. - D_prob + 1e-8))

    return  d_loss

def impute_with_prediction(original_data, mask, prediction):

    return mask * original_data + (1 - mask) * prediction




def test_g_loss(X, M, New_X,generator):

    G_sample = generator(New_X,M)

    #%% MSE Performance metric
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)

    imputed = impute_with_prediction(X,M,G_sample)
    return MSE_test_loss ,G_sample,imputed




def check_and_fill_nan(array, reference_array):
    num_rows, num_cols = array.shape

    for col_idx in range(num_cols):
        if np.all(np.isnan(array[:, col_idx])):
            array[0, col_idx] = np.nanmean(reference_array[:, col_idx])

    return array



def run(dataset_file,missing_rule):
    
   

    Imputer_RMSE = []
    baseline_RMSE = []
    rule_list = []
    
    for rule_name in missing_rule:
        trainX, testX, train_Mask, test_Mask, train_input, test_input, No, Dim, train_H,test_H = load_dataloader(dataset_file,missing_type, rule_name)

    
        train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input,train_H), MyDataset(testX, test_Mask, test_input,test_H)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        G = Imputation_model(Dim, Dim, Dim)
        D = Discriminator_model(Dim, Dim, Dim)
        #imputer = Simple_imputer(Dim)
        optimizer_G = torch.optim.Adam(params=G.parameters())
        optimizer_D = torch.optim.Adam(params=D.parameters())


        for it in tqdm(range(epoch)):
            G.train()
            D.train()
            total_g_loss = 0
            total_d_loss = 0
            batch_no = 0
            for truth_X, mask, data_X, h in train_loader:
                batch_no += 1

                optimizer_D.zero_grad()
                D_loss_curr = D_loss(mask=mask, data=data_X, imputer = G,h = h,discriminator = D )
                total_d_loss += D_loss_curr
                D_loss_curr.backward()
                optimizer_D.step()


                optimizer_G.zero_grad()

                G_loss_curr = G_loss(mask=mask, data=data_X, imputer = G,h = h,discriminator = D )
                total_g_loss += G_loss_curr
                G_loss_curr.backward()
                optimizer_G.step()

            print('Iter: {}'.format(it), end='\t')
            print('G_loss: {:.4} | D_loss: {:.4}'.format(np.sqrt(total_g_loss.item()/batch_no),np.sqrt(total_d_loss.item()/batch_no)))


        # Evaluation

        with torch.no_grad():
            G.eval()
            RMSE_total = []
            imputed_total = []
            for truth_X, mask, data_X, h in test_loader:
                RMSE, prediction,imputed =  test_g_loss(X=truth_X, M=mask, New_X=data_X,generator = G)
                RMSE_total.append(RMSE)
                imputed_total.append(imputed)


        RMSE_tensor = torch.tensor(RMSE_total)
        rmse_final = torch.mean(RMSE_tensor)

        Imputer_RMSE.append(round(rmse_final.item(),5))

        pd.DataFrame(torch.cat(imputed_total, 0).detach().numpy()).to_csv("results/gain/Imputation_{}_{}_{}.csv".format(dataset_file,missing_type,rule_name),index=False)
  

        print('Final Test RMSE: {:.4f}'.format(rmse_final.item()))

        ###################Baseline###############################
 
        train_nan = trainX.copy()
        test_nan = testX.copy()
        train_nan[train_Mask == 0] = np.nan
        test_nan[test_Mask == 0] = np.nan

        train_nan = check_and_fill_nan(train_nan,trainX)

        # ------------------------------------------------------------------------------
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(train_nan)
        train_imp = imp.transform(train_nan)
        test_imp = imp.transform(test_nan)
        #train_rmse = np.sqrt(np.sum((trainX - train_imp) ** 2 * (1 - train_mask)) / np.sum(1 - train_mask))
        test_rmse = np.sqrt(np.sum((testX - test_imp) ** 2 * (1 - test_Mask)) / np.sum(1 - test_Mask))


        print("Mean Imputer test_rmse:",test_rmse)


        baseline_RMSE.append(round(test_rmse,5))
        rule_list.append(rule_name)



    result = pd.DataFrame({"Missing_Rule":rule_list,"Imputer RMSE":Imputer_RMSE,"Baseline RMSE":baseline_RMSE})
    result.to_csv("results/gain/RMSE_{}_{}.csv".format(dataset_file,missing_type),index=False)


run(dataset_file,missing_rule)