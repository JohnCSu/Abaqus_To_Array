import copy
from scipy.io import savemat
import pickle
import numpy as np
import torch
from helper_functions import *

from torch.utils.data import DataLoader,Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn

import glob
from pathlib import Path
import random

def sort_data(dir,shuffle = False,max_points  = None):
    '''
    Extracts the data found in each .mat file in the specified dir. We need the parameters as well as the input  vars from the key 'data'. The first part
    are the input variables and the second part are the output vars which forms the target data (y) that our Network needs to match.

    Output is
        - train_set. A tuple of input training data (i.e coordinates and parameters) and output data (e.g. U or stress) that we want the network to learn from
        - input vars: List of input variables
        - output vars: list of output variables
    '''
    ps = glob.glob(dir)
    print(len(ps))
    train_points = len(ps)
    input_data = []
    output_data = []
    if shuffle:
        random.shuffle(ps)
    if max_points is not None:
        ps = ps[:max_points]
    for p in ps:
        
        file = Path(p)
        params = file.stem.split('_')[1:]
        print(file.name)
        param_dict = {}
        for i in range(0,6,2):
            key = params[i]
            val = float(params[i+1].replace('-','.'))
            param_dict[key] = val

        mat_file = loaddata(file)

        input_data.append(mat_file['data'][:,1:len(mat_file['input vars'])])
        output_data.append(mat_file['data'][:,len(mat_file['input vars']):])


    train_set = (input_data[:train_points],output_data[:train_points])
    train_set = [torch.tensor(np.concatenate(s,axis=0,dtype= np.float32)) for s in train_set]
     
    return train_set,mat_file['input vars'][1:],mat_file['output vars'][1:]


def normalize(x:torch.Tensor,x1=None,x2=None,dim =0,mode = 'minmax'):
    if mode == 'minmax':
        if x1 is None:
            x1 = x.min(dim)[0]
        if x2 is None:
            x2 = x.max(dim)[0]
        return (x - x1)/(x2-x1),x1,x2

    elif mode == 'standard':
        if x1 is None:
            x1 = x.mean(dim)
        if x2 is None:
            x2 = x.std(dim)

        mean = x1
        var = x2
        return (x - mean)/var,mean,var 

    raise KeyError()

def unnormalize(x,x1,x2,mode ='minmax'):
    if mode == 'minmax':
        return x*(x2-x1) + x1
    elif mode =='standard':
        mean = x1
        var = x2
        return (x*var + mean) 

class FE_dataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index) -> tuple[torch.Tensor,torch.Tensor]:
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

    def cuda(self):
        self.x = self.x.cuda()
        self.y = self.y.cuda()

class MLP(nn.Module):
    '''
    Simple Multi Layer Perceptron Network. Good as a baseline.

    in_features: size of model input
    out_features: size of model output
    hidden_features: number of features in each hidden layer
    num_hidden_layers: number of hidden layers of network
    activation: activation function of model. type string will use the correspong torch function or you can pass your own activation function.

    '''
    def __init__(self,in_features : int,out_features: int,hidden_features: int,num_hidden_layers: int) -> None:
        super().__init__()

       
        linear = nn.Linear        
        self.linear_in = linear(in_features,hidden_features)
        self.linear_out = linear(hidden_features,out_features)
        
        self.activation = torch.relu
        self.layers = nn.ModuleList([linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)  ])
        self.bnorms = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(num_hidden_layers)])
        # self.bnorms = nn.ModuleList([nn.Dropout(p=0.2) for _ in range(num_hidden_layers)])
         
    def forward(self,x):
        x = self.activation(self.linear_in(x))
        for layer,bnorm in zip(self.layers,self.bnorms):
            x = self.activation(bnorm(layer(x)))
            # x = bnorm(x)
            
        return self.linear_out(x)

import copy
from scipy.io import savemat


def save_net_results(data_dict,net,device = 'cpu'):
    nn_data = copy.deepcopy(data_dict)
    data = data_dict['data']
    
    with torch.no_grad():
        node_labels = data[:,0][:,np.newaxis]
        inputs = data[:,1:len(data_dict['input vars'])]
        xyzt = data[:,1:len(data_dict['input vars'])]
        xyzt = torch.tensor(xyzt,dtype=torch.float32,device=device)[:,[0,1,3,4,5,6]]
        xyzt,_,_ = normalize(xyzt,x1,x2)

        net = net.to(device = device)
        output = unnormalize(net(xyzt),y1,y2)
        # output = net(xyzt)
        output = np.array(output)

    nn_data['data'] = np.concatenate([node_labels,inputs,output],axis = -1)

    return nn_data