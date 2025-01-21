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

from Network_Training_functions import sort_data,normalize,unnormalize,FE_dataset,MLP,save_net_results
from scipy.io import savemat

'''
Note That this script does not contain ANY ABAQUS API COMMANDs. As such it can be run on native python processes (or on abaqus python with pytorch and appropriate libraries installed)

'''

if __name__ == '__main__':
    # We seperated the 2 dataset already
    # Extract Results from training and Test
    train_set,input_vars,output_vars = sort_data('ML_Folder/*.mat',shuffle = False)
    valid_set,_,_ = sort_data('ML_Folder/Test/*.mat',shuffle = False)
    # For this example we eliminate the velocity paramter (parameter 2)
    train_set[0] =  train_set[0][:,[0,1,3,4,5,6]]
    valid_set[0] =  valid_set[0][:,[0,1,3,4,5,6]]

    #Scale Data to [0,1]
    out_weights = torch.tensor([0.4,3e-2,3e-2])
    in_weights = torch.abs(train_set[0]).max(0)[0]

    train_set[0],x1,x2 = normalize(train_set[0])
    train_set[1],y1,y2 =  normalize(train_set[1])
    valid_set[0],_,_ =  normalize(valid_set[0],x1,x2)

    train_dataset = FE_dataset(*train_set)
    valid_dataset = FE_dataset(*valid_set)
    train_dataset.cuda()
    valid_dataset.cuda()


    train_DL = DataLoader(train_dataset,batch_size=256,shuffle= True,num_workers = 0)
    valid_DL = DataLoader(valid_dataset,batch_size=len(valid_dataset),shuffle= False,num_workers = 0)

    net = MLP(6,3,256,3)

    # net = Modified_Fourier_Net(7,1,128,4,10,activation='relu')
    net = net.cuda()
    optimizer = Adam(net.parameters(),lr=1e-3)
    lr_scheduler = ExponentialLR(optimizer,gamma = 0.95)


    y1,y2 = y1.cuda(),y2.cuda()
    out_weights = out_weights.cuda()
    in_weights = in_weights.cuda()
    optimizer.zero_grad()


    '''
    Training Loop
    '''

    for i in range(100):
        net.train()
        train_loss = 0
        train_norm = 0
        for x,y in train_DL:
            

            # x,y = xx[0].cuda(),xx[1].cuda()
            out = net(x).squeeze()
            # out = scale_inputs(x,net,in_weights,out_weights)

            train_error =(out-y).pow(2)
            loss = train_error.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                train_loss += (train_error.sum())
                train_norm += (y.pow(2).sum())
            
        if i % 3 == 0:
            net.eval() 
            train_l2 = torch.sqrt(train_loss/train_norm)
            train_loss = train_loss/len(train_dataset)
            print(len(train_dataset))
            error = 0
            y_norm = 0
            loss = 0
            with torch.no_grad():
                for x,y in (valid_DL):

                    out = unnormalize(net(x),y1,y2).squeeze()
                    out = out
                    err = (out-y).pow(2)
                    error += (err.sum())
                    y_norm += (y.pow(2).sum())
                    
                l2_error = torch.sqrt(error/y_norm)
                loss = error/len(valid_dataset)
                print(f'Epoch {i} LR {float(lr_scheduler.get_last_lr()[0]):.3E} Valid l2_error = {float(l2_error):.3E}, Train L2 Error: {float(train_l2):.3E} Train Loss {float(train_loss):.3E}')
                node_label = err.argmax(0)[0]
                print(node_label)
                print(f'Worst Elements {node_label}, Net Value: {out[node_label]}, True Value {y[node_label]} at Point {[float(xx) for xx in x[node_label]]}')

        if (i+1) % 100 == 0:
            lr_scheduler.step()



y1,y2 = y1.cpu(),y2.cpu()

ps = glob.glob('ML_Folder/Test/*.mat')
os.makedirs('Test',exist_ok=True)
for p in ps:
    file = Path(p)
    print(file.name)    
    data = loaddata(file)

    to_export = save_net_results(data,net)
    to_export_compact = convert_tabular_to_compact(to_export)
    savemat(f'Test/{file.stem}.mat',to_export)