import os
import uuid
import shutil
import json
# from botocore.client import Config
import ibm_boto3
import copy
from datetime import datetime
from skillsnetwork import cvstudio 

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.pyplot import imshow
from tqdm import tqdm
from ipywidgets import IntProgress
import time 

import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import lr_scheduler
import torch.nn as nn
torch.manual_seed(0)
import ssl 


from torchvision import transforms
import torchvision.models as models

def plot_stuff(COST,ACC):    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color = color)
    ax1.set_xlabel('Iteration', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color = color)
    ax2.tick_params(axis = 'y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()

def imshow_(inp, title=None):
    """Imshow for Tensor."""
    inp = inp .permute(1, 2, 0).numpy() 
    print(inp.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  
    plt.show()

def result(model,x,y):
    #x,y=sample
    z=model(x.unsqueeze_(0))
    _,yhat=torch.max(z.data, 1)
    
    if yhat.item()!=y:
        text="predicted: {} actual: {}".format(str(yhat.item()),y)
        print(text)

def train_model(model, train_loader,validation_loader, criterion, optimizer, val_set, device ,scheduler,n_epochs,print_=True):
    loss_list = []
    accuracy_list = []
    correct = 0
    #global:val_set
    n_test = len(val_set)
    accuracy_best=0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Loop through epochs
        # Loop through the data in loader
    print("The first epoch should take several minutes")
    for epoch in tqdm(range(n_epochs)):
        
        loss_sublist = []
        # Loop through the data in loader

        for x, y in train_loader:
            x, y=x.to(device), y.to(device)
            model.train() 

            z = model(x)
            loss = criterion(z, y)
            loss_sublist.append(loss.data.item())
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
        print("epoch {} done".format(epoch) )

        scheduler.step()    
        loss_list.append(np.mean(loss_sublist))
        correct = 0


        for x_test, y_test in validation_loader:
            x_test, y_test=x_test.to(device), y_test.to(device)
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / n_test
        accuracy_list.append(accuracy)
        if accuracy>accuracy_best:
            accuracy_best=accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        
        
        if print_:
            print('learning rate',optimizer.param_groups[0]['lr'])
            print("The validaion  Cost for each epoch " + str(epoch + 1) + ": " + str(np.mean(loss_sublist)))
            print("The validation accuracy for epoch " + str(epoch + 1) + ": " + str(accuracy)) 
    model.load_state_dict(best_model_wts)    
    return accuracy_list,loss_list, model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("the device type is", device)

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    cvstudioClient = cvstudio.CVStudio(interactive=True, public = True)
    cvstudioClient.downloadAll()

    percentage_train=0.9
    train_set=cvstudioClient.getDataset(train_test='train',percentage_train=percentage_train)
    val_set=cvstudioClient.getDataset(train_test='test',percentage_train=percentage_train)
    
    i=0
    for x,y  in val_set:
        imshow_(x,"y=: {}".format(str(y.item())))
        i+=1
        if i==3:
            break
    
    n_epochs=10
    batch_size=32
    lr=0.000001
    momentum=0.9
    lr_scheduler=True
    base_lr=0.001
    max_lr=0.01

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    n_classes=train_set.n_classes
    n_classes

    model.fc = nn.Linear(512, n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(dataset=train_set , batch_size=batch_size,shuffle=True)
    validation_loader= torch.utils.data.DataLoader(dataset=val_set , batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,step_size_up=5,mode="triangular2")

    start_datetime = datetime.now()
    start_time=time.time()

    accuracy_list,loss_list, model=train_model(model,train_loader , validation_loader, criterion, optimizer, val_set, device, scheduler, n_epochs=n_epochs )

    end_datetime = datetime.now()
    current_time = time.time()
    elapsed_time = current_time - start_time
    print("elapsed time", elapsed_time )

    torch.save(model.state_dict(), 'model.pt')
    plot_stuff(loss_list,accuracy_list)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, n_classes)
    model.load_state_dict(torch.load( "model.pt"))
    model.eval()

if __name__ == '__main__':
    main()