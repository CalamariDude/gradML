# -*- coding: utf-8 -*-
"""
#
@author: {Last Name} {First Name} 
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train_net import train_model

#Network parameters
epochs = 100
lr_rate = .1

# Number of units in each hidden layer
# List is [units in 1st hidden layer, units in 2nd hidden layer]
num_units = [1,1]

#Print training loss every # of progress epochs
progress = 100

#Use mean squared error loss function
loss_fxn = nn.MSELoss()

#Load data
dataset = torch.Tensor(np.load('UF_Training_Data.npy'))

#Create training data and labels, convert to tensor for Pytorch
training_data = dataset[:,:-1]
desired = dataset[:,-1]

#Visualize training data
colors = {1: "orange", -1: "blue"}
color_list = []
for label in range(0,len(desired)):
    color_list.append(colors[desired.numpy()[label]])
plt.figure()
plt.scatter(training_data[:,0],training_data[:,1],c=color_list)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training Data")

#Compute the input dimension and initialize the error track
num_feats = training_data.shape[1]
Error_track = np.zeros([epochs])

#Create model
class UF_NET(nn.Module):
    def __init__(self,num_feats,num_units):
        super(UF_NET,self).__init__()
        self.first_hidden_layer = nn.Linear(num_feats,num_units[0])
        self.second_hidden_layer = nn.Linear(num_units[0],num_units[1])
        self.output_layer = nn.Linear(num_units[1],1)
        self.activation = nn.Tanh()
    def forward(self,x):
        x = self.activation(self.first_hidden_layer(x))
        x = self.activation(self.second_hidden_layer(x))
        x = self.activation(self.output_layer(x))
        return x

#Initialize model
model = UF_NET(num_feats,num_units)
#model.load_state_dict(best_model_wts)

#Train model
best_model_wts, Error_track = train_model(model,epochs,loss_fxn,lr_rate,
                                         training_data,desired,progress)

#Take weights learned and load into model
model.load_state_dict(best_model_wts)
    

#Visualize decision boundary using meshgrid of points
plt.figure()
x1 = np.arange(-2.5,2.5,0.1)
x2 = np.arange(-2.5,2.5,0.1)
x1,x2 = np.meshgrid(x1,x2)
X = torch.Tensor(np.c_[x1.ravel(),x2.ravel()])
model.eval()
outputs = model(X)
plt.contourf(x1,x2,outputs.detach().reshape(x1.shape),alpha=.8)
plt.colorbar()