from __future__ import print_function
import torch
import loaddata
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os, sys
from itertools import chain 
from sklearn.model_selection import train_test_split
import torch.optim as optim
from PIL import Image
import PIL
from scipy import ndimage
import skimage


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading in the train data
train_data = loaddata.load_pkl('train_data.pkl')

# Loading in the labels
train_labels = np.load('finalLabelsTrain.npy')

# When attempting to only classify a and b, looking only at reduced set.
ab_train_data = train_data[np.logical_or((train_labels == 1),(train_labels == 2))]
ab_train_labels = train_labels[np.logical_or((train_labels == 1),(train_labels == 2))]

# It appears that in the original data, the letter switches after every count of 9

# List of a,b data points that need to be rotated - this is not used

rot_list = []
j = 0

for i in range(1600):
    if(np.shape(ab_train_data[i])[0] < np.shape(ab_train_data[i])[1]):
        rot_list.append(j)
    j = j + 1


# The list actually had many points that do not need to be rotated...

rot_list = [241,242,243,244,245,246,247,250,
           251,252,253,254,255,256,257,258,259,
           500,501,502,503,504,505,506,507,508,509,
           510,511,512,513,514,515,516,517,518,519]

# Some things should just be thrown out
trash_list = [240,248,249,960]

# Indices of interest
# 480 is an example of a well written "a" that was not automatically centered
#whatsoever. Many examples of this.


# Conclusions from above:

# Not actually that many bad data points. Possible I missed some since 
# I was going by every 10 then checking every one once I found a bad spot,
# but unlikely since the same individual does every ten
# There were many cases though where resizing is important.
# Every rotated image is solved by 270 degrees

# Rotating the above images by 270 degrees, seems to be the only way things went wrong
for index in rot_list:
    img = (ab_train_data[index])
    lx, ly = img.shape
    rot_img = ndimage.rotate(img, 270)
    ab_train_data[index] = rot_img

# For every image we resize to (50,50)
for i in range(1600):
    ab_train_data[i] = skimage.transform.resize(np.asarray(ab_train_data[i]), (50,50))


for i,integers in enumerate(ab_train_data):
    for x in range(0, 50):
        for y in range(0, 50):
            if integers[x][y] < 0.1:
                integers[x][y] = 0
            else:  
                integers[x][y] = 1
    ab_train_data[i] = integers

plt.imshow(ab_train_data[0] , interpolation='nearest')
plt.show()

new_truth_labels = []
for label in ab_train_labels:
    if label == 1:
        new_truth_labels.append(1)
    else:
        new_truth_labels.append(2)

ab_train_data = ab_train_data.tolist()

TrainingData, TestData, TrainingTruth, TestTruth = train_test_split(ab_train_data,new_truth_labels,test_size=0.2)

def transform(pictures, truth):
    trans= transforms.Compose([transforms.ToTensor()])
    for p,data in enumerate(pictures):
        normal = data

        normal = PIL.Image.fromarray(normal)
        normal = trans(normal)
        pictures[p] = normal

        add_to_truth = torch.zeros(1,dtype=torch.long)
        add_to_truth[0] = truth[p]

        truth[p] = add_to_truth

transform(TrainingData, TrainingTruth)
transform(TestData, TestTruth)
for index, _ in enumerate(TrainingData):
    TrainingData[index] = torch.unsqueeze(input = TrainingData[index], dim = 0)
for index, _ in enumerate(TestData):
    TestData[index] = torch.unsqueeze(input = TestData[index], dim = 0)

# class Unit(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(Unit,self).__init__()
        

#         self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
#         self.bn = nn.BatchNorm2d(num_features=out_channels)
#         self.relu = nn.ReLU()

#     def forward(self,input):
#         output = self.conv(input)
#         output = self.bn(output)
#         output = self.relu(output)

#         return output

# class SimpleNet(nn.Module):
#     def __init__(self,num_classes=2):
#         super(SimpleNet,self).__init__()
        
#         #Create 14 layers of the unit with max pooling in between
#         self.unit1 = Unit(in_channels=1,out_channels=16)
#         self.unit3 = Unit(in_channels=16, out_channels=16)

#         self.pool1 = nn.MaxPool2d(kernel_size=2)

#         self.unit4 = Unit(in_channels=16, out_channels=32)
#         self.unit7 = Unit(in_channels=32, out_channels=32)


#         self.unit8 = Unit(in_channels=32, out_channels=64)

#         self.pool2 = nn.MaxPool2d(kernel_size=2)

#         self.unit14 = Unit(in_channels=64, out_channels=64)
#         self.dropout = nn.Dropout(p=0.5)

        
#         #Add all the units into the Sequential layer in exact order
#         self.net = nn.Sequential(self.unit1, self.unit3, self.pool1, self.unit4
#                                  ,self.unit7,  self.unit8, self.pool2, 
#                                  self.unit14, self.dropout)

#         self.fc = nn.Linear(in_features=9216,out_features=num_classes)

#     def forward(self, input):
#         output = self.net(input)
#         output = output.view(-1,9216)
#         output = self.fc(output)
#         return output
      
class SimpleNet(nn.Module):
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(50*50, 15)  # 6*6 from image dimension
        self.fc2 = nn.Linear(15, 2)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

print("The CNN has been defined")

net = SimpleNet()
net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

epoch_num = 10


for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss = 0
    for i, data in enumerate(TrainingData, start = 0):
        inputs = data
#         if(list(inputs.size()) != [1, 1, 100, 100]):
#             print("skipped because the size turned out to be", inputs.size())
#             continue

        labels = TrainingTruth[i]

        inputs, labels = inputs.to(device), labels.to(device)
                
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print statistics
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

correct = 0
total = 0
with torch.no_grad():
    for index, data in enumerate(TestData):
        #images, labels = data
        images= data
        #print(images.size() )
        #print(images)
#         if(list(images.size()) != [1, 1, 100, 100]):
#             print("skipped")
#             continue
        labels = TestTruth[index]
        images, labels = images.to(device), labels.to(device)
        #print("The label is",labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        total += 1
        if(labels == predicted):
            correct = correct +1
        #correct += (predicted == labels).sum().item()

print('Accuracy of the network on the',total, 'test images: %d %%' % (
    100 * correct / total))