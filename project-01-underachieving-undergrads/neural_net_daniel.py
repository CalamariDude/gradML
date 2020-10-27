import loaddata
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage
from PIL import Image
from matplotlib import pyplot as plt
import sys
from scipy import ndimage
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import PIL


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

for integers in ab_train_data:
    for x in range(0, 50):
        for y in range(0, 50):
            if integers[x][y] < 0.1:
                integers[x][y] = 0
            else:  
                integers[x][y] = 1

#print(ab_train_data)

#change format of truth values so that they can correspond to neural network output
new_truth_labels = []
for label in ab_train_labels:
    updated_truth_label = np.array([])
    if label == 1:
        updated_truth_label = np.array([1,0])
    else:
        updated_truth_label = np.array([0,1])
    new_truth_labels.append(updated_truth_label)



#print(type(ab_train_data[0]))
#convert data into tensors
def transform(pictures):
    trans = transforms.Compose([transforms.ToTensor()])
    for p,data in enumerate(pictures):
        pictures[p] = torch.from_numpy(data)

transform(ab_train_data)
transform(new_truth_labels)
for index, _ in enumerate(ab_train_data):
    ab_train_data[index] = torch.unsqueeze(input = ab_train_data[index], dim = 0)

TrainingData, TestData, TrainingTruth, TestTruth = train_test_split(ab_train_data,new_truth_labels,test_size=0.2)

print(type(TrainingData[0]))

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(50*50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

        def forward(self, x):
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

print(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Started Training")
epoch_num = 10
learning_rate = 0.001


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

        print("here")
        outputs = net(inputs)
        print("here")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print statistics
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
print("Finished Training")
