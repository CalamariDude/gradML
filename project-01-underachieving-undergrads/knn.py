# Loading packages
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


def calculate_furthest_point_from_com(picture, center_of_mass):
    max_distance = 0
    for x in range(0, 50):
        for y in range(0, 50):
            if picture[x][y] == 1:
                distance = math.sqrt( ((x-center_of_mass[0])**2)+(y-center_of_mass[1])**2) 
                if distance * distance > max_distance:
                    max_distance = distance * distance
    return max_distance

#print(ab_train_data[0])
np.set_printoptions(threshold=sys.maxsize)

print(ab_train_labels.shape)
print(ab_train_data[0].shape)
index = 0
a_data = []
b_data = []
distance_data_a = []
distance_data_b = []
for integers in ab_train_data:
    for x in range(0, 50):
        for y in range(0, 50):
            if integers[x][y] < 0.1:
                integers[x][y] = 0
            else:  
                integers[x][y] = 1
    #print(ab_train_labels[index])
    
    #print(ndimage.measurements.center_of_mass(integers))
    x_axis , y_axis = ndimage.measurements.center_of_mass(integers)
    data_point = [x_axis,y_axis]
    distance = calculate_furthest_point_from_com(integers,ndimage.measurements.center_of_mass(integers))

    if(ab_train_labels[index] == 1):
        a_data.append(data_point)
        distance_data_a.append(distance)
    else:
        b_data.append(data_point)
        distance_data_b.append(distance)

    #plt.imshow(integers, interpolation='nearest')
    #plt.show()
    index = index + 1
print(ab_train_data[1])
#print(len(a_data))
#print(len(b_data))
plt.plot(a_data , 'o', color = 'brown')
plt.plot(b_data, 'o', color = "cyan")
plt.show()

print(distance_data_a)
plt.plot(distance_data_a , 'o', color = 'brown')
plt.plot(distance_data_b, 'o', color = "cyan")
plt.show()

TrainingData, TestData, TrainingTruth, TestTruth = train_test_split(ab_train_data,ab_train_labels,test_size=0.2)
print("The shape of the Training Data is", len(TrainingData))