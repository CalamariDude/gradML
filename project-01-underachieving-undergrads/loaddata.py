#! /bin/env python3

import pickle
import numpy as np
from matplotlib import pyplot as plt
import sys

def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)

#check if an array is a numpy array 
#works because only numpy appays have the shape paramater
#--update, this isn't needed because the function that reads in the images can handle images that arent np arrays
def check_array(x):
    try:
        x.shape
        return True
    except:
        return False

# index = 0
# x = load_pkl("train_data.pkl")
# train_labels = np.load('finalLabelsTrain.npy')
# for i in x:
# 		print(train_labels[index])
# 		plt.imshow(i, interpolation='nearest')		
# 		plt.show()
# 		index = index + 1

