# project-01-underachieving-undergrads
project-01-underachieving-undergrads created by GitHub Classroom

Hello, TA. Welcome to our project README. 

I will explain how our test.py file works so you can run it smoothly. 

To my understanding, train.py does not need to be run because we have exported our 
trained model into the "logreg.sav" file using scikit-learn. Still:

1. The 'train_model' function is designed like asked for in the project documentation when 
asking for "a *function* that will run your training code on an input data set X and desired output vector Y. 
Any parameter settings must be easy to find and modify."

When running test.py, some things to be aware of:

1. In line 20, insert the name of your test data path where 'train_data.pkl' is. 
2. There are lines that can be commented out to support importing labels. 
3. The 'test_model' function returns the predicted vector like asked for in project
documentation, but also outputs the prediction vector into a .npy file. "A *function* that will run your testing code on
an input data set X. Note: Your test.py code should already be trained and have parameters
set! Any parameter settings must be easy to find and modify. It should return a vector with
the class label associated with each input data point X." 

Libraries used:

import pickle

import numpy as np

from scipy import ndimage

import skimage

import pandas as pd

from sklearn.linear_model import LogisticRegression



