# -*- coding: utf-8 -*-
"""
File:   hw0.py
Author: Jad Zeineddine
Date:   Sept 6, 2019
Desc:   Homework 1A: Comparing linear regression with different loss functions
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """


def plotK(Kvalues, error_IRLS):
    p1 = plt.plot(Kvalues, error_IRLS, 'g')
    plt.title("Absolute error of IRLS by varying K")
    plt.ylabel("Absolute Error")
    plt.xlabel("K Value")
    plt.xlim((0,1))
    return 0


def plotM(Mvalues, error_LS, error_IRLS):
    legend = ["LS", "IRLS"]
    p1 = plt.plot(Mvalues, error_LS, 'g')
    p2 = plt.plot(Mvalues, error_IRLS, 'r')
    plt.title("Absolute error of least-squares vs IRLS by varying M")
    plt.ylabel("Absolute Error")
    plt.xlabel("M value")
    plt.xlim((1,20))
    plt.legend((p1[0],p2[0]),legend)
    return 0



def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data

    #add title, legend and axes labels
    plt.title("Comparing MHuber loss to Least Squares")
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    plt.xlim((-4.5,4.5))
    plt.ylim((-2, 2))
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
      
def fitdataLS(x,t,M):
    '''fitdataLS(x,t,M): Fit a polynomial of order M to the data (x,t) using LS''' 
    #This needs to be filled in
    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w
        
def fitdataIRLS(x,t,M,k):
    '''fitdataIRLS(x,t,M,k): Fit a polynomial of order M to the data (x,t) using IRLS''' 
    #This needs to be filled in

    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X) @ X.T @ t
    ybar = X @ w
    tminy = abs(t-ybar)
    b = np.zeros((X@X.T).shape)
    numiter = 0

    while (numiter < 10):
        print((X@X.T).shape)
        bnew = np.zeros((X@X.T).shape)
        for i in range(b.shape[0]):
            if tminy[i] <= k:
                bnew[i][i] = 1
            elif tminy[i] > k:
                bnew[i][i] = k/tminy[i]
        wnew = np.linalg.inv(X.T@bnew@X) @ X.T @ bnew @ t
        wdiff=np.linalg.norm(wnew - w)
        bdiff=np.linalg.norm(bnew - b)
        print("wdiff", wdiff)
        print("bdiff", bdiff)
        if wdiff < .01 and bdiff < .01:
            break
        else:
            w = wnew
            b = bnew
        numiter += 1
        #check for convergence
    return w
        

""" ======================  Variable Declaration ========================== """
M =  10 #regression model order
k = .01 #Huber M-estimator tuning parameter

""" =======================  Load Training Data ======================= """
data_uniform = np.load('TrainData.npy')
x1 = data_uniform[:,0]
t1 = data_uniform[:,1]

""" ========================  Train the Model ============================= """
wLS = fitdataLS(x1,t1,M) 
wIRLS = fitdataIRLS(x1,t1,M,k) 


""" ======================== Load Test Data  and Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
test_uniform = np.load('TestData.npy')
x2 = data_uniform[:,0]
t2 = data_uniform[:,1]

""" ========================  Plot Results ============================== """

""" This is where you should create the plots requested """

x3 = x2
X3 = np.array([x3**m for m in range(M+1)]).T
X2 = X3
t_2 = X2 @ wLS
t3 =  X3 @ wIRLS
plt.figure()
plotData(x1,t1,x2,t_2,x3,t3, legend=["Data", "Least Squares", "M-Huber"])
plt.figure()
Mvalues = np.arange(1,20)
error_ls = []
error_irls = []
for M in Mvalues:
    X3 = np.array([x3**m for m in range(M+1)]).T
    wLS = fitdataLS(x1,t1,M) 
    wIRLS = fitdataIRLS(x1,t1,M,k) 
    t_LS = X3 @ wLS
    t_IRLS = X3 @ wIRLS
    errorls = sum(abs(t_LS - t2))
    errorirls = sum(abs(t_IRLS-t2))
    error_ls.append(errorls)
    error_irls.append(errorirls)
plotM(Mvalues,error_ls,error_irls)
M = 10
Kvalues = [.0001, .001, .005, .01, .05, .09, .14, .3, .5, .7, .9 ]
error_irls = []
for k in Kvalues:
    X3 = np.array([x3**m for m in range(M+1)]).T
    wIRLS = fitdataIRLS(x1,t1,M,k)
    t_IRLS = X3 @ wIRLS
    errorirls = sum(abs(t_IRLS-t2))
    error_irls.append(errorirls)
plt.figure()
plotK(Kvalues, error_irls)
plt.show()
