"""
File:   hw02.py
Author: Jad Zeineddine
Date:   10/03
Desc:   Assignment 02B 
    
"""
##Imports
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
train1_file = 'CrabDatasetforTrain.txt'
train2_file = '10dDatasetforTrain.txt'
test1_file =  'CrabDatasetforTest.txt'
test2_file =  '10dDatasetforTest.txt'

train1 = pd.read_csv(train1_file, sep=" ", header=None)
train2 = pd.read_csv(train2_file, sep=" ", header=None)
test1  = pd.read_csv(test1_file, sep=" ", header=None)
test2 =  pd.read_csv(test2_file, sep=" ", header=None)


def vote(datapoint, X, y, k):
    distances = np.linalg.norm(X-datapoint, axis = 1)
    tosort = np.vstack([distances, y]).T
    tosort = tosort[tosort[:,0].argsort()]
    voters = np.asarray(tosort[:k,1])
    counter1 = 0
    counter2 = 0
    for i in voters:
        if i > 0:
            counter2 += 1
        else:
            counter1 += 1
    if counter2 >  counter1:
        return 1
    return 0 

def knn(X, y, x_test, k=3):
    pred = []
    for datapoint in x_test:
        pred.append(vote(datapoint, X, y, k))
    return pred


def splitter(data, targets):
    class1 = []
    class2 = []
    for i in range(len(targets)):
        if (int(targets[i]) == 0):
            class1.append(data[i])
        else:
            class2.append(data[i])
    return np.asarray(class1), np.asarray(class2)


#########################################################
title = "Crab Dataset"
print("*********", title, "**********")
train1 = np.asarray(train1)
test1 = np.asarray(test1)

train1_X = train1[:,:-1]
train1_y = train1[:,-1]
test1_X = test1[:,:-1]
test1_y = test1[:,-1]

##Split the data into classes
train1_X_1, train1_X_2 = splitter(train1_X, train1_y)

mu1_1 = np.mean(train1_X_1, axis = 0)
mu1_2 = np.mean(train1_X_2, axis = 0)
cov1_1 = np.cov(train1_X_1.T)
cov1_2 = np.cov(train1_X_1.T)

Pc1 = float(len(train1_X_1)) / (len(train1_X_1) + len(train1_X_2))
Pc2 = float(len(train1_X_2)) / (len(train1_X_1) + len(train1_X_2))

#look at the pdf for class 1

#singular matrix so make it non singular by perturbing it a bit
cov1_1 += .01 * np.identity(len(cov1_1))
cov1_2 += .01 * np.identity(len(cov1_2))

probsclass1 = multivariate_normal.pdf(test1_X, mean=mu1_1, cov=cov1_1, allow_singular=False)
probsclass2 = multivariate_normal.pdf(test1_X, mean=mu1_2, cov=cov1_2, allow_singular=False)

probsclass1 = np.asarray(probsclass1)
probsclass2 = np.asarray(probsclass2)


pred = []
for i in range(len(probsclass1)):
    if(probsclass1[i] >probsclass2[i]):
        pred.append(0)
    else:
        pred.append(1)
print("Classification report for MAP")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))

scores = []
threshs = np.arange(0,1,.005)
for thresh in threshs:
    pred = []
    for i in range(len(probsclass1)):
        if(probsclass1[i] > thresh):
            pred.append(0)
        else:
            pred.append(1)
    scores.append(accuracy_score(test1_y, pred))
plt.figure()
plt.plot(threshs,scores, color='blue')
plt.title("ROC curve for"+ title)
plt.ylabel("Threshold")
plt.xlabel("Score")
print("\n\n")

## KNN time
scores = []
ks = range(1,15)
for k in ks:
    pred = knn(train1_X, train1_y, test1_X, k)
    scores.append(accuracy_score(test1_y, pred))
best = (np.argmax(scores)+1)
print("Best k value = ", best)
pred = knn(train1_X, train1_y, test1_X, best)
plt.figure()
plt.plot(ks, scores, color='r')
plt.title("KNN with different values of K for "+ title)
plt.xlabel("K")
plt.ylabel("Score")
plt.ylim((0,1))
print("Classification report for KNN")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))



############################################
############################################
############################################
############################################

title = "10 Dataset"

print("*********", title, "**********")
train1 = np.asarray(train2)
test1 = np.asarray(test2)

train1_X = train1[:,:-1]
train1_y = train1[:,-1]
test1_X = test1[:,:-1]
test1_y = test1[:,-1]

plt.figure()
plt.scatter(train1_X[:,3], train1_X[:,4])
plt.title("Scatter of "+ title)
plt.xlabel("4th Feature")
plt.ylabel("5th Feature")
##Split the data into classes
train1_X_1, train1_X_2 = splitter(train1_X, train1_y)

mu1_1 = np.mean(train1_X_1, axis = 0)
mu1_2 = np.mean(train1_X_2, axis = 0)
cov1_1 = np.cov(train1_X_1.T)
cov1_2 = np.cov(train1_X_1.T)

Pc1 = float(len(train1_X_1)) / (len(train1_X_1) + len(train1_X_2))
Pc2 = float(len(train1_X_2)) / (len(train1_X_1) + len(train1_X_2))

#look at the pdf for class 1

#singular matrix so make it non singular by perturbing it a bit
cov1_1 += .01 * np.identity(len(cov1_1))
cov1_2 += .01 * np.identity(len(cov1_2))

probsclass1 = multivariate_normal.pdf(test1_X, mean=mu1_1, cov=cov1_1, allow_singular=False)
probsclass2 = multivariate_normal.pdf(test1_X, mean=mu1_2, cov=cov1_2, allow_singular=False)

probsclass1 = np.asarray(probsclass1)
probsclass2 = np.asarray(probsclass2)


pred = []
for i in range(len(probsclass1)):
    if(probsclass1[i] >probsclass2[i]):
        pred.append(0)
    else:
        pred.append(1)
print("Classification report for MAP")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))

scores = []
threshs = np.arange(0,1,.005)
for thresh in threshs:
    pred = []
    for i in range(len(probsclass1)):
        if(probsclass1[i] > thresh):
            pred.append(0)
        else:
            pred.append(1)
    scores.append(accuracy_score(test1_y, pred))
plt.figure()
plt.plot(threshs,scores, color='blue')
plt.title("ROC curve for"+ title)
plt.ylabel("Threshold")
plt.xlabel("Score")
print("\n\n")

## KNN time
scores = []
ks = range(1,15)
for k in ks:
    pred = knn(train1_X, train1_y, test1_X, k)
    scores.append(accuracy_score(test1_y, pred))
best = (np.argmax(scores)+1)
print("Best k value = ", best)
pred = knn(train1_X, train1_y, test1_X, best)
plt.figure()
plt.plot(ks, scores, color='r')
plt.title("KNN with different values of K for "+ title)
plt.xlabel("K")
plt.ylabel("Score")
plt.ylim((0,1))
print("Classification report for KNN")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))



####### Cross validation
title = "10 Dataset cross validation"

print("*********", title, "**********")
train1 = np.asarray(train2)
test1 = np.asarray(test2)

train1_X = train1[:,:-1]
train1_y = train1[:,-1]


train1_X, test1_X, train1_y, test1_y = train_test_split(train1_X, train1_y , test_size = .7, shuffle = True)

##Split the data into classes
train1_X_1, train1_X_2 = splitter(train1_X, train1_y)

mu1_1 = np.mean(train1_X_1, axis = 0)
mu1_2 = np.mean(train1_X_2, axis = 0)
cov1_1 = np.cov(train1_X_1.T)
cov1_2 = np.cov(train1_X_1.T)

Pc1 = float(len(train1_X_1)) / (len(train1_X_1) + len(train1_X_2))
Pc2 = float(len(train1_X_2)) / (len(train1_X_1) + len(train1_X_2))

#look at the pdf for class 1

#singular matrix so make it non singular by perturbing it a bit
cov1_1 += .01 * np.identity(len(cov1_1))
cov1_2 += .01 * np.identity(len(cov1_2))

probsclass1 = multivariate_normal.pdf(test1_X, mean=mu1_1, cov=cov1_1, allow_singular=False)
probsclass2 = multivariate_normal.pdf(test1_X, mean=mu1_2, cov=cov1_2, allow_singular=False)

probsclass1 = np.asarray(probsclass1)
probsclass2 = np.asarray(probsclass2)


pred = []
for i in range(len(probsclass1)):
    if(probsclass1[i] >probsclass2[i]):
        pred.append(0)
    else:
        pred.append(1)
print("Classification report for MAP")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))

scores = []
threshs = np.arange(0,1,.005)
for thresh in threshs:
    pred = []
    for i in range(len(probsclass1)):
        if(probsclass1[i] > thresh):
            pred.append(0)
        else:
            pred.append(1)
    scores.append(accuracy_score(test1_y, pred))
plt.figure()
plt.plot(threshs,scores, color='blue')
plt.title("ROC curve for"+ title)
plt.ylabel("Threshold")
plt.xlabel("Score")
print("\n\n")

## KNN time
scores = []
ks = range(1,15)
for k in ks:
    pred = knn(train1_X, train1_y, test1_X, k)
    scores.append(accuracy_score(test1_y, pred))
best = (np.argmax(scores)+1)
print("Best k value = ", best)
pred = knn(train1_X, train1_y, test1_X, best)
plt.figure()
plt.plot(ks, scores, color='r')
plt.title("KNN with different values of K for "+ title)
plt.xlabel("K")
plt.ylabel("Score")
plt.ylim((0,1))
print("Classification report for KNN")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))

#####

###




title = "Crab cross validation"

print("*********", title, "**********")
train1 = np.asarray(train1)
test1 = np.asarray(test1)

train1_X = train1[:,:-1]
train1_y = train1[:,-1]


train1_X, test1_X, train1_y, test1_y = train_test_split(train1_X, train1_y , test_size = .7, shuffle = True)

##Split the data into classes
train1_X_1, train1_X_2 = splitter(train1_X, train1_y)

mu1_1 = np.mean(train1_X_1, axis = 0)
mu1_2 = np.mean(train1_X_2, axis = 0)
cov1_1 = np.cov(train1_X_1.T)
cov1_2 = np.cov(train1_X_1.T)

Pc1 = float(len(train1_X_1)) / (len(train1_X_1) + len(train1_X_2))
Pc2 = float(len(train1_X_2)) / (len(train1_X_1) + len(train1_X_2))

#look at the pdf for class 1

#singular matrix so make it non singular by perturbing it a bit
cov1_1 += .01 * np.identity(len(cov1_1))
cov1_2 += .01 * np.identity(len(cov1_2))

probsclass1 = multivariate_normal.pdf(test1_X, mean=mu1_1, cov=cov1_1, allow_singular=False)
probsclass2 = multivariate_normal.pdf(test1_X, mean=mu1_2, cov=cov1_2, allow_singular=False)

probsclass1 = np.asarray(probsclass1)
probsclass2 = np.asarray(probsclass2)


pred = []
for i in range(len(probsclass1)):
    if(probsclass1[i] >probsclass2[i]):
        pred.append(0)
    else:
        pred.append(1)
print("Classification report for MAP")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))

scores = []
threshs = np.arange(0,1,.005)
for thresh in threshs:
    pred = []
    for i in range(len(probsclass1)):
        if(probsclass1[i] > thresh):
            pred.append(0)
        else:
            pred.append(1)
    scores.append(accuracy_score(test1_y, pred))
plt.figure()
plt.plot(threshs,scores, color='blue')
plt.title("ROC curve for"+ title)
plt.ylabel("Threshold")
plt.xlabel("Score")
print("\n\n")

## KNN time
scores = []
ks = range(1,15)
for k in ks:
    pred = knn(train1_X, train1_y, test1_X, k)
    scores.append(accuracy_score(test1_y, pred))
best = (np.argmax(scores)+1)
print("Best k value = ", best)
pred = knn(train1_X, train1_y, test1_X, best)
plt.figure()
plt.plot(ks, scores, color='r')
plt.title("KNN with different values of K for "+ title)
plt.xlabel("K")
plt.ylabel("Score")
plt.ylim((0,1))
print("Classification report for KNN")
print(classification_report(test1_y, pred))
print("Confusion Matrix (Rows are true, Prediction are columms)")
print(confusion_matrix(test1_y, pred))


plt.show()