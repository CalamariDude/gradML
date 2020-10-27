import loaddata
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Loading in the train data
train_data = loaddata.load_pkl('train_data.pkl')

# Loading in the labels
train_labels = np.load('finalLabelsTrain.npy')

# When attempting to only classify a and b, looking only at reduced set.
ab_train_data = train_data[np.logical_or((train_labels == 1),(train_labels == 2))]
ab_train_labels = train_labels[np.logical_or((train_labels == 1),(train_labels == 2))]

newdata = []
for i in range(len(train_data)):
    # print(np.asarray(train_data[i]).shape)
    data = np.asarray(train_data[i])
    data = data.flatten()
    length = len(data)
    if length < 50*50:
        amount = 50*50 - length
        data = np.pad(data, (0,amount), 'constant')
    train_data[i] = data
    newdata.append(data)

X_train, X_test, y_train, y_test = train_test_split(newdata, train_labels, train_size=.7)

pca = PCA(n_components=15)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf_pca = SVC(gamma='auto')
clf_pca.fit(X_train_pca, y_train);
y_pred_pca = clf_pca.predict(X_test_pca)

print("Classification report of raw picture")
print(classification_report(y_pred, y_test))

print("Classification report of pca signals")
print(classification_report(y_pred_pca, y_test))


