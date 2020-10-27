# Importing libraries
import loaddata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Loading in training data, I ran on my 50 x 50 data but Delaney mentioned
# that it should only be scaled in one direction.

# Loading in the train data
train_data = loaddata.load_pkl('ab_train_data.pkl')

# Loading in the labels
train_labels = np.load('ab_train_labels.pkl')

# Flattening and converting to data frame
for i in range(len(train_data)):
    train_data[i] = train_data[i].flatten()
train_data = pd.DataFrame(np.vstack(train_data))

# Simple split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=20)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying lda
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Using a random forest classifier to assess accuracy of model
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Results
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))
