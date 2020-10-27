# Loading in the train data
train_data = loaddata.load_pkl('train_data.pkl')

# Loading in the labels
train_labels = np.load('finalLabelsTrain.npy')

# Flattening and converting to data frame
for i in range(len(train_data)):
    train_data[i] = skimage.transform.resize(np.asarray(train_data[i]), (50,50))
    train_data[i] = train_data[i].flatten()
train_data = pd.DataFrame(np.vstack(train_data))

# Simple split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=20)
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Applying lda
lda = LDA(n_components=7)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Using a random forest classifier to assess accuracy of model
classifier = RandomForestClassifier(max_depth=8, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Output results
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))
