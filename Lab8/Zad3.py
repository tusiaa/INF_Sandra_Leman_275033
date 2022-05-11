import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
# setosa -> 0, versicolor -> 1, virginica -> 2
df[['class']] = df[['class']].replace(['setosa', 'versicolor', 'virginica'], [0, 1, 2])
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=275033)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=275033)
clf.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = clf.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla 4-2-1:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=275033, max_iter=1000)
clf.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = clf.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla 4-3-1:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=275033, max_iter=1000)
clf.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = clf.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla 4-3-3-1:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)

# 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
train_classes2 = np.zeros((train_set.shape[0], 3))
test_classes2 = np.zeros((test_set.shape[0], 3))
for i in range(train_set.shape[0]):
    if train_classes[i] == 0:
        train_classes2[i][0] = 1
    elif train_classes[i] == 1:
        train_classes2[i][1] = 1
    elif train_classes[i] == 2:
        train_classes2[i][2] = 1

for i in range(test_set.shape[0]):
    if test_classes[i] == 0:
        test_classes2[i][0] = 1
    elif test_classes[i] == 1:
        test_classes2[i][1] = 1
    elif test_classes[i] == 2:
        test_classes2[i][2] = 1

# Zmieniona alpha (inaczej wskazuje same [0, 0, 0])
clf = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(3,), random_state=275033, max_iter=1000)
clf.fit(train_inputs, train_classes2)

good_predictions = 0
len = test_set.shape[0]
prediction = clf.predict(test_inputs)
for i in range(len):
    if np.array_equal(prediction[i], test_classes2[i]):
        good_predictions = good_predictions + 1

print("\nDla 4-3-3:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes2.argmax(axis=1), prediction.argmax(axis=1))
print(cm)
