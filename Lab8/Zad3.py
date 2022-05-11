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


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3,), random_state=275033, max_iter=1000)
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


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=275033, max_iter=1000)
clf.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = clf.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla 4-3-3:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)