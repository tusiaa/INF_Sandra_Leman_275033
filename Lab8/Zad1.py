import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=275033)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = knn.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla k=3:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = knn.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla k=5:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)


knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = knn.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla k=11:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)


knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = knn.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print("\nDla k=1:")
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)
