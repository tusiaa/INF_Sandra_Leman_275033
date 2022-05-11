import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=275033)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

gnb = GaussianNB()
gnb.fit(train_inputs, train_classes)

good_predictions = 0
len = test_set.shape[0]
prediction = gnb.predict(test_inputs)
for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)
print(cm)



