import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=275033)

print(train_set)
print(test_set)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs, train_classes)

tree.plot_tree(clf)
plt.show()

r = tree.export_text(clf)
print(r)

good_predictions = 0
len = test_set.shape[0]
prediction = clf.predict(test_inputs)

for i in range(len):
    if prediction[i] == test_classes[i]:
        good_predictions = good_predictions + 1

# Takie samo moje drzewko :)
print(good_predictions)
print(good_predictions / len * 100, "%")

cm = confusion_matrix(test_classes, prediction)

print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

