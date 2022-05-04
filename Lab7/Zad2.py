import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=275033)

print(train_set[train_set[:, 4].argsort()])


def classify_iris(sl, sw, pl, pw):
    if pw < 0.7:
        return ("setosa")
    elif pl+pw >= 6:
        return ("virginica")
    else:
        return ("versicolor")


good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[:, 0:4][i][0], test_set[:, 0:4][i][1],
                     test_set[:, 0:4][i][2], test_set[:, 0:4][i][3]) == test_set[:, 4][i]:
        good_predictions = good_predictions + 1

print(good_predictions)
print(good_predictions / len * 100, "%")
