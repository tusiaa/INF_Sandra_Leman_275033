import pandas as pd

missing_values = ["n/a", "na", "NA", "-", "--"]
df = pd.read_csv("iris_with_errors.csv", na_values=missing_values)

print("total missing values: ", df.isnull().sum().sum())
print(df.isnull().sum())


def num_data_check(item, column):
    if 0 < item < 15:
        return item
    else:
        return df[column].median()


print("sepal.length before:", df["sepal.length"].between(0, 15, inclusive="neither").sum(), "/ 150")
df["sepal.length"] = df["sepal.length"].map(lambda x: num_data_check(x, "sepal.length"))
print("sepal.length after:", df["sepal.length"].between(0, 15, inclusive="neither").sum(), "/ 150")

print("sepal.width before:", df["sepal.width"].between(0, 15, inclusive="neither").sum(), "/ 150")
df["sepal.width"] = df["sepal.width"].map(lambda x: num_data_check(x, "sepal.width"))
print("sepal.width after:", df["sepal.width"].between(0, 15, inclusive="neither").sum(), "/ 150")

print("petal.length before:", df["petal.length"].between(0, 15, inclusive="neither").sum(), "/ 150")
df["petal.length"] = df["petal.length"].map(lambda x: num_data_check(x, "petal.length"))
print("petal.length after:", df["petal.length"].between(0, 15, inclusive="neither").sum(), "/ 150")

print("petal.width before:", df["petal.width"].between(0, 15, inclusive="neither").sum(), "/ 150")
df["petal.width"] = df["petal.width"].map(lambda x: num_data_check(x, "petal.width"))
print("petal.width after:", df["petal.width"].between(0, 15, inclusive="neither").sum(), "/ 150")

wrong = []


def variety_check(item):
    if item in ["Setosa", "Versicolor", "Virginica"]:
        return item
    elif item.capitalize() in ["Setosa", "Versicolor", "Virginica"]:
        wrong.append(item)
        return item.capitalize()
    elif item.capitalize() == "Versicolour":
        wrong.append(item)
        return "Versicolor"
    else:
        wrong.append(item)


print("variety before:", df["variety"].isin(["Setosa", "Versicolor", "Virginica"]).sum(), "/ 150")
df["variety"] = df["variety"].map(lambda x: variety_check(x))
print("Wrong values: ", wrong)
print("variety after:", df["variety"].isin(["Setosa", "Versicolor", "Virginica"]).sum(), "/ 150")






