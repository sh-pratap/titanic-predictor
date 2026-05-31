# Importing libs
import pandas as pd

# Loading the dataset
df = pd.read_csv("dataset/titanic.csv")

# Cutting the survived column and placing it at the end
survived = df.pop("Survived")
df["Survived"] = survived

# Dropping the cabin column as it contains a lot of missing values
df.pop("Cabin")

# Splitting the dataset into matrix of features and dependent variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

