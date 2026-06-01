# Importing libs
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Loading the dataset
df = pd.read_csv("dataset/titanic.csv")

# Cutting the survived column and placing it at the end
survived = df.pop("Survived")
df["Survived"] = survived

# Dropping the useless features
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Splitting the dataset into matrix of features and dependent variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Getting the mean of Age and filling empty values
imputer = SimpleImputer(strategy="mean")
X["Age"] = imputer.fit_transform(X[["Age"]]).ravel()

# Getting the most frequent of Embarked and filling empty values
embarked_imputer = SimpleImputer(strategy="most_frequent")
X["Embarked"] = embarked_imputer.fit_transform(X[["Embarked"]]).ravel()

# Encoding categorical data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), ["Sex", "Embarked"])], remainder="passthrough"
)

X = ct.fit_transform(X)
