# Importing libs
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Loading dataset
df = pd.read_csv("dataset/titanic.csv")

# Split features and target
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
y = df["Survived"]

# Define columns
numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# Numeric pipeline
numeric_transformer = SimpleImputer(strategy="mean")

# Categorical pipeline 
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=400))
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.pkl")
print("Saved model to models/pipeline.pkl")
