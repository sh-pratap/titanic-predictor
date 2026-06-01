import joblib
import pandas as pd

# Load model
model = joblib.load("models/pipeline.pkl")

# Sample inputs
sample = pd.DataFrame([{
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 70.25,
    "Embarked": "S"
}])

prediction = model.predict(sample)[0]

print("Prediction:", "Survived" if prediction == 1 else "Did not survive")
