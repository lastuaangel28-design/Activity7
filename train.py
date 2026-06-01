
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/train/iris/Iris.csv")

X = df.iloc[:,1:5]
y = df["Species"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

model.fit(X, y)

joblib.dump(model, "iris_model.pkl")
print("Model saved as iris_model.pkl")
