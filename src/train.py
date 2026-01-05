# src/train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading dataset...")

data = pd.read_csv("data/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

# -----------------------------
# 2. Train-test split
# -----------------------------
print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Create ML pipeline
# -----------------------------
print("Creating pipeline...")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# -----------------------------
# 4. Train model
# -----------------------------
print("Training model...")

pipeline.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate model
# -----------------------------
print("Evaluating model...")

predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy}")

# -----------------------------
# 6. Save trained model
# -----------------------------
print("Saving model...")

joblib.dump(pipeline, "model.pkl")

print("Training completed successfully!")
