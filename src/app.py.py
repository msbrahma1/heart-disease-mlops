from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict

# Initialize FastAPI
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps REST API for Heart Disease Prediction",
    version="1.0"
)

# Load trained model
model = load_model("heart_disease_best_model.pkl")

# Input schema
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Health check
@app.get("/")
def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict_heart_disease(data: HeartDiseaseInput):

    input_data = data.dict()
    prediction = predict(model, input_data)

    return {
        "prediction": int(prediction),
        "message": "1 = Heart Disease, 0 = No Heart Disease"
    }
