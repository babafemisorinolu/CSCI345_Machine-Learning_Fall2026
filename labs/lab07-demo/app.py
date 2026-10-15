# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Iris Classifier API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Load model at startup (not per-request!)
model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    sepal_length: float = Field(gt=0, le=20)
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float

CLASSES = ["setosa", "versicolor", "virginica"]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])
    
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    return PredictionResponse(
        prediction=int(pred),
        class_name=CLASSES[pred],
        confidence=round(float(proba[pred]), 4)
    )