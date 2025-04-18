from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("asthma_model.pkl")

class Symptoms(BaseModel):
    symptoms: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the Asthma Prediction API"}

@app.post("/predict")
def predict(data: Symptoms):
    symptoms = data.symptoms
    try:
        prediction = model.predict([symptoms])[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}
