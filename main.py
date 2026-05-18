from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Medical Insurance Cost Predictor")

# মডেল এবং স্কেলার লোড করা
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")

# ইউজার ইনপুটের ফরম্যাট ঠিক করার জন্য Pydantic Model
class CustomerData(BaseModel):
    age: float
    bmi: float
    children: int
    sex_male: int      # 1 = Male, 0 = Female
    smoker_yes: int    # 1 = Yes, 0 = No
    region_northwest: int
    region_southeast: int
    region_southwest: int

@app.get("/")
def home():
    return {"message": "Welcome to Medical Insurance Cost Predictor API! Use /docs for testing."}

@app.post("/predict")
def predict_cost(data: CustomerData):
    # ইনপুট ডাটাকে অ্যারেতে রূপান্তর
    input_data = np.array([[
        data.age, data.bmi, data.children, data.sex_male, 
        data.smoker_yes, data.region_northwest, data.region_southeast, data.region_southwest
    ]])
    
    # স্কেলিং করা
    scaled_data = scaler.transform(input_data)
    
    # প্রেডিকশন
    prediction = model.predict(scaled_data)
    
    return {"predicted_insurance_cost": round(float(prediction[0]), 2)}