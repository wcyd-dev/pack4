from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import os

# Load the trained model
model = joblib.load('sales_predictor_model.pkl')

# Create FastAPI app
app = FastAPI()

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open('templates/index.html', 'r') as file:
        return file.read()

# Define the input data format
class SalesData(BaseModel):
    Marketing_Spend: float
    Price: float
    Inventory_Level: float
    Economic_Conditions: float
    Adherence_Rate: float

# Prediction endpoint
@app.post("/predict/")
def predict_sales(data: SalesData):
    # Convert input data into a format suitable for prediction
    input_data = [[data.Marketing_Spend, data.Price, data.Inventory_Level, data.Economic_Conditions, data.Adherence_Rate]]

    # Make prediction
    prediction = model.predict(input_data)

    # Return the predicted sales
    return {"predicted_sales": prediction[0]}
