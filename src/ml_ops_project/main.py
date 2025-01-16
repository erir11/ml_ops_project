from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List

# Mock model to simulate predictions
class MockModel:
    def predict(self, X):
        # Always return a fixed prediction for testing
        return [0] * len(X)

# Define input schema
class ImagePathInput(BaseModel):
    image_path: str  # Path to the input image

# Initialize FastAPI app
app = FastAPI()

# Use a mock model for testing
model = MockModel()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI ML App!"}

@app.post("/predict")
def predict(input_data: ImagePathInput):
    try:
        # Simulate preprocessing (normally you'd process the image here)
        fake_input = np.random.rand(1, 3, 224, 224)  # Example input shape

        # Simulate prediction
        prediction = model.predict(fake_input).tolist()

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
