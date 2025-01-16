from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Optional
import os
from PIL import Image

# Paths
MODEL_PATH = os.path.join("models", "model.pkl")

# Mock model for test mode
class MockModel:
    def predict(self, X):
        return [0] * len(X)  # Always predicts 0

# Load the actual model
def load_model(test_mode: bool):
    if test_mode:
        return MockModel()
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

# Define input schema
class ImagePathInput(BaseModel):
    image_path: str  # Path to the input image

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI ML App!"}

@app.post("/predict")
def predict(input_data: ImagePathInput, test_mode: Optional[bool] = Query(False)):
    """
    Predict endpoint with optional test mode.
    If test_mode=True, the endpoint will use a mock model.
    """
    try:
        # Load the model (real or mock based on test_mode)
        model = load_model(test_mode=test_mode)

        # Validate image path
        if not os.path.isfile(input_data.image_path):
            raise FileNotFoundError(f"File not found: {input_data.image_path}")

        # Load and preprocess image
        image = Image.open(input_data.image_path).convert("RGB")
        image_array = np.array(image).flatten().reshape(1, -1)  # Flatten and reshape for the model

        # Make prediction
        prediction = model.predict(image_array).tolist()
        return {"prediction": prediction, "test_mode": test_mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
