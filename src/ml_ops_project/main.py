from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
from torchvision import transforms
from PIL import Image
from src.ml_ops_project.model import CarDamageModel

# Path to the model checkpoint
MODEL_PATH = os.path.join("models", "model.ckpt")

# Load the model
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Load the CarDamageModel from the checkpoint
    model = CarDamageModel.load_from_checkpoint(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    return transform(image).unsqueeze(0)  # Add batch dimension

# FastAPI app
app = FastAPI()
model = load_model()  # Load the model at startup

class ImagePathInput(BaseModel):
    image_path: str  # Path to the input image

@app.get("/")
def read_root():
    return {"message": "Car Damage Detection API"}

@app.post("/predict")
def predict(input_data: ImagePathInput):
    try:
        # Check if the image file exists
        if not os.path.isfile(input_data.image_path):
            raise FileNotFoundError(f"File not found: {input_data.image_path}")
        
        # Preprocess the image
        image_tensor = preprocess_image(input_data.image_path)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.argmax(outputs, dim=1).tolist()  # Get predicted class
        
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
