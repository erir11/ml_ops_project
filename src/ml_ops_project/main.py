from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
from torchvision import transforms
from src.ml_ops_project.model import CarDamageModel

# Path to the model checkpoint
MODEL_PATH = "models/model.ckpt"

# Load the model
def load_model():
    try:
        # Load the CarDamageModel from the checkpoint
        model = CarDamageModel.load_from_checkpoint(MODEL_PATH, map_location=torch.device("cpu"))
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Preprocess the uploaded image
def preprocess_image(file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])
    image = Image.open(file).convert("RGB")  # Ensure 3 channels
    return transform(image).unsqueeze(0)  # Add batch dimension

# FastAPI app
app = FastAPI()
model = load_model()  # Load the model at startup

@app.get("/")
def read_root():
    return {"message": "Car Damage Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Preprocess the uploaded file
        image_tensor = preprocess_image(file.file)

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.argmax(outputs, dim=1).tolist()  # Get predicted class

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
