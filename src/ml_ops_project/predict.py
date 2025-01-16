import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torchvision.transforms as T
from PIL import Image
import logging
from ml_ops_project.model import CarDamageModel

class DamagePrediction:
    """Class for making car damage predictions using a trained model."""

    DAMAGE_CLASSES = {
        0: "Crack",
        1: "Scratch",
        2: "Flat tire",
        3: "Dent",
        4: "Glass shatter",
        5: "Lamp broken"
    }

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the prediction model.
        
        Args:
            model_path: Path to the model checkpoint. If None, will try to load from
                       DAMAGE_MODEL_PATH environment variable or use default path.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_path = model_path or os.getenv(
            "DAMAGE_MODEL_PATH",
            "models/model.ckpt"  # Default production model path
        )
        self.model = self._load_model()
        self.transform = self._get_transforms()
        logging.info(f"Model loaded successfully on {self.device}")

    def _load_model(self) -> CarDamageModel:
        """Load the model from checkpoint."""
        try:
            model = CarDamageModel.load_from_checkpoint(self.model_path)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

    def _get_transforms(self) -> T.Compose:
        """Get image preprocessing transforms."""
        return T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _validate_image(self, image: Image.Image) -> None:
        """Validate the input image."""
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        if image.mode not in ['RGB', 'RGBA']:
            raise ValueError("Image must be in RGB or RGBA format")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a single image."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0)

    def predict_single(self, image_path: Union[str, Path]) -> Dict:
        """Make prediction for a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary containing prediction results.
        """
        try:
            # Load and validate image
            image = Image.open(image_path)
            self._validate_image(image)
            
            # Preprocess
            tensor = self._preprocess_image(image)
            tensor = tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return {
                "status": "success",
                "prediction": {
                    "class_id": predicted_class,
                    "class_name": self.DAMAGE_CLASSES[predicted_class],
                    "confidence": round(confidence * 100, 2),
                },
                "model_version": os.path.basename(self.model_path)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """Make predictions for a batch of images.
        
        Args:
            image_paths: List of paths to image files.
            
        Returns:
            List of dictionaries containing prediction results.
        """
        return [self.predict_single(path) for path in image_paths]


def main():
    """Example usage of the prediction class."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict car damage from images")
    parser.add_argument("image_paths", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--model_path", help="Path to model checkpoint", default=None)
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DamagePrediction(model_path=args.model_path)
    
    # Make predictions
    if len(args.image_paths) == 1:
        result = predictor.predict_single(args.image_paths[0])
    else:
        result = predictor.predict_batch(args.image_paths)
    
    # Print results
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()