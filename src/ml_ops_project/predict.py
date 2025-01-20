import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image

from ml_ops_project.model import CarDamageModel


class DamagePrediction:
    """Class for making car damage predictions using a trained model."""

    DAMAGE_CLASSES = {0: "Crack", 1: "Scratch", 2: "Flat tire", 3: "Dent", 4: "Glass shatter", 5: "Lamp broken"}

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the prediction model.

        Args:
            model_path: Path to the model checkpoint. If None, use default or no model.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model_path = model_path or os.getenv("DAMAGE_MODEL_PATH")
        self.model = self._load_model() if self.model_path else None
        self.transform = self._get_transforms()
        if self.model:
            logging.info(f"Model loaded successfully on {self.device}")
        else:
            logging.warning("No model loaded; predictions will be random.")

    def _load_model(self) -> Optional[CarDamageModel]:
        """Load the model from checkpoint."""
        try:
            model = CarDamageModel.load_from_checkpoint(self.model_path)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_path}: {str(e)}")
            return None

    def _get_transforms(self) -> T.Compose:
        """Get image preprocessing transforms."""
        return T.Compose(
            [T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def _validate_image(self, image: Image.Image) -> None:
        """Validate the input image."""
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        if image.mode not in ["RGB", "RGBA"]:
            raise ValueError("Image must be in RGB or RGBA format")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a single image."""
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)

    def predict_single(self, image_input: Union[str, Path, Image.Image]) -> Dict:
        """Make prediction for a single image.

        Args:
            image_input: Path to the image file or a PIL Image object.
        Returns:
            Dictionary containing prediction results.
        """
        try:
            # Determine the type of image_input and load accordingly
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Invalid image input type")

            self._validate_image(image)

            # Preprocess the image
            tensor = self._preprocess_image(image)
            tensor = tensor.to(self.device)

            # Make prediction
            if self.model:
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
            else:
                # Random prediction placeholder in case no model is loaded
                predicted_class = torch.randint(0, len(self.DAMAGE_CLASSES), (1,)).item()
                confidence = 1.00  # Assume full confidence for the mock prediction

            return {
                "status": "success",
                "prediction": {
                    "class_id": predicted_class,
                    "class_name": self.DAMAGE_CLASSES[predicted_class],
                    "confidence": round(confidence * 100, 2),
                },
                "model_version": os.path.basename(self.model_path) if self.model_path else "no_model",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def predict_batch(self, image_inputs: List[Union[str, Path, Image.Image]]) -> List[Dict]:
        """Make predictions for a batch of images.

        Args:
            image_inputs: List of paths to image files or PIL Image objects.
        Returns:
            List of dictionaries containing prediction results.
        """
        return [self.predict_single(image_input) for image_input in image_inputs]


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
