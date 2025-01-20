import logging
import os
import tempfile
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from ml_ops_project.predict import DamagePrediction

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DamageDetectionAPI:
    def __init__(self, model_path: str = None):
        """Initialize the API with model predictor."""
        self.app = FastAPI(title="Car Damage Detection API")
        try:
            self.predictor = DamagePrediction(model_path=model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

        self.register_routes()

    def register_routes(self):
        @self.app.get("/")
        def read_root():
            model_version = Path(self.predictor.model_path).stem if self.predictor.model_path else "no_model"
            return {
                "status": "healthy",
                "model_version": model_version,
                "damage_classes": self.predictor.DAMAGE_CLASSES,
            }

        @self.app.post("/predict")
        async def predict_damage(file: UploadFile = File(...)):
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            try:
                # Use a temporary file to save the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    contents = await file.read()
                    temp_file.write(contents)
                    temp_file.flush()
                    temp_file.seek(0)

                    logger.debug(f"File {file.filename} saved to temporary path {temp_file.name}")

                    # Pass the file path to predict_single
                    result = self.predictor.predict_single(temp_file.name)

                # Delete the temporary file after prediction
                os.unlink(temp_file.name)

                if result["status"] == "error":
                    raise HTTPException(status_code=500, detail=result["error"])

                return result
            except HTTPException as http_exc:
                raise http_exc
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

        @self.app.post("/predict/batch")
        async def predict_batch(files: List[UploadFile] = File(...)):
            results = []
            try:
                for file in files:
                    if not file.content_type.startswith("image/"):
                        raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
                    # Temporary file handling for batch process
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        contents = await file.read()
                        temp_file.write(contents)
                        temp_file.flush()
                        temp_file.seek(0)

                        logger.debug(f"File {file.filename} saved to temporary path {temp_file.name}")

                        # Pass the file path to predict_single
                        result = self.predictor.predict_single(temp_file.name)
                        results.append(result)

                    # Delete the temporary file after prediction
                    os.unlink(temp_file.name)

                return results
            except HTTPException as http_exc:
                raise http_exc
            except Exception as e:
                logger.error(f"Batch prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"An error occurred during batch prediction: {str(e)}")


def create_app(model_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    api = DamageDetectionAPI(model_path=model_path)
    return api.app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
