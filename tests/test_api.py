import io
import logging
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from ml_ops_project.api import create_app

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def test_app():
    """Create a test instance of the FastAPI application."""
    try:
        app = create_app(model_path="models/model.ckpt")
        return TestClient(app)
    except Exception as e:
        logger.error(f"Failed to create test app: {str(e)}")
        raise

@pytest.fixture
def test_image():
    """Create a test image file."""
    def _create_test_image():
        # Create a test image with specific size
        image = Image.new('RGB', (224, 224), color='white')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    return _create_test_image

def test_read_root(test_app):
    """Test the root endpoint."""
    response = test_app.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_version" in response.json()
    assert "damage_classes" in response.json()

def test_predict_valid_image(test_app, test_image):
    """Test prediction with a valid image."""
    try:
        # Create and send image
        image_bytes = test_image()
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        response = test_app.post("/predict", files=files)
        
        # Log response details
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response content: {response.json()}")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "prediction" in data
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

def test_predict_invalid_file_type(test_app):
    """Test prediction with an invalid file type."""
    files = {"file": ("test.txt", io.BytesIO(b"invalid data"), "text/plain")}
    response = test_app.post("/predict", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()

def test_predict_batch_valid_images(test_app, test_image):
    """Test batch prediction with valid images."""
    try:
        # Create test files
        files = [
            ("files", ("test1.jpg", test_image(), "image/jpeg")),
            ("files", ("test2.jpg", test_image(), "image/jpeg"))
        ]
        
        response = test_app.post("/predict/batch", files=files)
        logger.debug(f"Batch response status: {response.status_code}")
        logger.debug(f"Batch response content: {response.json()}")
        
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert result["status"] == "success"
            assert "prediction" in result
            
    except Exception as e:
        logger.error(f"Batch test failed: {str(e)}")
        raise

def test_predict_batch_empty(test_app):
    """Test batch prediction with no files."""
    response = test_app.post("/predict/batch", files=[])
    assert response.status_code == 422

def test_predict_batch_mixed_files(test_app, test_image):
    """Test batch prediction with mixed valid and invalid files."""
    try:
        files = [
            ("files", ("test1.jpg", test_image(), "image/jpeg")),
            ("files", ("test.txt", io.BytesIO(b"not an image"), "text/plain"))
        ]
        
        response = test_app.post("/predict/batch", files=files)
        logger.debug(f"Mixed files response status: {response.status_code}")
        logger.debug(f"Mixed files response content: {response.json()}")
        
        assert response.status_code == 400
        assert "detail" in response.json()
        
    except Exception as e:
        logger.error(f"Mixed files test failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main(["-v", "--log-cli-level=DEBUG"])