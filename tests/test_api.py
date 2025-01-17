import pytest
from httpx import AsyncClient

BASE_URL = "http://127.0.0.1:8000"
VALID_IMAGE_PATH = "data/example/val/images/1.jpg"
INVALID_IMAGE_PATH = "data/example/val/images/nonexistent.jpg"


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root (/) endpoint."""
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/")
    assert response.status_code == 200, "Root endpoint failed: Status code is not 200"
    assert response.json() == {"message": "Car Damage Detection API"}, "Root endpoint failed: Unexpected response JSON"
    print("✅ test_root_endpoint passed.")


@pytest.mark.asyncio
async def test_predict_endpoint_valid_input():
    """Test the /predict endpoint with a valid image path."""
    payload = {"image_path": VALID_IMAGE_PATH}
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 200, "Predict endpoint failed: Status code is not 200"
    assert "predictions" in response.json(), "Predict endpoint failed: No 'predictions' key in response"
    print("✅ test_predict_endpoint_valid_input passed.")


@pytest.mark.asyncio
async def test_predict_endpoint_invalid_input():
    """Test the /predict endpoint with an invalid image path."""
    payload = {"image_path": INVALID_IMAGE_PATH}
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 500, "Predict endpoint failed: Status code is not 500 for invalid input"
    assert "detail" in response.json(), "Predict endpoint failed: No 'detail' key in response"
    assert "File not found" in response.json()["detail"], "Predict endpoint failed: Error message not as expected"
    print("✅ test_predict_endpoint_invalid_input passed.")
