import pytest
from httpx import AsyncClient
import os

BASE_URL = "http://127.0.0.1:8000"
VALID_IMAGE_PATH = "data/example/val/images/1.jpg"
INVALID_IMAGE_PATH = "data/example/val/images/nonexistent.jpg"

@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root (/) endpoint."""
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI ML App!"}

@pytest.mark.asyncio
async def test_predict_endpoint_valid_input():
    """Test the /predict endpoint with a valid image path."""
    payload = {"image_path": VALID_IMAGE_PATH}
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

@pytest.mark.asyncio
async def test_predict_endpoint_invalid_input():
    """Test the /predict endpoint with an invalid image path."""
    payload = {"image_path": INVALID_IMAGE_PATH}
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 500
    assert "detail" in response.json()
    assert "File not found" in response.json()["detail"]
