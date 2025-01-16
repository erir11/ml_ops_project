import pytest
from httpx import AsyncClient

# Base URL of the API (change if running on a different host or port)
BASE_URL = "http://127.0.0.1:8000"

@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root (/) endpoint."""
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI ML App!"}

@pytest.mark.asyncio
async def test_predict_endpoint_valid_input():
    """Test the /predict endpoint with valid input."""
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}  # Example valid input
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

@pytest.mark.asyncio
async def test_predict_endpoint_invalid_input():
    """Test the /predict endpoint with invalid input."""
    payload = {"features": "invalid_data"}  # Example invalid input
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity for validation errors
