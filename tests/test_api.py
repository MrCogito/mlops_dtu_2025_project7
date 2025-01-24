import pytest
from httpx import AsyncClient
from app import main
from locust import HttpUser, task, between

# Unit tests using pytest
@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(app=main.app, base_url="http://testserver") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Flood Probability Prediction API!"}

@pytest.mark.asyncio
async def test_prediction_endpoint():
    async with AsyncClient(app=main.app, base_url="http://testserver") as client:
        payload = {
            "precipitation_sequence": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        response = await client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "flood_probability" in data
    assert 0 <= data["flood_probability"] <= 1

# Load testing with Locust
class FloodPredictionUser(HttpUser):
    wait_time = between(1, 5)  # Simulates a user waiting between 1-5 seconds between requests

    @task
    def load_test_root(self):
        response = self.client.get("/")
        assert response.status_code == 200

    @task
    def load_test_prediction(self):
        payload = {
            "precipitation_sequence": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "flood_probability" in data
        assert 0 <= data["flood_probability"] <= 1

# Instructions to run load testing with Locust:
# 1. Save this file as `test_api.py`.
# 2. Install Locust using `pip install locust`.
# 3. Run Locust with the command: `locust -f test_api.py`.
# 4. Open the web interface at http://localhost:8089, specify the number of users and spawn rate, and start the load test.
