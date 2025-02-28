"""Unit tests"""

from fastapi.testclient import TestClient
from app import app  # Import your FastAPI app

# This will create a TestClient instance to simulate requests
client = TestClient(app)


def test_health_check():
    """Test health check of the app."""
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok.",
        "Message": "The FastAPI app is up and running!",
    }


def test_answer():
    """Test the app's QnA logic."""
    query = "What is FastAPI?"
    method = "bart"
    response = client.get(f"/answer_with_no_summary?query={query}&method={method}")
    assert response.status_code == 200
    assert "Output of" in response.json().get("response", "")
