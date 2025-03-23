# Copyright 2025 Elizabeth Marcellina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
