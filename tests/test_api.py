import unittest
import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app as app_module
from app import app

class TestFraudAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Mock model so tests run instantly without needing to train a 284K-row model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0])
        self.mock_model.predict_proba.return_value = np.array([[0.985, 0.015]])
        app_module.model = self.mock_model

        self.valid_sample = [
            0.0, -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395,
            0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681,
            -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104,
            0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 149.62
        ]

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "Credit Card Fraud Detection API")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())

    def test_predict_success(self):
        response = self.client.post("/predict", json={"features": self.valid_sample})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["is_fraud"], 0)
        self.assertAlmostEqual(data["fraud_probability"], 0.015, places=3)
        self.assertEqual(data["risk_level"], "LOW")
        self.assertIn("inference_latency_ms", data)

    def test_predict_validation_error_wrong_length(self):
        # Only 5 features instead of 30 -> should return 422 Unprocessable Entity
        response = self.client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]})
        self.assertEqual(response.status_code, 422)

    def test_batch_prediction(self):
        batch = [{"features": self.valid_sample}, {"features": self.valid_sample}]
        self.mock_model.predict.return_value = np.array([0, 1])
        self.mock_model.predict_proba.return_value = np.array([[0.95, 0.05], [0.15, 0.85]])

        response = self.client.post("/predict/batch", json=batch)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["batch_size"], 2)
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][1]["risk_level"], "CRITICAL")

if __name__ == '__main__':
    unittest.main()
