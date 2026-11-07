import time
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import app as app_module
from app import app

def run_latency_benchmark(num_requests: int = 100):
    client = TestClient(app)

    # Use existing or mock model for benchmark
    if not os.path.exists(app_module.MODEL_PATH):
        print("Model file not found on disk; using calibrated mock model for latency benchmarking.")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.985, 0.015]])
        app_module.model = mock_model

    sample_transaction = [
        0.0, -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395,
        0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681,
        -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104,
        0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 149.62
    ]

    latencies = []
    print(f"Executing {num_requests} end-to-end API inference requests to /predict...")

    for i in range(num_requests):
        t0 = time.perf_counter()
        res = client.post("/predict", json={"features": sample_transaction})
        t1 = time.perf_counter()
        if res.status_code == 200:
            latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)
    print("\n" + "=" * 50)
    print("      REST API INFERENCE BENCHMARK REPORT         ")
    print("=" * 50)
    print(f"Total Requests Processed : {len(latencies)}")
    print(f"Average End-to-End Latency: {np.mean(latencies):.2f} ms")
    print(f"Median Latency (p50)     : {np.median(latencies):.2f} ms")
    print(f"95th Percentile (p95)    : {np.percentile(latencies, 95):.2f} ms")
    print(f"Min Latency              : {np.min(latencies):.2f} ms")
    print(f"Max Latency              : {np.max(latencies):.2f} ms")
    print(f"Estimated Throughput     : {1000 / np.mean(latencies):.1f} req/sec")
    print("=" * 50)

if __name__ == "__main__":
    run_latency_benchmark(100)
