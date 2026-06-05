"""
tests/test_api.py — Full test suite for the Diabetes ML API.
"""

import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MODEL_ARTIFACTS_DIR", "./model_artifacts")
os.environ.setdefault("ENVIRONMENT", "development")

from app.main import app

client = TestClient(app)

VALID_INSTANCE = {
    "age":  0.038, "sex":  0.051, "bmi":  0.062,
    "bp":   0.022, "s1":  -0.044, "s2":  -0.035,
    "s3":  -0.043, "s4":  -0.003, "s5":   0.020, "s6": -0.018,
}


class TestHealth:
    def test_liveness(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in {"ok", "degraded"}
        assert "model_loaded" in r.json()

    def test_readiness(self):
        r = client.get("/ready")
        assert r.status_code in {200, 503}

    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()

    def test_model_info(self):
        r = client.get("/model-info")
        if r.status_code == 200:
            assert "model_id" in r.json()
            assert "feature_names" in r.json()

    def test_monitoring_snapshot(self):
        r = client.get("/monitoring")
        assert r.status_code == 200
        assert "total_predictions" in r.json()
        assert "uptime_seconds" in r.json()

    def test_drift_endpoint(self):
        r = client.get("/drift")
        assert r.status_code in {200, 503}

    def test_metrics_prometheus(self):
        r = client.get("/metrics")
        assert r.status_code == 200


class TestSinglePredict:
    def test_valid_single(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE]})
        assert r.status_code == 200
        body = r.json()
        assert len(body["predictions"]) == 1
        p = body["predictions"][0]
        assert "prediction" in p
        assert "prediction_lower" in p
        assert "prediction_upper" in p

    def test_prediction_in_valid_range(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE]})
        assert r.status_code == 200
        p = r.json()["predictions"][0]
        assert 0 < p["prediction"] < 500

    def test_prediction_interval_ordering(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE]})
        assert r.status_code == 200
        p = r.json()["predictions"][0]
        assert p["prediction_lower"] <= p["prediction"] <= p["prediction_upper"]

    def test_response_has_model_info(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE]})
        assert r.status_code == 200
        body = r.json()
        assert "model_id" in body
        assert "model_hash" in body
        assert body["latency_ms"] > 0

    def test_request_id_echoed(self):
        r = client.post("/predict", json={
            "instances": [VALID_INSTANCE], "request_id": "test-abc-123"})
        assert r.status_code == 200
        assert r.json()["request_id"] == "test-abc-123"

    def test_auto_request_id_generated(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE]})
        assert r.status_code == 200
        assert len(r.json()["request_id"]) > 0

    def test_request_id_header_propagated(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE]},
                        headers={"X-Request-ID": "header-id-xyz"})
        assert r.status_code == 200
        assert r.headers.get("X-Request-ID") == "header-id-xyz"


class TestBatchPredict:
    def test_batch_of_5(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE] * 5})
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 5

    def test_batch_of_50(self):
        r = client.post("/predict", json={"instances": [VALID_INSTANCE] * 50})
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 50

    def test_batch_preserves_order(self):
        hi = {**VALID_INSTANCE, "bmi": 0.17}
        lo = {**VALID_INSTANCE, "bmi": -0.09}
        r = client.post("/predict", json={"instances": [hi, lo]})
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert preds[0]["prediction"] > preds[1]["prediction"]


class TestValidation:
    def test_empty_instances_rejected(self):
        r = client.post("/predict", json={"instances": []})
        assert r.status_code == 422

    def test_missing_field_rejected(self):
        bad = {k: v for k, v in VALID_INSTANCE.items() if k != "bmi"}
        r = client.post("/predict", json={"instances": [bad]})
        assert r.status_code == 422

    def test_out_of_range_field_rejected(self):
        bad = {**VALID_INSTANCE, "bmi": 99.0}
        r = client.post("/predict", json={"instances": [bad]})
        assert r.status_code == 422

    def test_string_field_rejected(self):
        bad = {**VALID_INSTANCE, "age": "old"}
        r = client.post("/predict", json={"instances": [bad]})
        assert r.status_code == 422

    def test_null_field_rejected(self):
        bad = {**VALID_INSTANCE, "bp": None}
        r = client.post("/predict", json={"instances": [bad]})
        assert r.status_code == 422

    def test_extra_fields_ignored(self):
        extra = {**VALID_INSTANCE, "unknown_field": 999}
        r = client.post("/predict", json={"instances": [extra]})
        assert r.status_code == 200

    def test_wrong_http_method(self):
        r = client.get("/predict")
        assert r.status_code == 405


class TestEdgeCases:
    def test_all_zeros(self):
        r = client.post("/predict", json={"instances": [{k: 0.0 for k in VALID_INSTANCE}]})
        assert r.status_code == 200

    def test_extreme_values(self):
        hi = {"age": 0.11, "sex": 0.05, "bmi": 0.17, "bp": 0.13,
              "s1": 0.19, "s2": 0.19, "s3": 0.18, "s4": 0.18, "s5": 0.18, "s6": 0.13}
        r = client.post("/predict", json={"instances": [hi]})
        assert r.status_code == 200

    def test_deterministic_predictions(self):
        r1 = client.post("/predict", json={"instances": [VALID_INSTANCE]})
        r2 = client.post("/predict", json={"instances": [VALID_INSTANCE]})
        assert r1.status_code == 200
        assert (r1.json()["predictions"][0]["prediction"] ==
                r2.json()["predictions"][0]["prediction"])
