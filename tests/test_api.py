from fastapi.testclient import TestClient

from templateparity.api import app


def test_health_and_demo_routes():
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    demo = client.get("/api/demo").json()
    assert demo["synthetic"] is True
    assert "not measured inference" in demo["conclusion"]


def test_fixture_api_returns_validated_fixtures():
    response = TestClient(app).get("/api/fixtures")
    assert response.status_code == 200
    assert len(response.json()) >= 8


def test_experiment_api_rejects_unknown_id():
    assert TestClient(app).get("/api/experiments/../secrets").status_code in {400, 404, 422}
