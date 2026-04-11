from fastapi.testclient import TestClient

from src.app import app


def test_reset_accepts_max_steps_override() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task_id": "easy-memory-leak", "max_steps": 30})
    assert response.status_code == 200
    data = response.json()
    assert data["max_steps"] == 30


def test_reset_accepts_service_override() -> None:
    client = TestClient(app)
    payload = {
        "task_id": "easy-memory-leak",
        "services": {
            "web-frontend": {
                "replicas": 4,
                "cpu_utilization": 70.0,
                "memory_utilization": 50.0,
                "status": "healthy",
            }
        },
    }
    response = client.post("/reset", json=payload)
    assert response.status_code == 200
    data = response.json()
    svc = data["services"]["web-frontend"]
    assert svc["replicas"] == 4
    assert svc["cpu_utilization"] == 70.0
    assert svc["memory_utilization"] == 50.0
