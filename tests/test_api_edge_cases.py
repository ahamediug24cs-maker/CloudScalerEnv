from fastapi.testclient import TestClient

from src.app import app


client = TestClient(app)


def test_step_before_reset_returns_400() -> None:
    response = client.post("/step", json={"action_type": "do_nothing"})
    assert response.status_code == 400
    assert "Call reset() before step()" in response.json()["detail"]


def test_state_before_reset_returns_400() -> None:
    response = client.get("/state")
    assert response.status_code == 400
    assert "Call reset() before state()" in response.json()["detail"]


def test_reset_unknown_task_returns_400() -> None:
    response = client.post("/reset", json={"task_id": "does-not-exist"})
    assert response.status_code == 400
    assert "Unknown task_id" in response.json()["detail"]


def test_step_invalid_action_type_returns_422() -> None:
    client.post("/reset", json={"task_id": "easy-memory-leak"})
    response = client.post(
        "/step",
        json={"action_type": "explode", "service_name": "web-frontend", "count": 1},
    )
    assert response.status_code == 422


def test_step_missing_action_type_returns_422() -> None:
    client.post("/reset", json={"task_id": "easy-memory-leak"})
    response = client.post("/step", json={"service_name": "web-frontend", "count": 1})
    assert response.status_code == 422
