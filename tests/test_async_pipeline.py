"""
test_async_pipeline.py — ADR-012 tests

Verifies:
- POST /replica/pipeline/async creates job with status=queued
- GET /replica/pipeline/{job_id} returns current state
- Background task updates status to complete
- Old jobs pruned after TTL
- DELETE marks job cancelled
"""
import sys
import os
import asyncio
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_test_app(cfg=None):
    """Create a test FastAPI app instance with minimal config."""
    import api
    api._cfg = cfg or {
        "pipeline": {"job_ttl_s": 3600, "async_enabled": True},
        "api": {"port": 8779},
    }
    return api.app


def test_async_pipeline_queues_job(monkeypatch):
    """POST /replica/pipeline/async returns job_id with status=queued."""
    from fastapi.testclient import TestClient
    import api

    api._cfg = {"pipeline": {"job_ttl_s": 3600}, "api": {"port": 8779}}
    api._pipeline_jobs.clear()

    # Mock the background task execution to be a no-op
    async def noop_run_pipeline_job(job_id, body):
        pass
    monkeypatch.setattr(api, "_run_pipeline_job", noop_run_pipeline_job)

    client = TestClient(api.app, raise_server_exceptions=False)
    payload = {
        "pipeline": "test",
        "stages": [
            {"name": "writer", "role": "custom", "brief": "Write", "task": "Do task"},
        ]
    }
    resp = client.post("/replica/pipeline/async", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    api._pipeline_jobs.clear()


def test_async_pipeline_status_polling(monkeypatch):
    """GET /replica/pipeline/{job_id} returns correct job state."""
    from fastapi.testclient import TestClient
    import api

    api._cfg = {"pipeline": {"job_ttl_s": 3600}, "api": {"port": 8779}}
    api._pipeline_jobs.clear()

    job_id = str(uuid.uuid4())
    api._pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "stage": "writer",
        "results": {},
        "error": None,
        "pipeline": "test",
        "created_at": time.time(),
        "completed_at": None,
    }

    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.get(f"/replica/pipeline/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["stage"] == "writer"

    api._pipeline_jobs.clear()


def test_async_pipeline_complete():
    """GET shows complete + full results after job finishes."""
    from fastapi.testclient import TestClient
    import api

    api._cfg = {"pipeline": {"job_ttl_s": 3600}, "api": {"port": 8779}}
    api._pipeline_jobs.clear()

    job_id = str(uuid.uuid4())
    api._pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "complete",
        "stage": "critic",
        "results": {"writer": "skill content", "critic": "PASS"},
        "error": None,
        "pipeline": "eval",
        "created_at": time.time() - 10,
        "completed_at": time.time(),
    }

    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.get(f"/replica/pipeline/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"
    assert "writer" in data["results"]
    assert data["results"]["critic"] == "PASS"

    api._pipeline_jobs.clear()


def test_async_pipeline_prune():
    """Old jobs are removed after TTL."""
    import api

    api._cfg = {"pipeline": {"job_ttl_s": 10}, "api": {"port": 8779}}
    api._pipeline_jobs.clear()

    old_id = str(uuid.uuid4())
    new_id = str(uuid.uuid4())

    # Old job: completed 1 hour ago
    api._pipeline_jobs[old_id] = {
        "job_id": old_id, "status": "complete",
        "completed_at": time.time() - 3600,
        "created_at": time.time() - 3700,
        "results": {}, "error": None, "pipeline": "old", "stage": None,
    }
    # New job: completed recently
    api._pipeline_jobs[new_id] = {
        "job_id": new_id, "status": "complete",
        "completed_at": time.time() - 5,
        "created_at": time.time() - 60,
        "results": {}, "error": None, "pipeline": "new", "stage": None,
    }

    api._prune_pipeline_jobs()

    assert old_id not in api._pipeline_jobs
    assert new_id in api._pipeline_jobs
    api._pipeline_jobs.clear()


def test_async_pipeline_cancel():
    """DELETE marks job status=cancelled."""
    from fastapi.testclient import TestClient
    import api

    api._cfg = {"pipeline": {"job_ttl_s": 3600}, "api": {"port": 8779}}
    api._pipeline_jobs.clear()

    job_id = str(uuid.uuid4())
    api._pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "stage": "writer",
        "results": {},
        "error": None,
        "pipeline": "test",
        "created_at": time.time(),
        "completed_at": None,
    }

    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.delete(f"/replica/pipeline/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"
    assert api._pipeline_jobs[job_id]["status"] == "cancelled"

    api._pipeline_jobs.clear()


def test_async_pipeline_not_found():
    """GET unknown job returns 404."""
    from fastapi.testclient import TestClient
    import api

    api._cfg = {"pipeline": {"job_ttl_s": 3600}, "api": {"port": 8779}}
    api._pipeline_jobs.clear()

    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.get("/replica/pipeline/nonexistent-job-id")
    assert resp.status_code == 404
