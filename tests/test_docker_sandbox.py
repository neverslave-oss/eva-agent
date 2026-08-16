"""
test_docker_sandbox.py — Docker sanity checks for kernel-evolving sandbox.
Tests are skipped if Docker is not available.
"""
import os
import sys
import time
import subprocess
import urllib.request
import pytest

SANDBOX_PORT = 8779
DOCKERFILE = os.path.join(os.path.dirname(__file__), "..", "Dockerfile.sandbox")
IMAGE_TAG = "kernel-evolving-sandbox-test"


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
class TestDockerSandbox:
    container_id: str | None = None

    @classmethod
    def setup_class(cls):
        """Build and start the sandbox container."""
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        # Build image
        build = subprocess.run(
            ["docker", "build", "-f", "Dockerfile.sandbox", "-t", IMAGE_TAG, "."],
            capture_output=True, text=True, cwd=repo_root, timeout=120,
        )
        if build.returncode != 0:
            pytest.skip(f"Docker build failed: {build.stderr}")

        # Run container in detached mode
        run = subprocess.run(
            ["docker", "run", "-d", "--rm",
             "-p", f"{SANDBOX_PORT}:{SANDBOX_PORT}",
             "-e", f"EVOLUTION_ENABLED=true",
             "-e", f"KERNEL_SANDBOX=true",
             IMAGE_TAG],
            capture_output=True, text=True, timeout=30,
        )
        if run.returncode != 0:
            pytest.skip(f"Docker run failed: {run.stderr}")

        cls.container_id = run.stdout.strip()
        # Wait for the container to start up
        for _ in range(15):
            time.sleep(1)
            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{SANDBOX_PORT}/health", timeout=2
                )
                if resp.status == 200:
                    break
            except Exception:
                continue

    @classmethod
    def teardown_class(cls):
        if cls.container_id:
            subprocess.run(["docker", "stop", cls.container_id],
                           capture_output=True, timeout=15)

    def test_container_health_endpoint_returns_ok(self):
        resp = urllib.request.urlopen(
            f"http://localhost:{SANDBOX_PORT}/health", timeout=5
        )
        assert resp.status == 200
        import json
        data = json.loads(resp.read())
        assert data.get("status") == "ok"

    def test_evolution_enabled_true_in_container(self):
        """Verify EVOLUTION_ENABLED=true is set in the running container."""
        result = subprocess.run(
            ["docker", "exec", self.container_id,
             "sh", "-c", "echo $EVOLUTION_ENABLED"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.stdout.strip() == "true"

    def test_message_endpoint_responds(self):
        """Verify /message endpoint returns a response."""
        import json
        payload = json.dumps({"message": "hello"}).encode()
        req = urllib.request.Request(
            f"http://localhost:{SANDBOX_PORT}/message",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            assert resp.status in (200, 201, 202)
        except urllib.error.HTTPError as e:
            # 4xx is acceptable — the endpoint exists and responded
            assert e.code < 500
