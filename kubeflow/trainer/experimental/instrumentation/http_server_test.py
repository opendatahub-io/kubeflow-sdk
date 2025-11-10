# Copyright 2024 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for HTTP metrics server."""

from http.client import HTTPConnection
import json
import threading
import time

import pytest

from kubeflow.trainer.experimental.instrumentation.http_server import get_http_server_code


@pytest.fixture
def metrics_server():
    """Start a real metrics server for testing."""
    # Execute the server code
    server_code = get_http_server_code()
    namespace = {}
    exec(compile(server_code, "<string>", "exec"), namespace)

    # Find an available port dynamically
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]

    server = namespace["start_metrics_server"](port=port)
    handler_class = namespace["ProgressMetricsHandler"]

    # Give server time to start
    time.sleep(0.1)

    yield {"server": server, "handler": handler_class, "port": port}

    # Cleanup
    server.shutdown()


def test_http_server_starts_and_responds(metrics_server):
    """Test that metrics server actually starts and responds."""
    port = metrics_server["port"]

    conn = HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", "/health")
    response = conn.getresponse()

    assert response.status == 200
    assert response.read() == b"OK"
    conn.close()


def test_metrics_endpoint_returns_json(metrics_server):
    """Test /metrics endpoint returns valid JSON."""
    port = metrics_server["port"]

    conn = HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", "/metrics")
    response = conn.getresponse()

    assert response.status == 200
    assert response.getheader("Content-Type") == "application/json"

    data = json.loads(response.read().decode("utf-8"))
    assert "status" in data
    assert "progress" in data
    assert "metrics" in data

    conn.close()


def test_metrics_contains_expected_fields(metrics_server):
    """Test metrics response has all expected fields."""
    port = metrics_server["port"]

    conn = HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", "/metrics")
    response = conn.getresponse()
    data = json.loads(response.read().decode("utf-8"))

    # Check top-level keys
    assert "status" in data
    assert "status_message" in data
    assert "status_details" in data
    assert "progress" in data
    assert "time" in data
    assert "metrics" in data
    assert "checkpoint" in data

    # Check nested structure
    assert "step_current" in data["progress"]
    assert "elapsed_sec" in data["time"]
    assert "loss" in data["metrics"]

    conn.close()


def test_update_metrics_thread_safe(metrics_server):
    """Test that update_metrics is thread-safe."""
    handler = metrics_server["handler"]
    port = metrics_server["port"]

    def update_worker(value):
        for i in range(100):
            handler.update_metrics({"progress": {"step_current": value * 100 + i}})

    # Spawn multiple threads updating metrics
    threads = [threading.Thread(target=update_worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Read metrics - should not crash or have corrupted data
    conn = HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", "/metrics")
    response = conn.getresponse()
    data = json.loads(response.read().decode("utf-8"))

    # Should have a valid step value
    assert isinstance(data["progress"]["step_current"], int)
    assert data["progress"]["step_current"] >= 0

    conn.close()


def test_concurrent_reads_and_writes(metrics_server):
    """Test concurrent reads and writes don't cause issues."""
    handler = metrics_server["handler"]
    port = metrics_server["port"]
    errors = []

    def reader():
        try:
            for _ in range(50):
                conn = HTTPConnection("localhost", port, timeout=5)
                conn.request("GET", "/metrics")
                response = conn.getresponse()
                json.loads(response.read().decode("utf-8"))
                conn.close()
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def writer():
        try:
            for i in range(50):
                handler.update_metrics(
                    {"progress": {"step_current": i}, "metrics": {"loss": float(i) * 0.1}}
                )
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    # Run readers and writers concurrently
    threads = [
        threading.Thread(target=reader),
        threading.Thread(target=reader),
        threading.Thread(target=writer),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have no errors
    assert len(errors) == 0


def test_get_metrics_json_returns_consistent_snapshot(metrics_server):
    """Test that get_metrics_json returns consistent data."""
    handler = metrics_server["handler"]

    # Update multiple fields atomically
    handler.update_metrics(
        {
            "progress": {"step_current": 100, "step_total": 1000},
            "metrics": {"loss": 0.5, "learning_rate": 0.001},
        }
    )

    # Get JSON twice - should be identical
    json1 = handler.get_metrics_json()
    json2 = handler.get_metrics_json()

    assert json1 == json2

    # Parse and verify
    data = json.loads(json1)
    assert data["progress"]["step_current"] == 100
    assert data["metrics"]["loss"] == 0.5


def test_nested_dict_updates_preserve_other_fields(metrics_server):
    """Test that nested updates don't overwrite unrelated fields."""
    handler = metrics_server["handler"]

    # Set initial values
    handler.update_metrics({"progress": {"step_current": 50, "step_total": 100}})

    # Update only step_current
    handler.update_metrics({"progress": {"step_current": 75}})

    # step_total should still be 100
    metrics_json = handler.get_metrics_json()
    data = json.loads(metrics_json)
    assert data["progress"]["step_current"] == 75
    assert data["progress"]["step_total"] == 100


def test_404_on_unknown_path(metrics_server):
    """Test that unknown paths return 404."""
    port = metrics_server["port"]

    conn = HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", "/unknown")
    response = conn.getresponse()

    assert response.status == 404
    conn.close()


def test_root_path_returns_metrics(metrics_server):
    """Test that root path '/' returns metrics."""
    port = metrics_server["port"]

    conn = HTTPConnection("localhost", port, timeout=5)
    conn.request("GET", "/")
    response = conn.getresponse()

    assert response.status == 200
    data = json.loads(response.read().decode("utf-8"))
    assert "status" in data

    conn.close()


def test_update_metrics_with_new_keys(metrics_server):
    """Test that update_metrics can add new keys."""
    handler = metrics_server["handler"]

    # Add a completely new top-level key
    handler.update_metrics({"custom_field": "custom_value"})

    metrics_json = handler.get_metrics_json()
    data = json.loads(metrics_json)
    assert "custom_field" in data
    assert data["custom_field"] == "custom_value"


def test_update_metrics_with_none_values(metrics_server):
    """Test that None values are handled correctly."""
    handler = metrics_server["handler"]

    handler.update_metrics({"metrics": {"loss": None, "learning_rate": 0.001}})

    metrics_json = handler.get_metrics_json()
    data = json.loads(metrics_json)
    assert data["metrics"]["loss"] is None
    assert data["metrics"]["learning_rate"] == 0.001
