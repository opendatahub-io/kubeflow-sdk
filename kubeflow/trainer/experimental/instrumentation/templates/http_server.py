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

"""HTTP Metrics Server - Template for code injection.

This template provides a threadsafe HTTP server for exposing real-time training metrics.
"""

import http.server
import json
import socketserver
import threading


class ProgressMetricsHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that serves training progress metrics as JSON."""

    metrics = {
        "status": "initializing",
        "status_message": "Training initialization in progress",
        "status_details": {"last_event": "init", "last_event_time": None, "warnings": []},
        "progress": {"step_current": 0, "step_total": 0, "percent": 0.0, "epoch": 0},
        "time": {
            "started_sec": None,
            "started_at": None,
            "elapsed_sec": 0,
            "elapsed": "0s",
            "remaining_sec": None,
            "remaining": None,
            "updated_sec": None,
            "updated_at": None,
        },
        "metrics": {
            "loss": None,
            "learning_rate": None,
            "throughput_samples_sec": None,
        },
        "checkpoint": {
            "last_step": None,
            "last_path": None,
            "saved_at": None,
            "resumed_from": None,
        },
    }
    metrics_lock = threading.Lock()

    def log_message(self, format, *args):  # type: ignore
        """Suppress default HTTP server logging to avoid cluttering training logs."""
        pass

    def do_GET(self):  # type: ignore
        """Respond to GET requests with health check or metrics."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path in ("/", "/metrics"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            metrics_json = ProgressMetricsHandler.get_metrics_json()
            self.wfile.write(metrics_json.encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    @classmethod
    def update_metrics(cls, updates: dict):  # type: ignore
        """Thread-safe update of metrics dict.

        Args:
            updates: Dict of metrics to update (supports nested dicts).
        """
        with cls.metrics_lock:

            def deep_update(target, source):
                for key, value in source.items():
                    if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                        deep_update(target[key], value)
                    else:
                        target[key] = value

            deep_update(cls.metrics, updates)

    @classmethod
    def get_metrics_json(cls) -> str:
        """Thread-safe read of metrics as JSON string.

        Returns:
            JSON string of current metrics.
        """
        with cls.metrics_lock:
            return json.dumps(cls.metrics, indent=2)


def start_metrics_server(port: int = __PORT_PLACEHOLDER__):  # noqa: F821
    """Start HTTP server in background thread.

    Args:
        port: Port to listen on.

    Returns:
        Server object (has .shutdown() method for cleanup).
    """
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("", port), ProgressMetricsHandler)

    server_thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="metrics-server"
    )
    server_thread.start()

    print(f"[Kubeflow] Metrics server started on port {port}", flush=True)
    return server
