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

"""Progression tracking instrumentation templates for TransformersTrainer.

This module contains the actual implementation that is injected into training pods.
"""

import http.server
import json
import threading
import time
from typing import Any, Optional

# Transformers will be available in training pods where this code runs
try:
    from transformers import TrainerCallback
except ImportError:
    # Fallback for SDK development/testing without transformers installed
    TrainerCallback = object  # type: ignore[misc,assignment]


# Module-level state (encapsulated, not global class state)
_metrics_state: dict[str, Any] = {
    "progressPercentage": None,
    "estimatedRemainingSeconds": None,
    "currentStep": 0,
    "totalSteps": None,
    "currentEpoch": 0,
    "totalEpochs": None,
    "trainMetrics": {},
    "evalMetrics": {},
}
_metrics_lock = threading.Lock()


def _update_metrics(updates: dict) -> None:
    """Thread-safe metrics update."""
    with _metrics_lock:
        for key, value in updates.items():
            if isinstance(value, dict) and key in _metrics_state:
                existing_value = _metrics_state[key]
                if isinstance(existing_value, dict):
                    existing_value.update(value)
                else:
                    _metrics_state[key] = value
            else:
                _metrics_state[key] = value


def _get_metrics_json() -> str:
    """Get current metrics as JSON string."""
    with _metrics_lock:
        return json.dumps(_metrics_state)


class MetricsServer(http.server.BaseHTTPRequestHandler):
    """HTTP server that exposes training progress metrics as JSON."""

    def log_message(self, format, *args):
        """Suppress HTTP server logging."""
        pass

    def do_GET(self):
        """Handle GET requests for health check and metrics."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path in ("/", "/metrics"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(_get_metrics_json().encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class ProgressCallback(TrainerCallback):
    """Tracks training progress and updates metrics server."""

    def __init__(self, custom_metrics_map: dict, metrics_port: int = 28080):
        self.start_time: Optional[float] = None
        self.custom_metrics_map = custom_metrics_map
        self.metrics_port = metrics_port
        self.server = None

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Initialize progress tracking when training starts."""
        self.start_time = time.time()

        # Start metrics server only on rank-0 process
        if state.is_world_process_zero and self.server is None:
            self.server = start_server(self.metrics_port)
            print(f"[Kubeflow] Metrics server started on port {self.metrics_port}", flush=True)

        _update_metrics(
            {
                "currentStep": state.global_step,
                "totalSteps": state.max_steps if state.max_steps > 0 else None,
                "currentEpoch": (
                    int(state.epoch) if hasattr(state, "epoch") and state.epoch else 0
                ),
                "totalEpochs": (
                    int(args.num_train_epochs) if hasattr(args, "num_train_epochs") else None
                ),
                "progressPercentage": 0,
            }
        )

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Update progress after each training step."""
        current_time = time.time()
        current_step = state.global_step
        total_steps = state.max_steps

        progress_pct = int(current_step / total_steps * 100) if total_steps > 0 else None

        elapsed_sec = current_time - self.start_time if self.start_time else 0
        remaining_sec = None
        if total_steps > 0 and current_step > 0 and elapsed_sec > 0:
            estimated_total_time = elapsed_sec / (current_step / total_steps)
            remaining_sec = int(estimated_total_time - elapsed_sec)

        _update_metrics(
            {
                "currentStep": current_step,
                "totalSteps": total_steps if total_steps > 0 else None,
                "currentEpoch": (
                    int(state.epoch) if hasattr(state, "epoch") and state.epoch else 0
                ),
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": remaining_sec,
            }
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Categorize and track training/evaluation metrics."""
        if logs:
            train_metrics = {}
            eval_metrics = {}

            for key, value in logs.items():
                # Convert value to numeric if possible
                metric_value = None
                if isinstance(value, (int, float)):
                    metric_value = value
                else:
                    # Handle PyTorch tensors (common in training loops)
                    try:
                        import torch

                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            metric_value = value.item()
                    except (ImportError, AttributeError):
                        pass

                # Skip non-numeric values
                if metric_value is None:
                    continue

                # Categorize the metric
                if key.startswith("eval_"):
                    eval_metrics[key] = metric_value
                elif key in ["loss", "learning_rate", "grad_norm", "train_loss"]:
                    train_metrics[key] = metric_value
                elif key == "train_samples_per_second":
                    train_metrics["throughput_samples_sec"] = metric_value
                elif self.custom_metrics_map and key in self.custom_metrics_map:
                    metric_name = self.custom_metrics_map[key]
                    if metric_name.startswith("eval_"):
                        eval_metrics[metric_name] = metric_value
                    else:
                        train_metrics[metric_name] = metric_value

            if train_metrics or eval_metrics:
                update_dict = {}
                if train_metrics:
                    update_dict["trainMetrics"] = train_metrics
                if eval_metrics:
                    update_dict["evalMetrics"] = eval_metrics
                _update_metrics(update_dict)

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """Mark training as complete."""
        _update_metrics(
            {
                "progressPercentage": 100,
                "estimatedRemainingSeconds": 0,
            }
        )


def start_server(port: int):
    """Start HTTP metrics server in background thread."""
    import http.server
    import threading

    server = http.server.HTTPServer(("0.0.0.0", port), MetricsServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def enable_tracking(custom_metrics: dict, metrics_port: int = 28080):
    """Enable progression tracking by patching Trainer.__init__.

    Args:
        custom_metrics: Dictionary mapping log keys to metric names
        metrics_port: Port for HTTP metrics server (default: 28080)
    """
    from transformers import trainer as trainer_module

    _original_init = trainer_module.Trainer.__init__

    def _instrumented_trainer_init(self, *args, **kwargs):
        result = _original_init(self, *args, **kwargs)
        callback = ProgressCallback(custom_metrics, metrics_port)
        if callback not in self.callback_handler.callbacks:
            self.add_callback(callback)
        return result

    trainer_module.Trainer.__init__ = _instrumented_trainer_init
