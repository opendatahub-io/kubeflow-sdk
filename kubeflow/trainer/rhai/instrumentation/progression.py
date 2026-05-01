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

"""Progression tracking instrumentation for Transformers trainers."""


def create_progression_instrumentation(metrics_port: int) -> tuple:
    """Instrumentation code injected into training pods (extracted via inspect.getsource).

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into user training scripts. This approach
    provides syntax highlighting, testability, and type checking while avoiding
    runtime SDK dependencies.

    Args:
        metrics_port: Port for HTTP metrics server

    Returns:
        Tuple of (apply_fn, callback_class, handler_class) for testing purposes
    """
    from dataclasses import asdict, dataclass, field
    import http.server
    import json
    import threading
    import time
    from typing import Any

    from transformers import TrainerCallback, trainer as trainer_module

    @dataclass
    class ProgressionMetricsState:
        """Progression metrics state (camelCase for Kubernetes API compatibility)."""

        progressPercentage: int | None = None  # noqa: N815
        estimatedRemainingSeconds: int | None = None  # noqa: N815
        currentStep: int = 0  # noqa: N815
        totalSteps: int | None = None  # noqa: N815
        currentEpoch: float = 0.0  # noqa: N815  # Changed to float for precision (1.98 not 1)
        totalEpochs: int | None = None  # noqa: N815
        trainMetrics: dict[str, Any] = field(default_factory=dict)  # noqa: N815
        evalMetrics: dict[str, Any] = field(default_factory=dict)  # noqa: N815

    _progression_metrics_state = ProgressionMetricsState()
    _progression_metrics_lock = threading.Lock()

    def _update_progression_metrics(updates: dict) -> None:
        """Thread-safe metrics update."""
        with _progression_metrics_lock:
            for key, value in updates.items():
                if hasattr(_progression_metrics_state, key):
                    current_value = getattr(_progression_metrics_state, key)
                    if isinstance(value, dict) and isinstance(current_value, dict):
                        current_value.update(value)
                    else:
                        setattr(_progression_metrics_state, key, value)

    def _get_progression_metrics_json() -> str:
        """Get current metrics as JSON string."""
        with _progression_metrics_lock:
            return json.dumps(asdict(_progression_metrics_state))

    class ProgressionMetricsHandler(http.server.BaseHTTPRequestHandler):
        """HTTP server that exposes training progress metrics as JSON."""

        def log_message(self, format, *args):
            """Suppress HTTP server logging."""
            pass

        def do_GET(self):
            """Handle GET requests to expose metrics as JSON."""
            try:
                payload = _get_progression_metrics_json()
            except Exception as e:
                print(f"[Kubeflow] Failed to create progress metrics payload: {e}")
                self.send_error(500)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))

    class KubeflowProgressCallback(TrainerCallback):
        """Tracks training progress and updates metrics server."""

        def __init__(self, metrics_port: int = 28080):
            self.start_time: float | None = None
            self.metrics_port = metrics_port
            self.server: http.server.HTTPServer | None = None
            self.training_finished: bool = False

        def _update_progress_state(self, args, state) -> None:
            """Calculate and update progression state during training."""
            current_step = state.global_step if state.global_step is not None else 0
            total_steps = state.max_steps

            # Calculate progress percentage (always rounds down, e.g., 374/375 = 99%)
            progress_pct = int(current_step / total_steps * 100) if total_steps > 0 else 0

            # Estimate remaining time based on elapsed time and progress
            current_time = time.time()
            elapsed_sec = current_time - self.start_time if self.start_time else 0
            remaining_sec = None
            if total_steps > 0 and current_step > 0 and elapsed_sec > 0:
                # If training reached completion, set remaining time to 0
                if current_step >= total_steps:
                    remaining_sec = 0
                else:
                    estimated_total_time = elapsed_sec / (current_step / total_steps)
                    remaining_sec = max(0, int(estimated_total_time - elapsed_sec))

            # Calculate current epoch (keep float precision, e.g., 1.98)
            current_epoch = 0.0
            if hasattr(state, "epoch") and state.epoch:
                current_epoch = float(state.epoch)

            _update_progression_metrics(
                {
                    "currentStep": current_step,
                    "totalSteps": total_steps if total_steps > 0 else None,
                    "currentEpoch": current_epoch,
                    "progressPercentage": progress_pct,
                    "estimatedRemainingSeconds": remaining_sec,
                }
            )

        def on_train_begin(self, args, state, control, **kwargs) -> None:
            """Initialize progress tracking when training starts."""
            self.start_time = time.time()

            if state.is_world_process_zero and self.server is None:
                try:
                    server = http.server.HTTPServer(
                        ("0.0.0.0", self.metrics_port), ProgressionMetricsHandler
                    )
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    self.server = server
                    print(
                        f"[Kubeflow] Metrics server started on port {self.metrics_port}",
                        flush=True,
                    )
                except OSError as e:
                    print(
                        f"[Kubeflow] Warning: Failed to start metrics server on port "
                        f"{self.metrics_port}: {e}. Training will continue without metrics server.",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[Kubeflow] Warning: Unexpected error starting metrics server: {e}. "
                        f"Training will continue without metrics server.",
                        flush=True,
                    )

            # Calculate initial progress (handles checkpoint resume scenarios)
            initial_progress = 0
            current_step = state.global_step if state.global_step is not None else 0
            if state.max_steps > 0 and current_step > 0:
                initial_progress = int(current_step / state.max_steps * 100)

            _update_progression_metrics(
                {
                    "currentStep": current_step,
                    "totalSteps": state.max_steps if state.max_steps > 0 else None,
                    "currentEpoch": (
                        float(state.epoch) if hasattr(state, "epoch") and state.epoch else 0.0
                    ),
                    "totalEpochs": (
                        int(args.num_train_epochs) if hasattr(args, "num_train_epochs") else None
                    ),
                    "progressPercentage": initial_progress,
                }
            )

        def on_step_end(self, args, state, control, **kwargs) -> None:
            """Update progress after each training step."""
            # Don't overwrite completion state if training has already ended
            if not self.training_finished:
                self._update_progress_state(args, state)

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Categorize and track training/evaluation metrics."""
            if logs:
                train_metrics = {}
                eval_metrics = {}

                for key, value in logs.items():
                    metric_value = None
                    if isinstance(value, (int, float)):
                        metric_value = value
                    else:
                        # Try to extract scalar from torch tensor (graceful degradation)
                        try:
                            import torch  # type: ignore[import-not-found]

                            if isinstance(value, torch.Tensor) and value.numel() == 1:
                                metric_value = value.item()
                        except (ImportError, AttributeError):
                            pass

                    if metric_value is None:
                        continue

                    if key.startswith("eval_"):
                        eval_metrics[key] = metric_value
                    elif key in ["loss", "learning_rate", "grad_norm", "train_loss"]:
                        train_metrics[key] = metric_value
                    elif key == "train_samples_per_second":
                        train_metrics["throughput_samples_sec"] = metric_value

                update_dict = {}
                if train_metrics:
                    update_dict["trainMetrics"] = train_metrics
                if eval_metrics:
                    update_dict["evalMetrics"] = eval_metrics

                if update_dict:
                    _update_progression_metrics(update_dict)

        def on_train_end(self, args, state, control, **kwargs) -> None:
            """Update final progression state and write to termination message."""
            self.training_finished = True

            total_steps = state.max_steps if state.max_steps > 0 else None
            total_epochs = int(args.num_train_epochs) if hasattr(args, "num_train_epochs") else None

            # Calculate actual progress percentage (with safety checks)
            current_step = state.global_step if state.global_step is not None else 0
            progress_pct = (
                int(current_step / total_steps * 100)
                if total_steps and total_steps > 0 and current_step >= 0
                else 100
            )

            final_metrics = {
                "currentStep": current_step,
                "totalSteps": total_steps,
                "currentEpoch": (
                    float(state.epoch) if hasattr(state, "epoch") and state.epoch else 0.0
                ),
                "totalEpochs": total_epochs,
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": 0,
            }

            # Update HTTP server metrics
            _update_progression_metrics(final_metrics)

            # Write final metrics to termination message for controller capture
            if state.is_world_process_zero:
                try:
                    import json

                    # Hold lock during message construction and file write
                    # (to prevent race conditions)
                    with _progression_metrics_lock:
                        termination_message = {
                            "progressPercentage": progress_pct,
                            "estimatedRemainingSeconds": 0,
                            "currentStep": current_step,
                            "totalSteps": total_steps,
                            "currentEpoch": (
                                float(state.epoch)
                                if hasattr(state, "epoch") and state.epoch
                                else 0.0
                            ),
                            "totalEpochs": total_epochs,
                            "trainMetrics": dict(_progression_metrics_state.trainMetrics),
                            "evalMetrics": dict(_progression_metrics_state.evalMetrics),
                        }

                        with open("/dev/termination-log", "w") as f:
                            json.dump(termination_message, f)
                    print("[Kubeflow] Final metrics written to termination message", flush=True)
                except Exception as e:
                    print(
                        f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                        f"Controller will fall back to HTTP polling.",
                        flush=True,
                    )

    def apply_progression_tracking():
        """Patch Trainer.__init__ to inject KubeflowProgressCallback."""
        _original_init = trainer_module.Trainer.__init__

        def _instrumented_trainer_init(self, *args, **kwargs):
            result = _original_init(self, *args, **kwargs)
            callback = KubeflowProgressCallback(metrics_port)
            if callback not in self.callback_handler.callbacks:
                self.add_callback(callback)
            return result

        trainer_module.Trainer.__init__ = _instrumented_trainer_init

    # Return callback class and helper functions (helpers exposed for testing)
    return (
        apply_progression_tracking,
        KubeflowProgressCallback,
        ProgressionMetricsHandler,
        _get_progression_metrics_json,
        _update_progression_metrics,
    )
