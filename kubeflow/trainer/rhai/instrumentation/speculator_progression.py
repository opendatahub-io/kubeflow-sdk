# Copyright 2025 The Kubeflow Authors.
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

"""Speculator progression tracking instrumentation injected into training pods.

This module contains the core instrumentation function that is extracted via
``inspect.getsource()`` and injected into generated training scripts. It is
NOT imported at runtime by the SDK — only its source code is used.
"""


def _create_speculator_progression_instrumentation(
    metrics_port: int,
    mode: str,
    num_epochs: int = 0,
) -> tuple:
    """Unified instrumentation for all speculator modes (extracted via inspect.getsource).

    Handles progression tracking for DATA_ONLY (file counting) and TRAIN_ONLY (log
    interception).

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into training scripts.

    Args:
        metrics_port: Port for HTTP metrics server.
        mode: Speculator mode string ("data_only", "train_only").
        num_epochs: Total training epochs (used for train_only).

    Returns:
        Tuple of (apply_fn, start_data_fn, handler_class) for testing purposes.
    """
    import http.server
    import json
    import logging
    import os
    import threading
    import time

    _hidden_states_dir: str | None = None
    _total_samples: int = 0
    _data_start_time: float | None = None

    _train_start_time: float | None = None
    _steps_per_epoch: int | None = None
    _max_step_in_epoch0 = 0
    _last_global_step = 0
    _last_epoch = 0
    _latest_metrics: dict = {}
    _metrics_lock = threading.Lock()
    _termination_message_written = False
    _current_phase: str | None = None
    _phase_floor_pct: int = 0
    _training_started = False

    class MetricsHandler(logging.Handler):
        """Captures speculators.metrics log records in memory."""

        def emit(self, record):
            nonlocal \
                _steps_per_epoch, \
                _max_step_in_epoch0, \
                _last_global_step, \
                _last_epoch, \
                _latest_metrics, \
                _training_started, \
                _train_start_time
            try:
                msg = record.msg
                if not isinstance(msg, dict):
                    return
                with _metrics_lock:
                    if not _training_started:
                        _training_started = True
                        _train_start_time = time.time()
                    _latest_metrics = msg
                    if "global_step" in msg:
                        _last_global_step = msg["global_step"]
                    if "epoch" in msg:
                        _last_epoch = msg["epoch"]
                    if (
                        _steps_per_epoch is None
                        and "train" in msg
                        and "epoch" in msg
                        and "global_step" in msg
                    ):
                        if msg["epoch"] == 0:
                            _max_step_in_epoch0 = max(_max_step_in_epoch0, msg["global_step"])
                        elif msg["epoch"] >= 1 and _max_step_in_epoch0 >= 0:
                            _steps_per_epoch = _max_step_in_epoch0 + 1
            except (KeyError, TypeError, ValueError) as e:
                print(f"[Kubeflow] Warning: Failed to parse metrics record: {e}", flush=True)

    class SpeculatorMetricsHTTPHandler(http.server.BaseHTTPRequestHandler):
        """HTTP handler that serves mode-aware progress to the controller."""

        def do_GET(self):
            try:
                transformed = self._get_progress()
            except Exception as e:
                print(f"[Kubeflow] Failed to create progress metrics payload: {e}", flush=True)
                self.send_error(500)
            else:
                self._maybe_write_termination_message(transformed)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(transformed, indent=2).encode())

        def _get_progress(self):
            if mode == "data_only":
                return self._data_progress(scale=100, offset=0)
            elif mode == "train_only":
                return self._training_progress(scale=100, offset=0)
            elif mode == "offline":
                if not _training_started:
                    return self._data_progress(scale=50, offset=0)
                else:
                    return self._training_progress(scale=50, offset=50)
            elif mode == "online":
                return self._training_progress(scale=100, offset=0)
            return self._empty_response()

        def _data_progress(self, scale, offset):
            if _hidden_states_dir is None or _total_samples <= 0:
                return self._empty_response()

            try:
                count = len(
                    [
                        f
                        for f in os.listdir(_hidden_states_dir)
                        if f.startswith("hs_") and f.endswith(".safetensors")
                    ]
                )
            except FileNotFoundError:
                count = 0

            raw_pct = count / _total_samples * 100
            progress_pct = min(offset + scale, offset + int(raw_pct * scale / 100))
            progress_pct = max(progress_pct, _phase_floor_pct)

            estimated_remaining = None
            if _data_start_time and count > 0:
                elapsed = time.time() - _data_start_time
                remaining = _total_samples - count
                if remaining <= 0:
                    estimated_remaining = 0
                else:
                    time_per_sample = elapsed / count
                    estimated_remaining = int(remaining * time_per_sample)

            return {
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": estimated_remaining,
                "currentStep": count,
                "totalSteps": _total_samples,
                "currentEpoch": None,
                "totalEpochs": None,
                "currentPhase": _current_phase,
                "trainMetrics": None,
                "evalMetrics": None,
            }

        def _training_progress(self, scale, offset):
            with _metrics_lock:
                metrics_snapshot = dict(_latest_metrics)

            if not metrics_snapshot:
                response = self._empty_response()
                response["progressPercentage"] = max(offset, _phase_floor_pct)
                return response

            global_step = metrics_snapshot.get("global_step", _last_global_step)
            epoch = metrics_snapshot.get("epoch", _last_epoch)
            train_metrics = metrics_snapshot.get("train", {})
            val_metrics = metrics_snapshot.get("val", {})

            total_steps = None
            progress_pct = offset
            estimated_remaining = None

            if _steps_per_epoch and _steps_per_epoch > 0:
                total_steps = _steps_per_epoch * num_epochs
                if total_steps > 0:
                    step_in_epoch = global_step % _steps_per_epoch
                    completed_steps = min(epoch * _steps_per_epoch + step_in_epoch + 1, total_steps)
                    raw_pct = completed_steps / total_steps * 100
                    progress_pct = min(offset + scale, offset + int(raw_pct * scale / 100))
                    progress_pct = max(progress_pct, _phase_floor_pct)

                    if _train_start_time and completed_steps > 0:
                        elapsed = time.time() - _train_start_time
                        remaining_steps = total_steps - completed_steps
                        if remaining_steps <= 0:
                            estimated_remaining = 0
                        else:
                            time_per_step = elapsed / completed_steps
                            estimated_remaining = int(remaining_steps * time_per_step)

            loss_val = train_metrics.get("loss")
            lr_val = metrics_snapshot.get("lr")

            return {
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": estimated_remaining,
                "currentStep": global_step,
                "totalSteps": total_steps,
                "currentEpoch": epoch + 1,
                "totalEpochs": num_epochs,
                "currentPhase": _current_phase,
                "trainMetrics": {
                    "loss": f"{loss_val:.4f}" if loss_val is not None else None,
                    "learning_rate": f"{lr_val:.6f}" if lr_val is not None else None,
                },
                "evalMetrics": {
                    k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                    for k, v in val_metrics.items()
                }
                if val_metrics
                else {},
            }

        def _empty_response(self):
            return {
                "progressPercentage": _phase_floor_pct if _phase_floor_pct > 0 else None,
                "estimatedRemainingSeconds": None,
                "currentStep": None,
                "totalSteps": None,
                "currentEpoch": None,
                "totalEpochs": None,
                "currentPhase": _current_phase,
                "trainMetrics": None,
                "evalMetrics": None,
            }

        def _maybe_write_termination_message(self, metrics):
            nonlocal _termination_message_written
            progress = metrics.get("progressPercentage")
            if progress is not None and progress >= 100:
                with _metrics_lock:
                    if _termination_message_written:
                        return
                    try:
                        with open("/dev/termination-log", "w") as f:
                            f.write(json.dumps(metrics))
                        _termination_message_written = True
                        print("[Kubeflow] Complete. Final metrics saved.", flush=True)
                    except (OSError, ValueError, TypeError) as e:
                        print(
                            f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                            f"Controller will fall back to HTTP polling.",
                            flush=True,
                        )

        def log_message(self, format, *args):
            pass

    def _start_data_progress_server(hidden_states_dir, total_samples):
        nonlocal _hidden_states_dir, _total_samples, _data_start_time
        _hidden_states_dir = hidden_states_dir
        _total_samples = total_samples
        _data_start_time = time.time()
        print(
            f"[Kubeflow] Data progress tracking active "
            f"({total_samples} samples in {hidden_states_dir})",
            flush=True,
        )

    def set_steps_per_epoch(steps):
        nonlocal _steps_per_epoch
        _steps_per_epoch = steps

    def _mark_data_complete():
        nonlocal _training_started, _train_start_time
        _training_started = True
        _train_start_time = time.time()

    def _set_phase(phase: str, floor_pct: int = 0):
        nonlocal _current_phase, _phase_floor_pct, _termination_message_written
        _current_phase = phase
        _phase_floor_pct = floor_pct
        if floor_pct >= 100:
            with _metrics_lock:
                if not _termination_message_written:
                    metrics = {
                        "progressPercentage": 100,
                        "estimatedRemainingSeconds": 0,
                        "currentPhase": phase,
                    }
                    try:
                        with open("/dev/termination-log", "w") as f:
                            f.write(json.dumps(metrics))
                        _termination_message_written = True
                        print("[Kubeflow] Complete. Final metrics saved.", flush=True)
                    except (OSError, ValueError, TypeError) as e:
                        print(
                            f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                            f"Controller will fall back to HTTP polling.",
                            flush=True,
                        )

    def apply_progression_tracking():
        if mode in ("train_only", "offline", "online"):
            handler = MetricsHandler()
            metrics_logger = logging.getLogger("speculators.metrics")
            metrics_logger.setLevel(logging.INFO)
            metrics_logger.addHandler(handler)

        try:
            server = http.server.HTTPServer(("0.0.0.0", metrics_port), SpeculatorMetricsHTTPHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print(f"[Kubeflow] Metrics server started on port {metrics_port}", flush=True)
        except OSError as e:
            print(
                f"[Kubeflow] Warning: Failed to start metrics server on port "
                f"{metrics_port}: {e}. Will continue without metrics server.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[Kubeflow] Warning: Unexpected error starting metrics server: {e}. "
                f"Will continue without metrics server.",
                flush=True,
            )

        return set_steps_per_epoch

    return (
        apply_progression_tracking,
        _start_data_progress_server,
        SpeculatorMetricsHTTPHandler,
        _mark_data_complete,
        _set_phase,
    )
