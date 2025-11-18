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

"""TransformersTrainer for HuggingFace Transformers and TRL with auto-instrumentation."""

from dataclasses import dataclass, field
from typing import Callable, Optional

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


@dataclass
class TransformersTrainer:
    """RHAI trainer for HuggingFace Transformers and TRL with auto-instrumentation.

    Args:
        func: The function that encapsulates the entire model training process.
        func_args: The arguments to pass to the function.
        packages_to_install: A list of Python packages to install before running the function.
        pip_index_urls: The PyPI URLs from which to install Python packages.
                       The first URL will be the index-url, and remaining ones are extra-index-urls.
        num_nodes: The number of nodes to use for training.
        resources_per_node: The computing resources to allocate per node.
        env: The environment variables to set in the training nodes.
        enable_progression_tracking: Enable HTTP metrics server. Default: True.
        metrics_port: Port for HTTP metrics server. Default: 28080.
        metrics_poll_interval_seconds: How often controller should poll metrics (seconds).
                                       Default: 30. Lower values = more frequent updates,
                                       higher controller load. Recommended range: 5-60 seconds.
    """

    # Core training function (same as CustomTrainer)
    func: Callable
    func_args: Optional[dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    env: Optional[dict[str, str]] = None

    # Instrumentation features
    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30


def _create_progression_instrumentation(metrics_port: int):
    """Instrumentation code injected into training pods (extracted via inspect.getsource).

    This function is NOT called directly - it's extracted as source code and injected
    into training scripts, providing syntax highlighting and testability while avoiding
    runtime SDK dependencies.

    Args:
        metrics_port: Port for HTTP metrics server
    """
    from dataclasses import asdict, dataclass, field
    import http.server
    import json
    import threading
    import time
    from typing import Any, Optional

    from transformers import TrainerCallback, trainer as trainer_module

    @dataclass
    class ProgressionMetricsState:
        """Progression metrics state (camelCase for Kubernetes API compatibility)."""

        progressPercentage: Optional[int] = None  # noqa: N815
        estimatedRemainingSeconds: Optional[int] = None  # noqa: N815
        currentStep: int = 0  # noqa: N815
        totalSteps: Optional[int] = None  # noqa: N815
        currentEpoch: int = 0  # noqa: N815
        totalEpochs: Optional[int] = None  # noqa: N815
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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(_get_progression_metrics_json().encode("utf-8"))

    class KubeflowProgressCallback(TrainerCallback):
        """Tracks training progress and updates metrics server."""

        def __init__(self, metrics_port: int = 28080):
            self.start_time: Optional[float] = None
            self.metrics_port = metrics_port
            self.server: Optional[http.server.HTTPServer] = None

        def _update_progress_state(self, args, state, is_final: bool = False) -> None:
            """Calculate and update progression state (shared logic for step_end and train_end)."""
            current_step = state.global_step
            total_steps = state.max_steps

            progress_pct = int(current_step / total_steps * 100) if total_steps > 0 else None
            if is_final and progress_pct is None:
                progress_pct = 100  # Default to 100% if total_steps unknown at end

            remaining_sec = None
            if not is_final:
                current_time = time.time()
                elapsed_sec = current_time - self.start_time if self.start_time else 0
                if total_steps > 0 and current_step > 0 and elapsed_sec > 0:
                    estimated_total_time = elapsed_sec / (current_step / total_steps)
                    remaining_sec = int(estimated_total_time - elapsed_sec)

            _update_progression_metrics(
                {
                    "currentStep": current_step,
                    "totalSteps": total_steps if total_steps > 0 else None,
                    "currentEpoch": (
                        int(state.epoch) if hasattr(state, "epoch") and state.epoch else 0
                    ),
                    "progressPercentage": progress_pct,
                    "estimatedRemainingSeconds": 0 if is_final else remaining_sec,
                }
            )

        def on_train_begin(self, args, state, control, **kwargs) -> None:
            """Initialize progress tracking when training starts."""
            self.start_time = time.time()

            if state.is_world_process_zero and self.server is None:
                server = http.server.HTTPServer(
                    ("0.0.0.0", self.metrics_port), ProgressionMetricsHandler
                )
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                self.server = server
                print(f"[Kubeflow] Metrics server started on port {self.metrics_port}", flush=True)

            _update_progression_metrics(
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
            self._update_progress_state(args, state, is_final=False)

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
                        try:
                            import torch

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

                if train_metrics or eval_metrics:
                    update_dict = {}
                    if train_metrics:
                        update_dict["trainMetrics"] = train_metrics
                    if eval_metrics:
                        update_dict["evalMetrics"] = eval_metrics
                    _update_progression_metrics(update_dict)

        def on_train_end(self, args, state, control, **kwargs) -> None:
            """Update final progression state based on actual training completion state."""
            self._update_progress_state(args, state, is_final=True)

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

    return apply_progression_tracking, KubeflowProgressCallback, ProgressionMetricsHandler


def get_transformers_instrumentation_wrapper(
    metrics_port: int,
) -> str:
    """Generate self-contained instrumentation wrapper via inspect.getsource.

    Extracts _create_progression_instrumentation as source code.

    Args:
        metrics_port: Port for HTTP metrics server.

    Returns:
        Python code as string with {{user_func_import_and_call}} placeholder.
    """
    import inspect
    import textwrap

    instrumentation_code = inspect.getsource(_create_progression_instrumentation)
    instrumentation_code = textwrap.dedent(instrumentation_code)

    lines = instrumentation_code.split("\n")
    code_start = 0
    code_end = len(lines)
    in_docstring = False
    for i, line in enumerate(lines):
        if '"""' in line:
            if not in_docstring:
                in_docstring = True
            else:
                code_start = i + 1
                break
        elif line.strip().startswith("def "):
            code_start = i + 1

    for i in range(len(lines) - 1, code_start - 1, -1):
        if lines[i].strip().startswith("return "):
            code_end = i
            break

    instrumentation_body = "\n".join(lines[code_start:code_end])
    instrumentation_body = textwrap.dedent(instrumentation_body)

    wrapper = f"""# =============================================================================
# Kubeflow SDK - Progression Tracking Instrumentation
# Generated by kubeflow.trainer.rhai.transformers
# Self-contained (no SDK dependency at runtime)
# =============================================================================

print("[Kubeflow] Initializing progression tracking", flush=True)

# Injected instrumentation code
metrics_port = {metrics_port}

{instrumentation_body}

# Apply instrumentation before user code runs
apply_progression_tracking_fn, _, _ = apply_progression_tracking, None, None
apply_progression_tracking_fn()
print("[Kubeflow] Progression tracking enabled", flush=True)

# =============================================================================
# USER TRAINING CODE
# =============================================================================

{{{{user_func_import_and_call}}}}"""

    return wrapper


def get_trainer_cr_from_transformers_trainer(
    runtime: types.Runtime,
    trainer: TransformersTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD for TransformersTrainer with optional progression tracking.

    Args:
        runtime: Training runtime configuration
        trainer: TransformersTrainer instance
        initializer: Optional dataset/model initializer

    Returns:
        Trainer CRD with wrapped training function and annotations
    """
    from kubeflow.trainer.backends.kubernetes import utils
    from kubeflow.trainer.rhai.constants import (
        ANNOTATION_FRAMEWORK,
        ANNOTATION_METRICS_POLL_INTERVAL,
        ANNOTATION_METRICS_PORT,
        ANNOTATION_PROGRESSION_TRACKING,
    )
    from kubeflow.trainer.types import types

    # NOTE: SDK not needed at runtime - instrumentation code is self-contained
    base_trainer = types.CustomTrainer(
        func=trainer.func,
        func_args=trainer.func_args,
        packages_to_install=trainer.packages_to_install,
        pip_index_urls=trainer.pip_index_urls,
        num_nodes=trainer.num_nodes,
        resources_per_node=trainer.resources_per_node,
        env=trainer.env,
    )

    trainer_crd = utils.get_trainer_cr_from_custom_trainer(runtime, base_trainer)

    if trainer.enable_progression_tracking:
        wrapper_code = get_transformers_instrumentation_wrapper(
            metrics_port=trainer.metrics_port,
        )

        wrapped_code = wrapper_code.replace("{{user_func_import_and_call}}", "{func_code}")
        trainer_crd.command = [
            cmd.replace("{func_code}", wrapped_code) if "{func_code}" in cmd else cmd
            for cmd in trainer_crd.command
        ]

        if trainer_crd.annotations is None:
            trainer_crd.annotations = {}

        trainer_crd.annotations.update(
            {
                ANNOTATION_PROGRESSION_TRACKING: "true",
                ANNOTATION_METRICS_PORT: str(trainer.metrics_port),
                ANNOTATION_METRICS_POLL_INTERVAL: str(trainer.metrics_poll_interval_seconds),
                ANNOTATION_FRAMEWORK: "transformers",
            }
        )

    return trainer_crd
