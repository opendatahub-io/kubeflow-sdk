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
              Must use transformers.Trainer or trl.SFTTrainer internally.
        func_args: The arguments to pass to the function as kwargs.
        packages_to_install: A list of Python packages to install before running the function.
        pip_index_urls: The PyPI URLs from which to install Python packages.
                       The first URL will be the index-url, and remaining ones are extra-index-urls.
        num_nodes: The number of nodes to use for training.
        resources_per_node: The computing resources to allocate per node.
        env: The environment variables to set in the training nodes.
        enable_progression_tracking: Enable HTTP metrics server. Default: True.
        metrics_port: Port for HTTP metrics server. Default: 28080.
                     Valid range: 1024-65535 (non-privileged ports).
                     Ports 0-1023 are reserved and require root privileges.
                     This range is required for OpenShift restricted SCCs and Kubernetes
                     non-root security policies. Common safe ports: 8080-8999, 28000-29000.
        metrics_poll_interval_seconds: How often controller should poll metrics (seconds).
                                       Default: 30. Range: 5-300 (5s to 5min).
                                       Fast jobs: use 5-10s. Long jobs: use 60-120s.

    Raises:
        ValueError: If metrics_port is not in range 1024-65535.
        ValueError: If metrics_poll_interval_seconds is not in range 5-300.
        ValueError: If func is not callable.
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

    def __post_init__(self):
        """Validate configuration after initialization.

        Validation ensures compatibility with:
        - OpenShift restricted Security Context Constraints (SCCs)
        - Kubernetes non-root security policies
        - Standard container best practices
        """
        # Validate func is callable
        if not callable(self.func):
            raise ValueError(
                f"func must be callable, got {type(self.func).__name__}. "
                f"Please provide a training function."
            )

        # Validate metrics_port (must work with OpenShift restricted SCCs)
        if not isinstance(self.metrics_port, int):
            raise ValueError(
                f"metrics_port must be an integer, got {type(self.metrics_port).__name__}"
            )

        if self.metrics_port < 1024 or self.metrics_port > 65535:
            raise ValueError(
                f"metrics_port must be in range 1024-65535 (non-privileged ports), "
                f"got {self.metrics_port}. Ports 0-1023 are reserved and require root privileges. "
                f"This range (1024-65535) is required for OpenShift restricted SCCs and "
                f"Kubernetes non-root containers."
            )

        # Validate metrics_poll_interval_seconds
        if not isinstance(self.metrics_poll_interval_seconds, int):
            raise ValueError(
                f"metrics_poll_interval_seconds must be an integer, "
                f"got {type(self.metrics_poll_interval_seconds).__name__}"
            )

        if self.metrics_poll_interval_seconds < 5 or self.metrics_poll_interval_seconds > 300:
            raise ValueError(
                f"metrics_poll_interval_seconds must be in range 5-300, "
                f"got {self.metrics_poll_interval_seconds}"
            )


def _create_progression_instrumentation(metrics_port: int) -> tuple:
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
    from typing import Any, Optional

    from transformers import TrainerCallback, trainer as trainer_module

    @dataclass
    class ProgressionMetricsState:
        """Progression metrics state (camelCase for Kubernetes API compatibility)."""

        progressPercentage: Optional[int] = None  # noqa: N815
        estimatedRemainingSeconds: Optional[int] = None  # noqa: N815
        currentStep: int = 0  # noqa: N815
        totalSteps: Optional[int] = None  # noqa: N815
        currentEpoch: float = 0.0  # noqa: N815  # Changed to float for precision (1.98 not 1)
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
            self.start_time: Optional[float] = None
            self.metrics_port = metrics_port
            self.server: Optional[http.server.HTTPServer] = None
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


def get_transformers_instrumentation_wrapper(
    metrics_port: int,
) -> str:
    """Generate self-contained instrumentation wrapper via inspect.getsource.

    Extracts _create_progression_instrumentation as source code and injects a call
    with the provided metrics_port parameter.

    Args:
        metrics_port: Port for HTTP metrics server.

    Returns:
        Python code as string with {{user_func_import_and_call}} placeholder.
    """
    import inspect
    import textwrap

    # Extract the entire function source
    progression_instrumentation_code = inspect.getsource(_create_progression_instrumentation)
    progression_instrumentation_code = textwrap.dedent(progression_instrumentation_code)

    # Build the wrapper with function call
    wrapper = f"""# =============================================================================
# Kubeflow SDK - Progression Tracking Instrumentation
# Generated by kubeflow.trainer.rhai.transformers
# =============================================================================

print("[Kubeflow] Initializing progression tracking", flush=True)

# Instrumentation function definition
{progression_instrumentation_code}

# Initialize and apply instrumentation
(
    apply_progression_tracking,
    _,
    _,
    _,
    _,
) = _create_progression_instrumentation(metrics_port={metrics_port})
apply_progression_tracking()
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
    import inspect
    import os
    import textwrap

    from kubeflow.trainer.backends.kubernetes import utils
    from kubeflow.trainer.constants import constants

    # Ensure runtime trainer has a command set
    # This handles cases where RuntimeTrainer is created without calling set_command()
    try:
        _ = runtime.trainer.command
    except AttributeError:
        # Command not set, use default based on framework
        if runtime.trainer.framework == "pytorch":
            runtime.trainer.set_command(constants.TORCH_COMMAND)
        else:
            runtime.trainer.set_command(constants.DEFAULT_COMMAND)

    trainer_crd = models.TrainerV1alpha1Trainer()

    # Add number of nodes
    if trainer.num_nodes:
        trainer_crd.num_nodes = trainer.num_nodes

    # Add resources per node
    if trainer.resources_per_node:
        trainer_crd.resources_per_node = utils.get_resources_per_node(trainer.resources_per_node)

    # Add environment variables
    if trainer.env:
        trainer_crd.env = [
            models.IoK8sApiCoreV1EnvVar(name=key, value=value) for key, value in trainer.env.items()
        ]

    # Generate function code
    func_code = inspect.getsource(trainer.func)
    func_code = textwrap.dedent(func_code)

    # Generate function call (use **kwargs unpacking like utils.get_command_using_train_func)
    if trainer.func_args is None:
        func_call = f"{trainer.func.__name__}()"
    else:
        # Always unpack kwargs for training function calls
        func_call = f"{trainer.func.__name__}(**{trainer.func_args})"

    func_code = f"{func_code}\n{func_call}\n"

    # Wrap with progression tracking instrumentation if enabled
    if trainer.enable_progression_tracking:
        wrapper_code = get_transformers_instrumentation_wrapper(
            metrics_port=trainer.metrics_port,
        )
        func_code = wrapper_code.replace("{{user_func_import_and_call}}", func_code)

    # Build the command directly with the wrapped function code
    func_file = os.path.basename(inspect.getfile(trainer.func))

    is_mpi = runtime.trainer.command[0] == "mpirun"
    if is_mpi:
        func_file = os.path.join(constants.DEFAULT_MPI_USER_HOME, func_file)

    # Install Python packages if required
    install_packages = ""
    if trainer.packages_to_install:
        install_packages = utils.get_script_for_python_packages(
            trainer.packages_to_install,
            trainer.pip_index_urls,
            is_mpi,
        )

    # Build the trainer command with wrapped function code
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_packages:
                exec_script = install_packages + exec_script
            command.append(exec_script)
        else:
            command.append(c)

    trainer_crd.command = command

    return trainer_crd
