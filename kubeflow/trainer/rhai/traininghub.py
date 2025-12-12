from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect
import os
import textwrap
from typing import Optional, Union

from kubeflow_trainer_api import models

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

# Training Hub specific file names and patterns
# These are the JSONL metrics files written by Training Hub backends
TRAININGHUB_SFT_METRICS_FILE_PATTERN = "training_params_and_metrics_global*.jsonl"
TRAININGHUB_SFT_METRICS_FILE_RANK0 = "training_params_and_metrics_global0.jsonl"
TRAININGHUB_OSFT_METRICS_FILE_PATTERN = "training_metrics_*.jsonl"
TRAININGHUB_OSFT_METRICS_FILE_RANK0 = "training_metrics_0.jsonl"
TRAININGHUB_OSFT_CONFIG_FILE = "training_params.json"


class TrainingHubAlgorithms(Enum):
    """Algorithm for TrainingHub Trainer."""

    SFT = "sft"
    OSFT = "osft"
    LORA_SFT = "lora_sft"


@dataclass
class TrainingHubTrainer:
    """TrainingHub RHAI trainer configuration.

    Args:
        func: Optional user-defined training function. If None, uses algorithm wrapper mode.
        func_args: Arguments to pass to the training function or algorithm.
            Note: `nnodes` and `nproc_per_node` can be specified here to control
            distributed training topology (maps to numNodes and numProcPerNode).
        packages_to_install: Python packages to install before training.
        pip_index_urls: PyPI index URLs for package installation.
        resources_per_node: The computing resources to allocate per node.
            Example: {"cpu": 4, "memory": "16Gi", "nvidia.com/gpu": 2}
        env: Environment variables to set in training pods.
        algorithm: Training Hub algorithm (SFT or OSFT). Required when func is None.
        enable_progression_tracking: Enable file-based progress tracking with HTTP server.
        metrics_port: HTTP server port for metrics endpoint.
        metrics_poll_interval_seconds: How often controller polls metrics endpoint.
    """

    func: Optional[Callable] = None
    func_args: Optional[dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: Optional[dict[str, str]] = None
    algorithm: Optional[TrainingHubAlgorithms] = None
    resources_per_node: Optional[dict] = None

    # Progress tracking parameters
    enable_progression_tracking: bool = True  # Enabled by default
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate metrics_port
        if not isinstance(self.metrics_port, int):
            raise ValueError(
                f"metrics_port must be an integer, got {type(self.metrics_port).__name__}"
            )

        if self.metrics_port < 1024 or self.metrics_port > 65535:
            raise ValueError(f"metrics_port must be in range 1024-65535, got {self.metrics_port}")

        # Validate metrics_poll_interval_seconds
        if not isinstance(self.metrics_poll_interval_seconds, int):
            raise ValueError(
                f"metrics_poll_interval_seconds must be an integer, "
                f"got {type(self.metrics_poll_interval_seconds).__name__}"
            )

        if self.metrics_poll_interval_seconds < 5 or self.metrics_poll_interval_seconds > 300:
            raise ValueError(
                f"metrics_poll_interval_seconds must be in range 5-300 seconds, "
                f"got {self.metrics_poll_interval_seconds}"
            )


def _derive_topology_from_func_args(
    func_args: Optional[dict],
) -> tuple[Optional[int], Optional[Union[int, str]]]:
    """Return (nnodes, nproc_per_node) based on provided func_args.

    If values are not provided in func_args, they are left as None so that the
    TrainingRuntime ML policy can supply appropriate defaults instead of the SDK.
    """
    nnodes: Optional[int] = None
    nproc_per_node: Optional[Union[int, str]] = None
    if isinstance(func_args, dict):
        nnodes_value = func_args.get("nnodes")
        if isinstance(nnodes_value, int):
            nnodes = nnodes_value
        npp_value = func_args.get("nproc_per_node")
        if isinstance(npp_value, (int, str)):
            nproc_per_node = npp_value
    return nnodes, nproc_per_node


def _build_install_snippet(
    packages_to_install: Optional[list[str]],
    pip_index_urls: list[str],
) -> str:
    """Build the shell snippet to install Python packages if requested."""
    if not packages_to_install:
        return ""
    return k8s_utils.get_script_for_python_packages(
        packages_to_install,
        pip_index_urls,
    )


def _create_training_hub_progression_instrumentation(
    algorithm: str,
    ckpt_output_dir: str,
    metrics_port: int,
) -> tuple:
    """Instrumentation code injected into training pods (extracted via inspect.getsource).

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into training scripts. This approach
    provides syntax highlighting, testability, and type checking while avoiding
    string templates.

    The constants are embedded directly in the function to make it self-contained
    when extracted via inspect.getsource().

    Args:
        algorithm: Training Hub algorithm ("sft" or "osft")
        ckpt_output_dir: Directory where metrics files are written
        metrics_port: Port for HTTP metrics server

    Returns:
        Tuple of (apply_fn, handler_class) for testing purposes
    """
    import glob
    import http.server
    import json
    import os
    import subprocess
    import threading

    # Training Hub file constants (embedded for self-contained extraction)
    # fmt: off
    SFT_METRICS_FILE_PATTERN = "training_params_and_metrics_global*.jsonl"  # noqa: N806, F841
    SFT_METRICS_FILE_RANK0 = "training_params_and_metrics_global0.jsonl"  # noqa: N806
    OSFT_METRICS_FILE_RANK0 = "training_metrics_0.jsonl"  # noqa: N806, F841
    OSFT_CONFIG_FILE = "training_params.json"  # noqa: N806, F841
    # fmt: on

    # Track if termination message has been written (to avoid duplicates)
    _termination_message_written = False

    class TrainingHubMetricsHandler(http.server.BaseHTTPRequestHandler):
        """HTTP handler that reads JSONL metrics from Training Hub backends."""

        def do_GET(self):
            """Handle GET requests to expose metrics as JSON."""
            try:
                # Read latest metrics
                metrics = self._read_latest_metrics()
                # Transform to controller-compatible schema
                transformed = self._transform_schema(metrics)
            except Exception as e:
                print(
                    f"[Kubeflow] Failed to create progress metrics payload: {e}",
                    flush=True,
                )
                self.send_error(500)
            else:
                # Write termination message when training completes (100%)
                self._maybe_write_termination_message(transformed)
                # Serve JSON
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(transformed, indent=2).encode())

        def _maybe_write_termination_message(self, metrics):
            """Write final metrics to termination message for reliable progress reporting.

            This ensures the controller can get the final training state even if
            HTTP polling misses it. Only writes once when training reaches 100%.
            """
            nonlocal _termination_message_written
            if _termination_message_written:
                return

            progress = metrics.get("progressPercentage")
            if progress is not None and progress >= 100:
                # Write final metrics to termination message for reliable progress reporting
                try:
                    final_metrics = json.dumps(metrics)
                    with open("/dev/termination-log", "w") as f:
                        f.write(final_metrics)
                    _termination_message_written = True
                    print(
                        "[Kubeflow] Final metrics written to termination message",
                        flush=True,
                    )
                except (OSError, ValueError, TypeError) as e:
                    print(
                        f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                        f"Controller will fall back to HTTP polling.",
                        flush=True,
                    )

        def _read_latest_metrics(self):
            """Read last line of JSONL file (most recent metrics from rank 0)."""
            if algorithm == "sft":
                return self._read_sft_metrics()
            elif algorithm == "osft":
                return self._read_osft_metrics()
            else:
                # TODO: Add support for other algorithms (e.g., lora_sft) in future
                return {}

        def _read_osft_metrics(self):
            """Read OSFT metrics from training_metrics_0.jsonl."""
            metrics_file = f"{ckpt_output_dir}/{OSFT_METRICS_FILE_RANK0}"

            try:
                if not os.path.exists(metrics_file):
                    return {}

                # Read config from training_params.json
                config = {}
                config_file = f"{ckpt_output_dir}/{OSFT_CONFIG_FILE}"
                if os.path.exists(config_file):
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                    except Exception:
                        print(
                            "[Kubeflow] Warning: Failed to read OSFT config",
                            flush=True,
                        )

                # Read last line of metrics using tail
                try:
                    result = subprocess.run(
                        ["tail", "-n", "1", metrics_file],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    last_line = result.stdout.strip()
                except subprocess.CalledProcessError:
                    return {}

                if last_line:
                    metrics = json.loads(last_line)
                    if config:
                        metrics["_config"] = config
                    return metrics

            except FileNotFoundError:
                return {}
            except json.JSONDecodeError:
                print("[Kubeflow] Warning: Failed to parse OSFT metrics JSON", flush=True)
                return {}
            except Exception:
                print("[Kubeflow] Error reading OSFT metrics", flush=True)
                return {}

            return {}

        def _read_sft_metrics(self):
            """Read SFT metrics from training_params_and_metrics_global*.jsonl."""
            # Find rank 0 metrics file
            pattern = f"{ckpt_output_dir}/{SFT_METRICS_FILE_PATTERN}"
            files = glob.glob(pattern)

            if not files:
                return {}

            # Prefer rank 0 file
            rank_0_files = [f for f in files if SFT_METRICS_FILE_RANK0 in f]
            metrics_file = rank_0_files[0] if rank_0_files else files[0]

            try:
                if not os.path.exists(metrics_file):
                    return {}

                # Read first line for config
                config = {}
                try:
                    with open(metrics_file) as f:
                        first_line = f.readline().strip()
                        if first_line:
                            config = json.loads(first_line)
                except Exception:
                    print(
                        "[Kubeflow] Warning: Failed to read SFT config",
                        flush=True,
                    )

                # Read last line for latest metrics using tail
                try:
                    result = subprocess.run(
                        ["tail", "-n", "1", metrics_file],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    last_line = result.stdout.strip()
                except subprocess.CalledProcessError:
                    return {}

                if last_line:
                    metrics = json.loads(last_line)
                    if config:
                        metrics["_config"] = config
                    return metrics

            except FileNotFoundError:
                return {}
            except json.JSONDecodeError:
                print("[Kubeflow] Warning: Failed to parse SFT metrics JSON", flush=True)
                return {}
            except Exception:
                print("[Kubeflow] Error reading SFT metrics", flush=True)
                return {}

            return {}

        def _transform_schema(self, metrics):
            """Transform backend schema to controller-compatible progress format."""
            if not metrics:
                return {
                    "progressPercentage": None,
                    "estimatedRemainingSeconds": None,
                    "currentStep": None,
                    "totalSteps": None,
                    "currentEpoch": None,
                    "totalEpochs": None,
                    "trainMetrics": None,
                    "evalMetrics": None,
                }

            # Use algorithm parameter to determine transformation
            if algorithm == "sft":
                return self._transform_sft(metrics)
            elif algorithm == "osft":
                return self._transform_osft(metrics)
            else:
                # TODO: Add support for other algorithms (e.g., lora_sft) in future
                return {}

        def _transform_osft(self, metrics):
            """Transform OSFT schema to controller-compatible format."""
            step = metrics.get("step", 0)
            epoch = metrics.get("epoch", 0)
            steps_per_epoch = metrics.get("steps_per_epoch", 0)

            config = metrics.get("_config", {})
            configured_max_epochs = config.get("max_epochs")

            total_epochs = configured_max_epochs or metrics.get("total_epochs", 1)

            step_total = steps_per_epoch * total_epochs if steps_per_epoch > 0 else 0

            current_step_absolute = step
            percent = min(100, (current_step_absolute / step_total * 100)) if step_total > 0 else 0

            time_per_batch = metrics.get("time_per_batch", 0)
            remaining_steps = step_total - current_step_absolute

            if percent >= 100 or remaining_steps <= 0:
                estimated_remaining_sec = 0
            else:
                estimated_remaining_sec = (
                    int(remaining_steps * time_per_batch) if time_per_batch > 0 else None
                )

            loss_val = metrics.get("loss", 0)
            lr_val = metrics.get("lr")
            grad_norm_val = metrics.get("grad_norm", 0)
            samples_per_second = metrics.get("samples_per_second")
            val_loss_val = metrics.get("val_loss")

            return {
                "progressPercentage": int(round(percent)),
                "estimatedRemainingSeconds": estimated_remaining_sec,
                "currentStep": current_step_absolute,
                "totalSteps": step_total,
                "currentEpoch": epoch + 1,
                "totalEpochs": total_epochs,
                "trainMetrics": {
                    "loss": f"{loss_val:.4f}" if loss_val is not None else None,
                    "learning_rate": f"{lr_val:.6f}" if lr_val is not None else None,
                    "grad_norm": f"{grad_norm_val:.4f}" if grad_norm_val is not None else None,
                    "throughput": (
                        f"{samples_per_second:.2f}" if samples_per_second is not None else None
                    ),
                },
                "evalMetrics": {
                    "eval_loss": f"{val_loss_val:.4f}" if val_loss_val is not None else None,
                },
            }

        def _transform_sft(self, metrics):
            """Transform SFT schema to controller-compatible format."""
            step = metrics.get("step", 0)
            epoch = metrics.get("epoch", 0)
            num_epoch_steps = metrics.get("num_epoch_steps") or metrics.get("num_batches", 0)
            total_samples = metrics.get("total_samples", 0)

            config = metrics.get("_config", {})
            configured_num_epochs = config.get("num_epochs")

            if not num_epoch_steps:
                num_epoch_steps = config.get("num_batches", 0)

            current_epoch = epoch + 1
            current_step_absolute = step
            samples_seen = metrics.get("samples_seen", 0)

            if configured_num_epochs:
                estimated_total_epochs = configured_num_epochs
            elif num_epoch_steps > 0 and samples_seen > 0:
                estimated_progress_through_epochs = (
                    samples_seen / (num_epoch_steps * total_samples / num_epoch_steps)
                    if total_samples > 0
                    else 0
                )
                if estimated_progress_through_epochs > current_epoch:
                    estimated_total_epochs = max(2, int(estimated_progress_through_epochs) + 1)
                else:
                    estimated_total_epochs = current_epoch
            else:
                estimated_total_epochs = (
                    max(2, current_epoch) if current_step_absolute > 0 else current_epoch
                )

            if num_epoch_steps > 0:
                step_total = num_epoch_steps * estimated_total_epochs
            else:
                step_total = max(step, step + 10)

            percent = min(100, (current_step_absolute / step_total * 100)) if step_total > 0 else 0

            throughput = metrics.get("overall_throughput", 0)
            remaining_steps = step_total - current_step_absolute

            if percent >= 100 or remaining_steps <= 0:
                estimated_remaining_sec = 0
            else:
                estimated_remaining_sec = (
                    int(remaining_steps / throughput)
                    if throughput > 0 and remaining_steps > 0
                    else None
                )

            loss_val = metrics.get("avg_loss", 0)
            lr_val = metrics.get("lr")
            grad_norm_val = metrics.get("gradnorm")
            throughput_val = metrics.get("overall_throughput")

            return {
                "progressPercentage": int(round(percent)),
                "estimatedRemainingSeconds": estimated_remaining_sec,
                "currentStep": current_step_absolute,
                "totalSteps": step_total,
                "currentEpoch": current_epoch,
                "totalEpochs": estimated_total_epochs,
                "trainMetrics": {
                    "loss": f"{loss_val:.4f}" if loss_val is not None else None,
                    "learning_rate": f"{lr_val:.6f}" if lr_val is not None else None,
                    "grad_norm": f"{grad_norm_val:.4f}" if grad_norm_val is not None else None,
                    "throughput": f"{throughput_val:.2f}" if throughput_val is not None else None,
                },
                "evalMetrics": {},
            }

        def log_message(self, format, *args):
            """Suppress default HTTP server logging."""
            pass

    def apply_progression_tracking():
        """Start HTTP server for metrics in background thread."""
        try:
            server = http.server.HTTPServer(("0.0.0.0", metrics_port), TrainingHubMetricsHandler)

            # Run server in background thread
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()

            print(
                f"[Kubeflow] Metrics server started on port {metrics_port} for {algorithm}",
                flush=True,
            )

            return server
        except OSError as e:
            print(
                f"[Kubeflow] Warning: Failed to start metrics server on port "
                f"{metrics_port}: {e}. Training will continue without metrics server.",
                flush=True,
            )
            return None
        except Exception as e:
            print(
                f"[Kubeflow] Warning: Unexpected error starting metrics server: {e}. "
                f"Training will continue without metrics server.",
                flush=True,
            )
            return None

    return (apply_progression_tracking, TrainingHubMetricsHandler)


def _render_algorithm_wrapper(algorithm_name: str, func_args: Optional[dict]) -> str:
    """Render a small Python script that calls training_hub.<algorithm>(**func_args).

    Includes termination message writing after training completes (on_train_end equivalent)
    to ensure controller captures final metrics even if HTTP server becomes unreachable.
    """
    base_script = textwrap.dedent("""
    def _write_termination_message(ckpt_output_dir, algorithm):
        \"\"\"Write final metrics to /dev/termination-log for reliable capture.

        Kubernetes reads /dev/termination-log after container exit, providing
        a reliable fallback mechanism for metrics capture that doesn't depend
        on pod lifecycle timing or network availability.
        \"\"\"
        import json
        import os
        import glob

        try:
            # Read final metrics based on algorithm
            metrics = None
            if algorithm == "sft":
                pattern = os.path.join(ckpt_output_dir, "training_params_and_metrics_global0.jsonl")
                files = glob.glob(pattern)
                if files:
                    with open(files[0], 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            metrics = json.loads(lines[-1])
            elif algorithm == "osft":
                metrics_file = os.path.join(ckpt_output_dir, "training_metrics_0.jsonl")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            metrics = json.loads(lines[-1])
            else:
                # TODO: Add support for other algorithms (e.g., lora_sft) in future
                metrics = None

            if metrics:
                # Build final progress JSON (matches controller's AnnotationStatus struct)
                final_progress = {{
                    "progressPercentage": 100,
                    "trainMetrics": {{
                        "loss": str(metrics.get("loss", metrics.get("train_loss", 0))),
                    }},
                    "evalMetrics": {{}},
                }}

                with open("/dev/termination-log", 'w') as f:
                    json.dump(final_progress, f)
                print("[Kubeflow] Termination message written", flush=True)
            else:
                print("[Kubeflow] No metrics found for termination message", flush=True)

        except PermissionError:
            print("[Kubeflow] Cannot write termination message (not in container)", flush=True)
        except Exception as e:
            print(f"[Kubeflow] Warning: Failed to write termination message: {{e}}", flush=True)

    def training_func(func_args):
        import os
        from training_hub import {algo}

        _dp = (func_args or {{}}).get('data_path')
        if _dp:
            print("[PY] Data file found: {{}}".format(_dp), flush=True)
        else:
            print("[PY] Data file NOT found: {{}}".format(_dp), flush=True)

        args = dict(func_args or {{}})
        ckpt_output_dir = args.get('ckpt_output_dir', '/tmp/checkpoints')
        algorithm = '{algo}'

        print("[PY] Launching {algo_upper} training...", flush=True)
        try:
            result = {algo}(**args)
            print("[PY] {algo_upper} training complete. Result=", result, flush=True)

            # Write termination message (on_train_end equivalent)
            # Ensures controller captures final metrics even if HTTP server unreachable
            _write_termination_message(ckpt_output_dir, algorithm)

        except ValueError as e:
            print("Configuration error:", e, flush=True)
        except Exception as e:
            import traceback
            print("[PY] Training failed with error:", e, flush=True)
            traceback.print_exc()

    """).format(algo=algorithm_name, algo_upper=algorithm_name.upper())

    if func_args is None:
        call_line = "training_func({})\n"
    elif isinstance(func_args, dict):
        params_lines: list[str] = ["training_func({\n"]
        for key, value in func_args.items():
            params_lines.append(f"    {repr(key)}: {repr(value)},\n")
        params_lines.append("})\n")
        call_line = "".join(params_lines)
    else:
        call_line = f"training_func({func_args})\n"

    return base_script + call_line


def _render_user_func_code(func: Callable, func_args: Optional[dict]) -> tuple[str, str]:
    """Return (func_code, func_file_basename) embedding the user function and call."""
    if not callable(func):
        raise ValueError(f"Training function must be callable, got function type: {type(func)}")

    func_code = inspect.getsource(func)
    func_code = textwrap.dedent(func_code)

    if func_args is None:
        call_block = f"{func.__name__}()"
    elif isinstance(func_args, dict):
        params_lines: list[str] = [f"{func.__name__}({{"]
        for key, value in func_args.items():
            params_lines.append(f"    {repr(key)}: {repr(value)},")
        params_lines.append("})")
        call_block = "\n".join(params_lines)
    else:
        call_block = f"{func.__name__}({func_args})"

    func_code = f"{func_code}\n{call_block}\n"
    func_file = os.path.basename(inspect.getfile(func))
    return func_code, func_file


def _compose_exec_script(func_code: str, func_file: str) -> str:
    """Compose the final exec script body using the common template."""
    return constants.EXEC_FUNC_SCRIPT.replace("__ENTRYPOINT__", "python").format(
        func_code=func_code,
        func_file=func_file,
    )


def get_training_hub_instrumentation_wrapper(
    algorithm: str,
    ckpt_output_dir: str,
    metrics_port: int = 28080,
) -> str:
    """Generate self-contained instrumentation wrapper via inspect.getsource.

    Extracts _create_training_hub_progression_instrumentation as source code and injects
    a call with the provided parameters.

    Args:
        algorithm: Training Hub algorithm ("sft" or "osft")
        ckpt_output_dir: Directory where metrics files are written
        metrics_port: Port for HTTP metrics server

    Returns:
        Python code as string to be injected before training code
    """
    import inspect
    import textwrap

    # Extract the entire function source
    instrumentation_code = inspect.getsource(_create_training_hub_progression_instrumentation)
    instrumentation_code = textwrap.dedent(instrumentation_code)

    # Build the wrapper with function call
    wrapper = f"""# =============================================================================
# Kubeflow SDK - Training Hub Progression Tracking Instrumentation
# Generated by kubeflow.trainer.rhai.traininghub
# =============================================================================

print("[Kubeflow] Initializing Training Hub progression tracking", flush=True)

# Instrumentation function definition
{instrumentation_code}

# Initialize and apply instrumentation
(
    apply_progression_tracking,
    _,
) = _create_training_hub_progression_instrumentation(
    algorithm="{algorithm}",
    ckpt_output_dir={ckpt_output_dir!r},
    metrics_port={metrics_port}
)
apply_progression_tracking()
print("[Kubeflow] Training Hub progression tracking enabled", flush=True)

# =============================================================================
# USER TRAINING CODE STARTS BELOW
# =============================================================================

"""

    return wrapper


def get_trainer_cr_from_training_hub_trainer(
    runtime: types.Runtime,
    trainer: TrainingHubTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD for TrainingHub trainer.

    Args:
        runtime: Runtime configuration
        trainer: TrainingHubTrainer configuration
        initializer: Optional initializer configuration

    Returns:
        Trainer CRD spec

    Note:
        Distributed training settings (num_nodes, resources) should be configured
        via TrainJob spec.mlPolicy, not in the trainer configuration.
    """
    trainer_crd = models.TrainerV1alpha1Trainer()

    # Derive topology (nnodes, nproc_per_node) from func_args, if provided.
    # nnodes controls TrainJob.spec.trainer.numNodes and therefore PET_NNODES.
    # nproc_per_node controls TrainJob.spec.trainer.numProcPerNode which in turn
    # drives PET_NPROC_PER_NODE via the Torch runtime plugin.
    nnodes, nproc_per_node = _derive_topology_from_func_args(trainer.func_args)
    if nnodes is not None:
        trainer_crd.num_nodes = nnodes

    # Map nproc_per_node directly to NumProcPerNode when provided so that it overrides the
    # runtime ML policy and sets PET_NPROC_PER_NODE as expected. If it is not provided,
    # we leave it unset so the runtime ML policy determines the value.
    if nproc_per_node is not None:
        trainer_crd.num_proc_per_node = models.IoK8sApimachineryPkgUtilIntstrIntOrString(
            nproc_per_node
        )

    # Map explicit resources_per_node exactly like CustomTrainer. If users want to
    # control GPU/CPU quantities, they should pass them via this field.
    if trainer.resources_per_node:
        trainer_crd.resources_per_node = k8s_utils.get_resources_per_node(
            trainer.resources_per_node
        )

    install_snippet = _build_install_snippet(trainer.packages_to_install, trainer.pip_index_urls)

    # Primary case: no user function; generate wrapper that imports and calls algorithm(**func_args)
    if trainer.func is None:
        if not trainer.algorithm:
            raise ValueError("TrainingHubTrainer requires 'algorithm' when 'func' is not provided")

        algorithm_name = trainer.algorithm.value
        raw_code = _render_algorithm_wrapper(algorithm_name, trainer.func_args)

        # Inject progress tracking code if enabled
        if trainer.enable_progression_tracking:
            # Use the same default as the wrapper if ckpt_output_dir is not provided.
            ckpt_dir = "/tmp/checkpoints"
            if trainer.func_args and "ckpt_output_dir" in trainer.func_args:
                ckpt_dir = trainer.func_args["ckpt_output_dir"]

            progress_code = get_training_hub_instrumentation_wrapper(
                algorithm=algorithm_name,
                ckpt_output_dir=ckpt_dir,
                metrics_port=trainer.metrics_port,
            )
            raw_code = progress_code + "\n" + raw_code

        exec_script = _compose_exec_script(raw_code, "training_script.py")
        full_script = install_snippet + exec_script

        trainer_crd.command = ["bash", "-c"]
        trainer_crd.args = [full_script]
    else:
        # Secondary case: user provided function; embed their function and call with kwargs
        func_code, func_file = _render_user_func_code(trainer.func, trainer.func_args)

        # Inject progress tracking code if enabled (for custom functions)
        # Try to extract ckpt_output_dir from the function code or use default
        if trainer.enable_progression_tracking and trainer.algorithm:
            # For custom functions, use a default metrics directory
            # User should ensure their function writes to this directory
            ckpt_dir = "/tmp/training_metrics"
            if trainer.func_args and "ckpt_output_dir" in trainer.func_args:
                ckpt_dir = trainer.func_args["ckpt_output_dir"]

            progress_code = get_training_hub_instrumentation_wrapper(
                algorithm=trainer.algorithm.value,
                ckpt_output_dir=ckpt_dir,
                metrics_port=trainer.metrics_port,
            )
            func_code = progress_code + "\n" + func_code

        exec_script = _compose_exec_script(func_code, func_file)
        full_script = install_snippet + exec_script

        trainer_crd.command = ["bash", "-c"]
        trainer_crd.args = [full_script]

    # Add environment variables to the Trainer if provided by user
    trainer_crd.env = (
        [models.IoK8sApiCoreV1EnvVar(name=k, value=v) for k, v in trainer.env.items()]
        if trainer.env
        else None
    )

    return trainer_crd
