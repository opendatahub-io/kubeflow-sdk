from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect
import os
import textwrap

from kubeflow_trainer_api import models

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


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

    func: Callable | None = None
    func_args: dict | None = None
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: dict[str, str] | None = None
    algorithm: TrainingHubAlgorithms | None = None
    resources_per_node: dict | None = None

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
    func_args: dict | None,
) -> tuple[int | None, int | str | None]:
    """Return (nnodes, nproc_per_node) based on provided func_args.

    If values are not provided in func_args, they are left as None so that the
    TrainingRuntime ML policy can supply appropriate defaults instead of the SDK.
    """
    nnodes: int | None = None
    nproc_per_node: int | str | None = None
    if isinstance(func_args, dict):
        nnodes_value = func_args.get("nnodes")
        if isinstance(nnodes_value, int):
            nnodes = nnodes_value
        npp_value = func_args.get("nproc_per_node")
        if isinstance(npp_value, (int, str)):
            nproc_per_node = npp_value
    return nnodes, nproc_per_node


def _build_install_snippet(
    packages_to_install: list[str] | None,
    pip_index_urls: list[str],
) -> str:
    """Build the shell snippet to install Python packages if requested."""
    if not packages_to_install:
        return ""
    return k8s_utils.get_script_for_python_packages(
        packages_to_install,
        pip_index_urls,
    )


def _render_algorithm_wrapper(algorithm_metadata: dict, func_args: dict | None) -> str:
    """Render a small Python script that calls training_hub.<algorithm>(**func_args).

    Includes termination message writing after training completes (on_train_end equivalent)
    to ensure controller captures final metrics even if HTTP server becomes unreachable.

    Args:
        algorithm_metadata: Pre-resolved algorithm metadata dict from
            get_algorithm_pod_metadata() containing:
            - name: Algorithm name
            - metrics_file_rank0: Filename for rank 0 metrics
        func_args: Arguments to pass to the training function
    """
    # Extract values from pre-validated metadata
    algorithm_name = algorithm_metadata["name"]
    metrics_file_rank0 = algorithm_metadata["metrics_file_rank0"]

    base_script = textwrap.dedent("""
    def _write_termination_message(ckpt_output_dir, algorithm, metrics_file_rank0):
        \"\"\"Write final metrics to /dev/termination-log for reliable capture.

        Kubernetes reads /dev/termination-log after container exit, providing
        a reliable fallback mechanism for metrics capture that doesn't depend
        on pod lifecycle timing or network availability.
        \"\"\"
        import json
        import os

        # Skip termination message for algorithms without metrics files
        if metrics_file_rank0 is None:
            print(
                "[Kubeflow] Algorithm produces no metrics files - skipping termination message",
                flush=True,
            )
            return

        try:
            # If we reach here, metrics ARE expected for this algorithm
            metrics_file = os.path.join(ckpt_output_dir, metrics_file_rank0)

            # Check if expected metrics file exists
            if not os.path.exists(metrics_file):
                print(
                    f"[Kubeflow] WARNING: Expected metrics file not found: {{metrics_file_rank0}}",
                    flush=True,
                )
                print(
                    "[Kubeflow] Training may have failed to write metrics or terminated early",
                    flush=True,
                )
                return

            # Read final metrics from rank 0 file
            metrics = None
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    metrics = json.loads(lines[-1])
                else:
                    print(
                        f"[Kubeflow] WARNING: Metrics file is empty: {{metrics_file_rank0}}",
                        flush=True,
                    )
                    return

            if metrics:
                # Build final progress JSON (matches controller's AnnotationStatus struct)
                # Try different metric names based on algorithm
                loss_value = (
                    metrics.get("loss")
                    or metrics.get("train_loss")
                    or metrics.get("avg_loss")
                    or 0
                )
                final_progress = {{
                    "progressPercentage": 100,
                    "trainMetrics": {{
                        "loss": str(loss_value),
                    }},
                    "evalMetrics": {{}},
                }}

                with open("/dev/termination-log", 'w') as f:
                    json.dump(final_progress, f)
                print("[Kubeflow] Termination message written with final metrics", flush=True)
            else:
                print(
                    "[Kubeflow] WARNING: Metrics file read but could not parse final metrics",
                    flush=True,
                )


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
        metrics_file_rank0 = {metrics_file_rank0!r}

        print("[PY] Launching {algo_upper} training...", flush=True)
        try:
            result = {algo}(**args)
            print("[PY] {algo_upper} training complete. Result=", result, flush=True)

            # Write termination message (on_train_end equivalent)
            # Ensures controller captures final metrics even if HTTP server unreachable
            _write_termination_message(ckpt_output_dir, algorithm, metrics_file_rank0)

        except ValueError as e:
            print("Configuration error:", e, flush=True)
            # Propagate configuration errors so the pod fails
            raise
        except Exception as e:
            import traceback
            print("[PY] Training failed with error:", e, flush=True)
            traceback.print_exc()
            # Propagate errors so the pod fails
            raise

    """).format(
        algo=algorithm_name,
        algo_upper=algorithm_name.upper(),
        metrics_file_rank0=metrics_file_rank0,
    )

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


def _render_user_func_code(func: Callable, func_args: dict | None) -> tuple[str, str]:
    """Return (func_code, func_file_basename) embedding the user function and call."""
    if not callable(func):
        raise ValueError(f"Training function must be callable, got function type: {type(func)}")

    func_code = inspect.getsource(func)
    func_code = textwrap.dedent(func_code)

    if func_args is None:
        call_block = f"{func.__name__}()"
    elif isinstance(func_args, dict):
        params_lines: list[str] = [f"{func.__name__}(**{{"]
        for key, value in func_args.items():
            params_lines.append(f"    {repr(key)}: {repr(value)},")
        params_lines.append("})")
        call_block = "\n".join(params_lines)
    else:
        call_block = f"{func.__name__}({func_args})"

    func_code = f"{func_code}\n{call_block}\n"
    func_file = os.path.basename(inspect.getfile(func))
    return func_code, func_file


def _get_command_from_runtime(
    runtime: types.Runtime,
    func_code: str,
    func_file: str,
    install_snippet: str,
) -> list[str]:
    """Build command using runtime's command template (matches CustomTrainer pattern).

    Args:
        runtime: Runtime configuration with command template
        func_code: The training function code to execute
        func_file: The filename to write the code to
        install_snippet: Package installation script to prepend

    Returns:
        Command list ready for trainer_crd.command/args

    Note:
        This matches CustomTrainer's approach of using runtime.trainer.command directly,
        ensuring consistent behavior across all trainer types.
    """
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            # Format the runtime's command template with our code
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_snippet:
                exec_script = install_snippet + exec_script
            command.append(exec_script)
        else:
            command.append(c)
    return command


def get_training_hub_instrumentation_wrapper(
    algorithm: str,
    ckpt_output_dir: str,
    metrics_port: int = 28080,
) -> str:
    """Generate Training Hub progression tracking wrapper with module import.

    Uses module-based instrumentation (same approach as TransformersTrainer) with upfront
    SDK version validation.

    Args:
        algorithm: Training Hub algorithm name ("sft" or "osft")
        ckpt_output_dir: Directory where metrics files are written
        metrics_port: Port for HTTP metrics server

    Returns:
        Python code as string to be injected before training code

    Raises:
        ValueError: If algorithm is not supported (via get_algorithm_pod_metadata).
    """
    import kubeflow
    from kubeflow.trainer.algorithms import get_algorithm_pod_metadata

    # Resolve algorithm metadata from centralized registry
    # This validates the algorithm name and retrieves its metadata
    algorithm_metadata = get_algorithm_pod_metadata(algorithm)

    client_sdk_version = kubeflow.__version__

    # Serialize algorithm_metadata as string for injection
    import pprint

    metadata_str = pprint.pformat(algorithm_metadata, indent=4, width=100, sort_dicts=False)

    # Build the wrapper with function call
    wrapper = f"""# =============================================================================
# Kubeflow SDK - Training Hub Progression Tracking Instrumentation
# Generated by kubeflow.trainer.rhai.traininghub (SDK {client_sdk_version})
# =============================================================================

import kubeflow as runtime_kubeflow
from packaging import version

CLIENT_SDK_VERSION = "{client_sdk_version}"
RUNTIME_SDK_VERSION = runtime_kubeflow.__version__
MIN_SDK_VERSION = "{constants.MIN_SDK_VERSION}"

# Check if both client and runtime SDK versions meet minimum requirement
client_version_valid = version.parse(CLIENT_SDK_VERSION) >= version.parse(MIN_SDK_VERSION)
runtime_version_valid = version.parse(RUNTIME_SDK_VERSION) >= version.parse(MIN_SDK_VERSION)

if not client_version_valid or not runtime_version_valid:
    error_parts = []
    error_parts.append("Training Hub progression tracking requires SDK version {{MIN_SDK_VERSION}} or later.")
    error_parts.append(f"Current versions - Client: {{CLIENT_SDK_VERSION}}, Runtime: {{RUNTIME_SDK_VERSION}}")

    if not client_version_valid and not runtime_version_valid:
        error_parts.append(
            f"Both client and runtime SDK versions are below {{MIN_SDK_VERSION}}.\\n"
            f"Actions required:\\n"
            f"  1. Upgrade client SDK: pip install --upgrade kubeflow-trainer>=0.4.0\\n"
            f"  2. Upgrade runtime SDK (choose one):\\n"
            f"     - Add 'kubeflow-trainer>=0.4.0' to packages_to_install field\\n"
            f"     - Use a training runtime image with SDK {{MIN_SDK_VERSION}}+ from clusterTrainingRuntimes"
        )
    elif not client_version_valid:
        error_parts.append(
            f"Client SDK version {{CLIENT_SDK_VERSION}} is below minimum {{MIN_SDK_VERSION}}.\\n"
            f"Action required: pip install --upgrade kubeflow-trainer>=0.4.0"
        )
    else:  # not runtime_version_valid
        error_parts.append(
            f"Runtime SDK version {{RUNTIME_SDK_VERSION}} is below minimum {{MIN_SDK_VERSION}}.\\n"
            f"Choose one of these options:\\n"
            f"  1. Add 'kubeflow-trainer>=0.4.0' to packages_to_install field\\n"
            f"  2. Use a training runtime image with SDK {{MIN_SDK_VERSION}}+ from clusterTrainingRuntimes"
        )

    raise RuntimeError("\\n".join(error_parts))

# Warn if versions are mismatched (both valid but different)
if CLIENT_SDK_VERSION != RUNTIME_SDK_VERSION:
    print(
        f"[Kubeflow] Warning: This job was created with SDK {{CLIENT_SDK_VERSION}} "
        f"but the runtime image has SDK {{RUNTIME_SDK_VERSION}}. "
        f"If you encounter unexpected errors, consider matching the versions by either: "
        f"(1) adding 'kubeflow-trainer={{{{CLIENT_SDK_VERSION}}}}' to packages_to_install, or "
        f"(2) using a training runtime image with SDK {{CLIENT_SDK_VERSION}}.",
        flush=True
    )

try:
    # Import Training Hub progression instrumentation (guaranteed to exist if both versions >= 0.4.0)
    from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (
        create_traininghub_progression_instrumentation,
    )

    print("[Kubeflow] Initializing Training Hub progression tracking", flush=True)
    print(f"[Kubeflow] Client SDK: {{CLIENT_SDK_VERSION}}, Runtime SDK: {{RUNTIME_SDK_VERSION}}", flush=True)

    algorithm_metadata = {metadata_str}

    (
        apply_progression_tracking,
        _,
    ) = create_traininghub_progression_instrumentation(
        algorithm_metadata=algorithm_metadata,
        ckpt_output_dir={ckpt_output_dir!r},
        metrics_port={metrics_port}
    )
    apply_progression_tracking()
    print("[Kubeflow] Training Hub progression tracking enabled", flush=True)
except Exception as e:
    raise RuntimeError(
        f"Failed to initialize Training Hub progression tracking: {{type(e).__name__}}: {{e}}"
    ) from e

# =============================================================================
# USER TRAINING CODE STARTS BELOW
# =============================================================================

"""

    return wrapper


def get_trainer_cr_from_training_hub_trainer(
    runtime: types.Runtime,
    trainer: TrainingHubTrainer,
    initializer: types.Initializer | None = None,
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
    # Determine the correct entrypoint command based on algorithm.
    # Each algorithm specifies its own entrypoint in the registry.
    entrypoint = constants.TORCH_COMMAND
    if trainer.algorithm:
        from kubeflow.trainer.algorithms import get_algorithm_spec

        entrypoint = get_algorithm_spec(trainer.algorithm.value).entrypoint

    runtime.trainer.set_command(entrypoint)

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
        # KFT v2.2+: TrainerV1alpha1Trainer.num_proc_per_node is an int (not IntOrString).
        trainer_crd.num_proc_per_node = int(nproc_per_node)

    # Map explicit resources_per_node exactly like CustomTrainer. If users want to
    # control GPU/CPU quantities, they should pass them via this field.
    if trainer.resources_per_node:
        trainer_crd.resources_per_node = k8s_utils.get_resources_per_node(
            trainer.resources_per_node
        )

    install_snippet = _build_install_snippet(trainer.packages_to_install, trainer.pip_index_urls)

    # Generate the training function code based on mode
    if trainer.func is None:
        # Primary case: no user function; generate wrapper that calls algorithm(**func_args)
        if not trainer.algorithm:
            raise ValueError("TrainingHubTrainer requires 'algorithm' when 'func' is not provided")

        from kubeflow.trainer.algorithms import get_algorithm_pod_metadata

        algorithm_name = trainer.algorithm.value

        # Resolve algorithm metadata from centralized registry
        algorithm_metadata = get_algorithm_pod_metadata(algorithm_name)

        func_code = _render_algorithm_wrapper(algorithm_metadata, trainer.func_args)
        func_file = "training_script.py"
    else:
        # Secondary case: user provided function; embed their function and call with kwargs
        func_code, func_file = _render_user_func_code(trainer.func, trainer.func_args)
        algorithm_name = trainer.algorithm.value if trainer.algorithm else None

    # Add progress tracking instrumentation if enabled (common for both modes)
    if trainer.enable_progression_tracking:
        # Determine checkpoint directory (algorithm mode vs user function mode)
        ckpt_dir = "/tmp/checkpoints" if trainer.func is None else "/tmp/training_metrics"

        # Override with user-provided value if available
        if trainer.func_args and "ckpt_output_dir" in trainer.func_args:
            ckpt_dir = trainer.func_args["ckpt_output_dir"]

        # Only add instrumentation if algorithm is specified
        if algorithm_name:
            progress_code = get_training_hub_instrumentation_wrapper(
                algorithm=algorithm_name,
                ckpt_output_dir=ckpt_dir,
                metrics_port=trainer.metrics_port,
            )
            func_code = progress_code + "\n" + func_code

    # Build command using runtime's template (common for both modes)
    trainer_crd.command = _get_command_from_runtime(
        runtime=runtime,
        func_code=func_code,
        func_file=func_file,
        install_snippet=install_snippet,
    )

    # Add environment variables to the Trainer if provided by user
    trainer_crd.env = (
        [models.IoK8sApiCoreV1EnvVar(name=k, value=v) for k, v in trainer.env.items()]
        if trainer.env
        else None
    )

    return trainer_crd
