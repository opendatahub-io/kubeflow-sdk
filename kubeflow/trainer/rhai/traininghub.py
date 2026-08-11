from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kubeflow_trainer_api import models

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class TrainingHubAlgorithms(Enum):
    """Algorithm for TrainingHub Trainer."""

    SFT = "sft"
    OSFT = "osft"
    LORA_SFT = "lora_sft"
    LORA_GRPO = "lora_grpo"


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
        callbacks: Optional list of Training Hub callback classes or instances.
            Classes are serialized into the training pod and passed to training_hub
            APIs (sft, osft, lora_sft, lora_grpo). Define callbacks at module level
            so inspect.getsource can serialize them.
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
    callbacks: list[Any] | None = None

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

        from kubeflow.trainer.rhai.instrumentation.traininghub_callbacks import (
            validate_callbacks,
        )

        validate_callbacks(self.callbacks)


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
        trainer_crd.num_proc_per_node = nproc_per_node

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

    if trainer.callbacks:
        from kubeflow.trainer.rhai.instrumentation.traininghub_callbacks import (
            build_training_hub_callback_injection_code,
        )

        callback_code = build_training_hub_callback_injection_code(trainer.callbacks)
        func_code = callback_code + "\n" + func_code

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


# Re-exports for backward compatibility
from kubeflow.trainer.rhai.instrumentation.traininghub import (  # noqa: E402, F401
    get_training_hub_instrumentation_wrapper,
)
from kubeflow.trainer.rhai.instrumentation.traininghub_codegen import (  # noqa: E402, F401
    _get_command_from_runtime,
    _render_algorithm_wrapper,
    _render_user_func_code,
)
from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (  # noqa: E402, F401
    _create_training_hub_progression_instrumentation,
)
