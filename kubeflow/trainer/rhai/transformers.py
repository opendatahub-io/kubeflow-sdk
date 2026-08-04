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

from collections.abc import Callable
from dataclasses import dataclass, field
import inspect
import os
import textwrap

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.constants import S3_URI_SCHEME
from kubeflow.trainer.types import types


@dataclass
class PeriodicCheckpointConfig:
    """Configuration for periodic checkpointing in Transformers trainers.

    Args:
        save_strategy: Strategy for saving checkpoints ("steps", "epoch", or "no")
        save_steps: Number of steps between checkpoints (required if save_strategy="steps")
        save_total_limit: Maximum number of checkpoints to keep (older ones are deleted)
    """

    save_strategy: str = "epoch"
    save_steps: int | None = None
    save_total_limit: int | None = 3

    def __post_init__(self):
        """Validate configuration."""
        valid_strategies = {"steps", "epoch", "no"}
        if self.save_strategy not in valid_strategies:
            raise ValueError(
                f"save_strategy must be one of {valid_strategies}, got '{self.save_strategy}'"
            )

        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("save_steps must be specified when save_strategy='steps'")

        if self.save_total_limit is not None and self.save_total_limit < 1:
            raise ValueError(f"save_total_limit must be >= 1, got {self.save_total_limit}")


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
        enable_jit_checkpoint: Enable just-in-time checkpointing on SIGTERM. Default: False.
                              Automatically enabled when output_dir is provided.
        output_dir: Directory for saving checkpoints. Supports PVC URIs (pvc://<name>/<path>)
                    or S3 URIs (s3://<bucket>/<path>) for automatic volume mounting.
                    When provided, automatically enables JIT checkpointing.
        periodic_checkpoint_config: Optional configuration for periodic checkpointing.
                                   See PeriodicCheckpointConfig for available options.
        data_connection_name: Name of the Kubernetes secret containing S3 credentials.
                              Required when output_dir uses s3:// scheme. To find or create
                              a data connection: in the RHOAI dashboard, navigate to your
                              Data Science project, go to the Connections tab, and either
                              copy an existing connection's resource name or create a new
                              S3-compatible connection.
        verify_cloud_storage_access: Test cloud storage access before training starts. When enabled,
                               writes and reads a small test file to validate that credentials,
                               permissions, and bucket access work correctly. This catches
                               configuration errors early before training begins. Default: True.
                               Only disable if experiencing false positives and you're confident
                               your storage configuration is correct.
        verify_cloud_storage_ssl: Verify SSL certificates for cloud checkpoint storage
                                       (S3, etc.). Default: True. Set to False only if using
                                       S3-compatible storage with self-signed certificates.
                                       This parameter only applies when using a custom S3-compatible
                                       endpoint (via AWS_S3_ENDPOINT environment variable).
                                       WARNING: Disabling SSL verification is a security risk.

    Raises:
        ValueError: If metrics_port is not in range 1024-65535.
        ValueError: If metrics_poll_interval_seconds is not in range 5-300.
        ValueError: If func is not callable.
        ValueError: If output_dir uses unsupported URI scheme (only pvc:// and s3:// are supported).
    """

    # Core training function (same as CustomTrainer)
    func: Callable
    func_args: dict | None = None
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: int | None = None
    resources_per_node: dict | None = None
    env: dict[str, str] | None = None

    # Instrumentation features
    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30

    # Checkpoint configuration
    enable_jit_checkpoint: bool = False
    output_dir: str | None = None
    periodic_checkpoint_config: PeriodicCheckpointConfig | None = None
    data_connection_name: str | None = None
    verify_cloud_storage_access: bool = True
    verify_cloud_storage_ssl: bool = True

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

        # Normalize and validate output_dir URI
        if self.output_dir:
            # Import here to avoid circular import
            from kubeflow.trainer.rhai.utils import normalize_and_validate_output_dir

            self.output_dir = normalize_and_validate_output_dir(self.output_dir)

        # Validate S3 output_dir requires data_connection_name
        if (
            self.output_dir
            and self.output_dir.startswith(S3_URI_SCHEME)
            and not self.data_connection_name
        ):
            raise ValueError(
                "data_connection_name is required when output_dir uses s3:// scheme. "
                "Please provide the name of the Kubernetes secret containing S3 credentials."
            )

        # Auto-enable JIT checkpoint if output_dir is provided
        if self.output_dir and not self.enable_jit_checkpoint:
            self.enable_jit_checkpoint = True


def get_trainer_cr_from_transformers_trainer(
    runtime: types.Runtime,
    trainer: TransformersTrainer,
    initializer: types.Initializer | None = None,
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

    # Inject checkpoint code if enabled (header before user code, footer after)
    checkpoint_header, checkpoint_footer = _build_checkpoint_code(trainer)
    if checkpoint_header:
        func_code = f"{checkpoint_header}\n\n{func_code}"
    if checkpoint_footer:
        func_code = f"{func_code}\n\n{checkpoint_footer}"

    # Build the command directly with the wrapped function code
    func_file = os.path.basename(inspect.getfile(trainer.func))

    # Install Python packages if required
    install_packages = ""
    if trainer.packages_to_install:
        install_packages = utils.get_script_for_python_packages(
            trainer.packages_to_install,
            trainer.pip_index_urls,
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


def _build_checkpoint_code(trainer: TransformersTrainer) -> tuple[str, str]:
    """Generate checkpoint injection code for the trainer.

    Returns:
        Tuple of (header_code, footer_code). Header runs before user code,
        footer runs after. Both are empty strings when checkpointing is disabled.
    """
    # Only inject if JIT or periodic checkpoint is enabled
    if not trainer.enable_jit_checkpoint and not trainer.periodic_checkpoint_config:
        return "", ""

    # Create default periodic config if JIT is enabled but no config provided
    periodic_config = trainer.periodic_checkpoint_config
    if trainer.enable_jit_checkpoint and periodic_config is None:
        periodic_config = PeriodicCheckpointConfig()

    # Convert PeriodicCheckpointConfig to dict for injection
    periodic_config_dict = None
    if periodic_config:
        periodic_config_dict = {
            "save_strategy": periodic_config.save_strategy,
            "save_steps": periodic_config.save_steps,
            "save_total_limit": periodic_config.save_total_limit,
        }

    # Parse output_dir URI to get resolved path for checkpoint code
    from kubeflow.trainer.rhai.utils import parse_output_dir_uri

    resolved_output_dir, _ = parse_output_dir_uri(trainer.output_dir)

    # Check if using S3 storage
    cloud_remote_storage_uri = None
    if trainer.output_dir and trainer.output_dir.startswith(S3_URI_SCHEME):
        cloud_remote_storage_uri = trainer.output_dir

    # Generate checkpoint injection code
    return get_jit_checkpoint_injection_code(
        output_dir=resolved_output_dir,
        cloud_remote_storage_uri=cloud_remote_storage_uri,
        periodic_checkpoint_config=periodic_config_dict,
        enable_jit_checkpoint=trainer.enable_jit_checkpoint,
        verify_cloud_storage_access=trainer.verify_cloud_storage_access,
        verify_cloud_storage_ssl=trainer.verify_cloud_storage_ssl,
    )


# Re-exports for backward compatibility
from kubeflow.trainer.rhai.instrumentation.transformers import (  # noqa: E402, F401
    _create_checkpoint_instrumentation,
    _create_progression_instrumentation,
    get_jit_checkpoint_injection_code,
    get_transformers_instrumentation_wrapper,
)
