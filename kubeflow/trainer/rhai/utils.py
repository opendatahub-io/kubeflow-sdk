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

"""Utilities for converting RHAI trainers to Kubernetes TrainJob CRDs."""

import inspect
import os
import textwrap
from typing import TYPE_CHECKING, Optional

from kubeflow.trainer.backends.kubernetes import utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

if TYPE_CHECKING:
    from kubeflow.trainer.rhai import RHAITrainer
    from kubeflow.trainer.rhai.transformers import TransformersTrainer
    from kubeflow.training import models


def get_trainer_crd_from_rhai_trainer(
    runtime: types.Runtime,
    trainer: "RHAITrainer",
    initializer: Optional[types.Initializer] = None,
) -> "models.TrainerV1alpha1Trainer":
    """Dispatcher: Convert RHAI trainer to Kubernetes TrainJob CRD.

    This follows PR #22's pattern - a single dispatcher function that routes
    to the appropriate converter based on trainer type.

    Args:
        runtime: Training runtime configuration.
        trainer: RHAI trainer (TransformersTrainer, TrainingHubTrainer, etc.).
        initializer: Optional dataset/model initializer configuration.

    Returns:
        TrainJob Trainer CR ready for Kubernetes submission.

    Raises:
        ValueError: If trainer type is not supported.
    """
    from kubeflow.trainer.rhai.transformers import TransformersTrainer

    if isinstance(trainer, TransformersTrainer):
        return get_trainer_crd_from_transformers_trainer(runtime, trainer)
    # Future: Add TrainingHubTrainer support when PR #22 merges
    # elif isinstance(trainer, TrainingHubTrainer):
    #     return get_trainer_crd_from_training_hub_trainer(runtime, trainer, initializer)
    else:
        raise ValueError(
            f"Unsupported RHAI trainer type: {type(trainer)}. Supported types: TransformersTrainer"
        )


def get_trainer_crd_from_transformers_trainer(
    runtime: types.Runtime,
    trainer: "TransformersTrainer",
) -> "models.TrainerV1alpha1Trainer":
    """Convert TransformersTrainer to Kubernetes TrainJob CRD.

    Args:
        runtime: Training runtime configuration.
        trainer: TransformersTrainer with instrumentation settings.

    Returns:
        TrainJob Trainer CR with instrumented command.
    """
    from kubeflow.training import models

    trainer_cr = models.TrainerV1alpha1Trainer()

    # Add number of nodes
    if trainer.num_nodes:
        trainer_cr.num_nodes = trainer.num_nodes

    # Add resources per node
    if trainer.resources_per_node:
        trainer_cr.resources_per_node = k8s_utils.get_resources_per_node(trainer.resources_per_node)

    # Generate command with or without instrumentation
    if trainer.enable_progression_tracking:
        trainer_cr.command = _get_instrumented_command(runtime, trainer)
    else:
        # No instrumentation - just run the function directly
        trainer_cr.command = k8s_utils.get_command_using_train_func(
            runtime,
            trainer.func,
            trainer.func_args,
            trainer.pip_index_urls,
            trainer.packages_to_install,
        )

    # Add environment variables
    if trainer.env:
        trainer_cr.env = [
            models.IoK8sApiCoreV1EnvVar(name=key, value=value) for key, value in trainer.env.items()
        ]

    return trainer_cr


def _get_instrumented_command(
    runtime: types.Runtime,
    trainer: "TransformersTrainer",
) -> list[str]:
    """Generate command with self-contained instrumentation wrapper.

    This generates a complete Python script with HTTP server, progress callback,
    and Trainer monkey-patching all baked in - no SDK imports needed at runtime.

    Args:
        runtime: Training runtime configuration.
        trainer: TransformersTrainer with instrumentation settings.

    Returns:
        Command list suitable for TrainJob trainer.command.

    Raises:
        ValueError: If runtime doesn't have a trainer or function is not callable.
    """
    from kubeflow.trainer.rhai.transformers import get_transformers_instrumentation_wrapper

    if not runtime.trainer:
        raise ValueError(f"Runtime must have a trainer: {runtime}")

    if not callable(trainer.func):
        raise ValueError(
            f"Training function must be callable, got function type: {type(trainer.func)}"
        )

    # Get self-contained instrumentation wrapper
    wrapper_script = get_transformers_instrumentation_wrapper(
        metrics_port=trainer.metrics_port,
        custom_metrics=trainer.custom_metrics or {},
    )

    # Extract user function code
    func_code = inspect.getsource(trainer.func)
    func_file = os.path.basename(inspect.getfile(trainer.func))
    func_code = textwrap.dedent(func_code)

    # Build function call
    if trainer.func_args is None:
        func_call = f"{trainer.func.__name__}()"
    else:
        func_call = f"{trainer.func.__name__}(**{trainer.func_args})"

    # Inject user code into wrapper
    user_code = f"{func_code}\n{func_call}\n"
    full_code = wrapper_script.replace("{{user_func_import_and_call}}", user_code)

    # Handle MPI-specific paths
    is_mpi = runtime.trainer.command[0] == "mpirun"
    if is_mpi:
        func_file = os.path.join(constants.DEFAULT_MPI_USER_HOME, func_file)

    # Install packages if needed
    install_packages = ""
    if trainer.packages_to_install:
        install_packages = k8s_utils.get_script_for_python_packages(
            trainer.packages_to_install,
            trainer.pip_index_urls,
            is_mpi,
        )

    # Build final command
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=full_code, func_file=func_file)
            if install_packages:
                exec_script = install_packages + exec_script
            command.append(exec_script)
        else:
            command.append(c)

    return command


# =============================================================================
# Progress Monitoring Utilities
# =============================================================================


def get_job_progress(
    name: str,
    namespace: str = "default",
    backend_config=None,
) -> Optional[dict]:
    """Get current training progress for a TrainJob with progression tracking.

    This function retrieves progression tracking metrics from the TrainJob's
    annotations. Only works for jobs with progression tracking enabled
    (e.g., TransformersTrainer with enable_progression_tracking=True).

    Args:
        name: Name of the TrainJob
        namespace: Kubernetes namespace (only for KubernetesBackend)
        backend_config: Backend configuration (defaults to KubernetesBackendConfig)

    Returns:
        Dictionary with training progress data, or None if progression tracking
        is not enabled or metrics are not yet available.

        Dictionary structure:
        {
            "status": str,              # "training", "completed", etc.
            "status_message": str,      # "45% complete • 2h 44m left"
            "progress": {
                "step_current": int,
                "step_total": int,
                "percent": float,
                "epoch": int
            },
            "time": {
                "elapsed": str,         # "2h 15m"
                "remaining": str,       # "2h 44m" (or None)
                "started_at": str,      # "12:00:00"
                "last_update": str      # "14:15:45"
            },
            "metrics": {
                "loss": float,
                "learning_rate": float,
                "throughput_samples_sec": float,
                # ... any custom metrics
            }
        }

    Example:
        ```python
        from kubeflow.trainer.rhai import get_job_progress

        progress = get_job_progress("my-training-job", namespace="default")
        if progress:
            print(f"Progress: {progress['progress']['percent']}%")
            print(f"Status: {progress['status_message']}")
        ```

    Raises:
        TimeoutError: Timeout to get TrainJob
        RuntimeError: Failed to get TrainJob
    """
    # Import here to avoid circular import
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.trainer.api import TrainerClient

    if backend_config is None:
        backend_config = KubernetesBackendConfig(namespace=namespace)

    client = TrainerClient(backend_config=backend_config)
    job = client.get_job(name=name)
    return _parse_progress_from_job(job)


def _parse_progress_from_job(job) -> Optional[dict]:
    """Parse progression tracking metrics from TrainJob annotations.

    Args:
        job: TrainJob object

    Returns:
        Dictionary with progress data or None if progression tracking not available
    """
    import json

    from kubernetes import client

    from kubeflow.trainer.rhai.constants import ANNOTATION_TRAINER_STATUS

    # Handle both dict and V1ObjectMeta types
    if hasattr(job, "metadata") and isinstance(job.metadata, client.V1ObjectMeta):
        annotations = job.metadata.annotations or {}
    elif isinstance(job, dict):
        annotations = job.get("metadata", {}).get("annotations", {})
    else:
        return None

    status_annotation = annotations.get(ANNOTATION_TRAINER_STATUS)
    if not status_annotation:
        return None

    try:
        return json.loads(status_annotation)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse progress annotation: {e}")
        return None
