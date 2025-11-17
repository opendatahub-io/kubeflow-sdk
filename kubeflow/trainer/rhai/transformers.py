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

    Provides instrumentation for:
    - Training progression tracking via HTTP metrics endpoint
    - Custom metrics tracking via callback

    Supported Trainers (via base class patching):
    - transformers.Trainer (standard fine-tuning)
    - trl.SFTTrainer (Supervised Fine-Tuning)
    - trl.DPOTrainer (Direct Preference Optimization)
    - trl.PPOTrainer (Proximal Policy Optimization)
    - trl.RewardTrainer (Reward Model Training)
    - Any other Trainer subclass

    When to use:
    - You need real-time progress tracking in the UI
    - You're using HuggingFace Transformers or TRL
    - You want to track custom metrics beyond loss/lr

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
        custom_metrics: Dict mapping log keys to metric names to track.
                       Example: {"eval_accuracy": "accuracy", "custom_metric": "custom"}
                       These will be exposed in the /metrics endpoint.
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
    custom_metrics: dict[str, str] = field(default_factory=dict)


def get_transformers_instrumentation_wrapper(
    metrics_port: int,
    custom_metrics: dict[str, str],
) -> str:
    """Generate self-contained instrumentation wrapper script.

    Extracts real Python classes using inspect.getsource() for better
    maintainability, type checking, and IDE support.

    Args:
        metrics_port: Port for HTTP metrics server.
        custom_metrics: Dict mapping log keys to metric names.

    Returns:
        Python code as string with {{user_func_import_and_call}} placeholder.
    """
    import inspect
    import json
    import textwrap

    from kubeflow.trainer.rhai import progression_instrumentation

    # Extract source from real Python module
    update_metrics_function = inspect.getsource(progression_instrumentation._update_metrics)
    get_metrics_json_function = inspect.getsource(progression_instrumentation._get_metrics_json)
    metrics_server_class = inspect.getsource(progression_instrumentation.MetricsServer)
    progress_callback_class = inspect.getsource(progression_instrumentation.ProgressCallback)
    start_server_function = inspect.getsource(progression_instrumentation.start_server)
    trainer_patch_function = inspect.getsource(progression_instrumentation.enable_tracking)

    # Build module-level state
    metrics_state_init = textwrap.dedent("""
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
    """).strip()

    # Build header with imports
    header = textwrap.dedent("""
        import http.server
        import json
        import threading
        import time
        from typing import Any, Optional

        from transformers import TrainerCallback
        from transformers import trainer as trainer_module

        print("[Kubeflow] Initializing progression tracking", flush=True)
    """).strip()

    # Build bootstrap code with custom metrics and port
    custom_metrics_json = json.dumps(custom_metrics)
    bootstrap_code = textwrap.dedent(f"""
        # Enable progression tracking
        # Server will start on rank-0 process only when training begins
        enable_tracking({custom_metrics_json}, metrics_port={metrics_port})
        print("[Kubeflow] Progression tracking enabled", flush=True)
    """).strip()

    return f"""
# =============================================================================
# Transformers Trainer SDK - INSTRUMENTATION WRAPPER
# Generated by kubeflow.trainer.rhai.transformers
# No SDK imports needed - all code is self-contained
# =============================================================================

{header}

# =============================================================================
# METRICS STATE - Module-level encapsulated state
# Shared across server and callback without global class variables
# =============================================================================

{metrics_state_init}

{update_metrics_function}

{get_metrics_json_function}

# =============================================================================
# METRICS SERVER - HTTP endpoint for controller to poll training progress
# Exposes /metrics endpoint with current step, loss, and custom metrics
# =============================================================================

{metrics_server_class}

{start_server_function}

# =============================================================================
# PROGRESS CALLBACK - Hooks into Transformers Trainer lifecycle events
# Automatically captures metrics during training and updates server state
# =============================================================================

{progress_callback_class}

# =============================================================================
# TRAINER MONKEY-PATCH - Injects callback into all Trainer instances
# Patches transformers.Trainer.__init__ to auto-register ProgressCallback
# =============================================================================

{trainer_patch_function}

# =============================================================================
# BOOTSTRAP - Start metrics server and enable tracking before user code runs
# =============================================================================

{bootstrap_code}

# =============================================================================
# USER TRAINING CODE
# =============================================================================

{{{{user_func_import_and_call}}}}
"""


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

    # Build base CustomTrainer-like configuration
    base_trainer = types.CustomTrainer(
        func=trainer.func,
        func_args=trainer.func_args,
        packages_to_install=trainer.packages_to_install,
        pip_index_urls=trainer.pip_index_urls,
        num_nodes=trainer.num_nodes,
        resources_per_node=trainer.resources_per_node,
        env=trainer.env,
    )

    # Get base CRD (without instrumentation)
    trainer_crd = utils.get_trainer_cr_from_custom_trainer(runtime, base_trainer)

    # Add progression tracking instrumentation if enabled
    if trainer.enable_progression_tracking:
        # Generate wrapper script
        wrapper_code = get_transformers_instrumentation_wrapper(
            metrics_port=trainer.metrics_port,
            custom_metrics=trainer.custom_metrics or {},
        )

        # Inject wrapper code into trainer command
        # Replace placeholder with {func_code} which gets substituted later
        wrapped_code = wrapper_code.replace("{{user_func_import_and_call}}", "{func_code}")
        trainer_crd.command = [
            cmd.replace("{func_code}", wrapped_code) if "{func_code}" in cmd else cmd
            for cmd in trainer_crd.command
        ]

        # Set annotations for controller
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
