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

"""Experimental trainer type definitions."""

from dataclasses import dataclass, field
from typing import Callable, Optional

from kubeflow.trainer.constants import constants


@dataclass
class TransformersTrainer:
    """Experimental trainer for HuggingFace Transformers with auto-instrumentation.

    EXPERIMENTAL: This API may change in future releases.

    Provides automatic instrumentation for:
    - Training progression tracking via HTTP metrics endpoint
    - Custom metrics tracking via callback

    When to use:
    - You need real-time progress tracking in the UI
    - You're using HuggingFace Transformers
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

    Example:
        ```python
        from kubeflow.trainer.types.experimental import TransformersTrainer


        def train():
            from transformers import Trainer, TrainingArguments

            # Your training code here
            trainer.train()


        client.train(
            trainer=TransformersTrainer(
                func=train,
                num_nodes=2,
                resources_per_node={"gpu": 1},
                enable_progression_tracking=True,
                custom_metrics={
                    "eval_accuracy": "accuracy",
                    "eval_f1": "f1_score",
                },
            ),
        )
        ```
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

    # Experimental instrumentation features
    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30
    custom_metrics: Optional[dict[str, str]] = field(default_factory=dict)
