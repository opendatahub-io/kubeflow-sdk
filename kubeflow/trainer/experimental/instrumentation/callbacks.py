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

"""Callback and wrapper script generators for experimental trainers.

These functions generate Python code that is injected into training containers.
The generated code uses template files for better DX with IDE support.
"""

import re
from typing import Optional

from kubeflow.trainer.experimental.instrumentation.http_server import get_http_server_code
from kubeflow.trainer.experimental.instrumentation.templates import get_progress_callback_code


def _validate_metric_name(name: str) -> None:
    """Validate metric name is safe for code injection.

    Ensures the metric name:
    - Is a valid Python identifier
    - Contains no quotes or special characters that could break string interpolation
    - Is not empty or only whitespace

    Args:
        name: The metric name to validate.

    Raises:
        ValueError: If the metric name is invalid or unsafe for code injection.
    """
    if not name or not name.strip():
        raise ValueError("Metric name cannot be empty or whitespace")

    # Remove leading/trailing whitespace for validation
    name = name.strip()

    # Check for dangerous characters that could break string interpolation
    if '"' in name or "'" in name or "\\" in name:
        raise ValueError(
            f"Metric name contains invalid characters (quotes or backslashes): {name!r}"
        )

    # Check if it's a valid Python identifier (allows alphanumeric + underscore, no spaces)
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(
            f"Metric name must be a valid Python identifier (alphanumeric + underscore): {name!r}"
        )


def get_transformers_trainer_wrapper_script(
    metrics_port: int = 28080,
    custom_metrics: Optional[dict[str, str]] = None,
) -> str:
    """Generate wrapper script that auto-instruments Transformers Trainer.

    This function assembles a complete Python script from templates that:
    - Starts an HTTP metrics server
    - Adds progress tracking callbacks
    - Monkey-patches Transformers Trainer to inject callbacks

    Args:
        metrics_port: Port for HTTP metrics server.
        custom_metrics: Dict mapping log keys to metric names to track.
                       Example: {"eval_accuracy": "accuracy", "custom_loss": "custom_loss"}

    Returns:
        Complete Python script as string for injection into training container.

    Raises:
        ValueError: If any custom metric name is invalid or unsafe for code injection.
    """
    # Validate custom metrics before processing
    if custom_metrics:
        for metric_name in custom_metrics.values():
            _validate_metric_name(metric_name)

    # Get code components from templates
    http_server_code = get_http_server_code(port=metrics_port)
    progress_callback_code = get_progress_callback_code(custom_metrics=custom_metrics)

    # Build custom metrics initialization
    custom_metrics_init = ""
    if custom_metrics:
        custom_metrics_init = "\n# Initialize custom metrics\n"
        for metric_name in custom_metrics.values():
            custom_metrics_init += (
                f'ProgressMetricsHandler.update_metrics({{"metrics": {{"{metric_name}": None}}}})\n'
            )

    # Assemble the complete wrapper script
    wrapper_script = f"""
import sys
import time
from transformers import TrainerCallback

{http_server_code}

# Start metrics server
metrics_server = start_metrics_server(port={metrics_port})
{custom_metrics_init}

# Import progress callback
{progress_callback_code}

# Monkey-patch Transformers Trainer to inject callbacks
from transformers import trainer as trainer_module

_original_init = trainer_module.Trainer.__init__

def _instrumented_init(self, *args, **kwargs):
    callbacks = kwargs.get('callbacks', [])
    callbacks.append(KubeflowProgressCallback())
    kwargs['callbacks'] = callbacks
    _original_init(self, *args, **kwargs)

trainer_module.Trainer.__init__ = _instrumented_init

{{{{user_func_import_and_call}}}}
"""

    return wrapper_script
