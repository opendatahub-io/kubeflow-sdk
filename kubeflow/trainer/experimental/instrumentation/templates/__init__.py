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

"""Template files for code injection into training pods.

These templates are proper Python files with IDE support for better DX.
They are read and rendered with placeholders replaced at runtime.
"""

from pathlib import Path
from typing import Optional


def get_progress_callback_code(custom_metrics: Optional[dict[str, str]] = None) -> str:
    """Get progress callback code with custom metrics support.

    Args:
        custom_metrics: Dict mapping log keys to metric names.

    Returns:
        Rendered Python code.
    """
    template_path = Path(__file__).parent / "progress_callback.py"
    code = template_path.read_text()

    # Replace custom metrics placeholder
    custom_metrics_str = repr(custom_metrics or {})
    code = code.replace("__CUSTOM_METRICS_PLACEHOLDER__", custom_metrics_str)

    return code


__all__ = [
    "get_progress_callback_code",
]
