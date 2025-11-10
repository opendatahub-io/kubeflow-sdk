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

"""Experimental trainer types with auto-instrumentation.

EXPERIMENTAL: These APIs are subject to change in future releases.

This module provides experimental trainer types that automatically add:
- Training progression tracking via HTTP metrics endpoint

These features are currently midstream-only and may be upstreamed in the future.
"""

from kubeflow.trainer.types.experimental.types import TransformersTrainer

__all__ = [
    "TransformersTrainer",
]
