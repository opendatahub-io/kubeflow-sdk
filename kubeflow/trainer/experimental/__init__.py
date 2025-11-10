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

"""Experimental features for Kubeflow Trainer SDK.

EXPERIMENTAL: All code in this module is subject to change and is not
guaranteed to be compatible with upstream Kubeflow releases.

This module contains midstream-specific implementations including:
- Progression tracking via HTTP metrics
- JIT checkpointing
- Framework-specific instrumentation
- Real-time training progress monitoring
"""

from kubeflow.trainer.experimental.backends.kubernetes import ExperimentalKubernetesBackend
from kubeflow.trainer.experimental.utils import get_job_progress

__all__ = [
    "ExperimentalKubernetesBackend",
    "get_job_progress",
]
