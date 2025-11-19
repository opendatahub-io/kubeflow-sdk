# Copyright 2025 The Kubeflow Authors.
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

"""Red Hat AI (RHAI) trainer implementations.

This module provides RHAI trainer types and utilities:
- TransformersTrainer: HuggingFace Transformers/TRL with instrumentation
- TrainingHubTrainer: RHAI Training Hub integration
"""

from typing import Union

from kubeflow.trainer.rhai.traininghub import TrainingHubAlgorithms, TrainingHubTrainer
from kubeflow.trainer.rhai.transformers import TransformersTrainer

# Type alias for all RHAI trainers
RHAITrainer = Union[TransformersTrainer, TrainingHubTrainer]

__all__ = [
    "RHAITrainer",
    "TrainingHubAlgorithms",
    "TrainingHubTrainer",
    "TransformersTrainer",
]
