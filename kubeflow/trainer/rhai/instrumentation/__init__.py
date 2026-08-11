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

"""Instrumentation code generation for RHAI trainers.

Submodules:
- speculator: Public API for SpeculativeDecodingTrainer pod script generation
- speculator_progression: Progression tracking injected into speculator pods
- traininghub: Public API for TrainingHubTrainer pod script generation
- transformers: Public API for TransformersTrainer pod script generation
- traininghub_codegen: Code generation helpers for TrainingHub scripts
- traininghub_progression: Progression tracking injected into TrainingHub pods
- transformers_checkpoint: Checkpoint instrumentation injected into Transformers pods
- transformers_progression: Progression tracking injected into Transformers pods
"""

from kubeflow.trainer.rhai.instrumentation.speculator import (
    get_speculator_instrumentation_wrapper,
)
from kubeflow.trainer.rhai.instrumentation.traininghub import (
    get_training_hub_instrumentation_wrapper,
)
from kubeflow.trainer.rhai.instrumentation.transformers import (
    get_jit_checkpoint_injection_code,
    get_transformers_instrumentation_wrapper,
)

__all__ = (
    "get_jit_checkpoint_injection_code",
    "get_speculator_instrumentation_wrapper",
    "get_training_hub_instrumentation_wrapper",
    "get_transformers_instrumentation_wrapper",
)
