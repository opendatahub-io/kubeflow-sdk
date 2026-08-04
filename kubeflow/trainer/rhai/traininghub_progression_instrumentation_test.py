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

"""Tests for traininghub_progression_instrumentation module."""

import inspect

import pytest

from kubeflow.trainer.rhai.traininghub_progression_instrumentation import (
    _create_training_hub_progression_instrumentation,
)


class TestCreateTrainingHubProgressionInstrumentation:
    """Tests for _create_training_hub_progression_instrumentation source introspection."""

    def test_function_is_inspectable(self) -> None:
        """inspect.getsource must succeed — used by get_training_hub_instrumentation_wrapper."""
        source = inspect.getsource(_create_training_hub_progression_instrumentation)
        assert source
        assert "def _create_training_hub_progression_instrumentation" in source

    def test_source_is_valid_python(self) -> None:
        """Extracted source must compile without errors."""
        source = inspect.getsource(_create_training_hub_progression_instrumentation)
        compile(source, "<_create_training_hub_progression_instrumentation>", "exec")

    @pytest.mark.parametrize(
        "required_symbol",
        [
            "TrainingHubMetricsHandler",
            "apply_progression_tracking",
            "_read_latest_metrics",
            "_transform_schema",
        ],
    )
    def test_source_contains_required_symbols(self, required_symbol: str) -> None:
        """Source must define all symbols expected by the generated wrapper."""
        source = inspect.getsource(_create_training_hub_progression_instrumentation)
        assert required_symbol in source

    def test_source_contains_metrics_endpoint(self) -> None:
        """HTTP handler must serve the /metrics path."""
        source = inspect.getsource(_create_training_hub_progression_instrumentation)
        assert '"/metrics"' in source

    def test_source_supports_all_algorithms(self) -> None:
        """Source must handle sft, osft, lora_sft, and lora_grpo algorithms."""
        source = inspect.getsource(_create_training_hub_progression_instrumentation)
        for algo in ("sft", "osft", "lora_sft", "lora_grpo"):
            assert algo in source, f"Algorithm {algo!r} missing from instrumentation source"
