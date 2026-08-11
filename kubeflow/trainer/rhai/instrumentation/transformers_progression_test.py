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

"""Tests for transformers_progression_instrumentation module."""

import inspect

import pytest

from kubeflow.trainer.rhai.instrumentation.transformers_progression import (
    _create_progression_instrumentation,
)


class TestCreateProgressionInstrumentation:
    """Tests for _create_progression_instrumentation source introspection."""

    def test_function_is_inspectable(self) -> None:
        """inspect.getsource must succeed — used by get_transformers_instrumentation_wrapper."""
        source = inspect.getsource(_create_progression_instrumentation)
        assert source
        assert "def _create_progression_instrumentation" in source

    def test_source_is_valid_python(self) -> None:
        """Extracted source must compile without errors."""
        source = inspect.getsource(_create_progression_instrumentation)
        compile(source, "<_create_progression_instrumentation>", "exec")

    @pytest.mark.parametrize(
        "required_symbol",
        [
            "KubeflowProgressCallback",
            "ProgressionMetricsHandler",
            "apply_progression_tracking",
            "_update_progression_metrics",
        ],
    )
    def test_source_contains_required_symbols(self, required_symbol: str) -> None:
        """Source must define all symbols expected by the generated training wrapper."""
        source = inspect.getsource(_create_progression_instrumentation)
        assert required_symbol in source
