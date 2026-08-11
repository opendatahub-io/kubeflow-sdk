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

"""Tests for speculator_progression instrumentation module."""

import inspect

import pytest

from kubeflow.trainer.rhai.instrumentation.speculator_progression import (
    _create_speculator_progression_instrumentation,
)


class TestCreateSpeculatorProgressionInstrumentation:
    """Tests for _create_speculator_progression_instrumentation source introspection."""

    def test_function_is_inspectable(self) -> None:
        """inspect.getsource must succeed — used by get_speculator_instrumentation_wrapper."""
        source = inspect.getsource(_create_speculator_progression_instrumentation)
        assert source
        assert "def _create_speculator_progression_instrumentation" in source

    def test_source_is_valid_python(self) -> None:
        """Extracted source must compile without errors."""
        source = inspect.getsource(_create_speculator_progression_instrumentation)
        compile(source, "<_create_speculator_progression_instrumentation>", "exec")

    @pytest.mark.parametrize(
        "required_symbol",
        [
            "MetricsHandler",
            "SpeculatorMetricsHTTPHandler",
            "apply_progression_tracking",
            "_start_data_progress_server",
            "set_steps_per_epoch",
            "_mark_data_complete",
            "_set_phase",
        ],
    )
    def test_source_contains_required_symbols(self, required_symbol: str) -> None:
        """Source must define all symbols expected by the generated wrapper."""
        source = inspect.getsource(_create_speculator_progression_instrumentation)
        assert required_symbol in source

    def test_source_supports_all_modes(self) -> None:
        """Source must handle all speculator modes."""
        source = inspect.getsource(_create_speculator_progression_instrumentation)
        for mode in ("data_only", "train_only", "offline", "online"):
            assert mode in source, f"Mode {mode!r} missing from instrumentation source"

    def test_returns_five_element_tuple(self) -> None:
        """Function must return a 5-element tuple of callables."""
        result = _create_speculator_progression_instrumentation(
            metrics_port=0, mode="train_only", num_epochs=1
        )
        assert isinstance(result, tuple)
        assert len(result) == 5
        apply_fn, start_data_fn, handler_class, mark_data_fn, set_phase_fn = result
        assert callable(apply_fn)
        assert callable(start_data_fn)
        assert callable(mark_data_fn)
        assert callable(set_phase_fn)
