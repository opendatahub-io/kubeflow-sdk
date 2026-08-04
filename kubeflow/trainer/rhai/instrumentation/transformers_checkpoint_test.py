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

"""Tests for transformers_checkpoint_instrumentation module."""

import inspect

import pytest

from kubeflow.trainer.rhai.instrumentation.transformers_checkpoint import (
    _create_checkpoint_instrumentation,
)


class TestCreateCheckpointInstrumentation:
    """Tests for _create_checkpoint_instrumentation source introspection."""

    def test_function_is_inspectable(self) -> None:
        """inspect.getsource must succeed — used by get_jit_checkpoint_injection_code."""
        source = inspect.getsource(_create_checkpoint_instrumentation)
        assert source
        assert "def _create_checkpoint_instrumentation" in source

    def test_source_is_valid_python(self) -> None:
        """Extracted source must compile without errors."""
        source = inspect.getsource(_create_checkpoint_instrumentation)
        compile(source, "<_create_checkpoint_instrumentation>", "exec")

    def test_source_contains_constants_import(self) -> None:
        """Source must contain the constants import that get_jit_checkpoint_injection_code strips."""
        source = inspect.getsource(_create_checkpoint_instrumentation)
        assert (
            "from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER, "
            "CHECKPOINT_STAGING_DIR"
        ) in source

    @pytest.mark.parametrize(
        "required_symbol",
        [
            "CheckpointManager",
            "apply_checkpointing",
            "upload_final_model_to_cloud",
        ],
    )
    def test_source_contains_required_symbols(self, required_symbol: str) -> None:
        """Source must define all symbols expected by the generated pod script."""
        source = inspect.getsource(_create_checkpoint_instrumentation)
        assert required_symbol in source
