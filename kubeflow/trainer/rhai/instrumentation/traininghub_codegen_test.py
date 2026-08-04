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

"""Tests for traininghub_codegen module."""

from types import SimpleNamespace

import pytest

from kubeflow.trainer.rhai.instrumentation.traininghub_codegen import (
    _get_command_from_runtime,
    _render_algorithm_wrapper,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


class TestRenderAlgorithmWrapper:
    """Tests for _render_algorithm_wrapper helper."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="sft algorithm no func_args",
                expected_status=SUCCESS,
                config={
                    "algorithm_metadata": {"name": "sft", "metrics_file_rank0": "metrics.jsonl"},
                    "func_args": None,
                },
                expected_output=None,
            ),
            TestCase(
                name="sft algorithm with func_args",
                expected_status=SUCCESS,
                config={
                    "algorithm_metadata": {"name": "sft", "metrics_file_rank0": "metrics.jsonl"},
                    "func_args": {"model": "llama", "epochs": 3},
                },
                expected_output=None,
            ),
        ],
    )
    def test_render_algorithm_wrapper(self, test_case: TestCase) -> None:
        print(f"Running test: {test_case.name}")
        if test_case.expected_status == FAILED:
            with pytest.raises(ValueError):
                _render_algorithm_wrapper(
                    test_case.config["algorithm_metadata"],
                    test_case.config["func_args"],
                )
        else:
            result = _render_algorithm_wrapper(
                test_case.config["algorithm_metadata"],
                test_case.config["func_args"],
            )
            assert isinstance(result, str)
            assert "training_func" in result
            assert (
                f"from training_hub import {test_case.config['algorithm_metadata']['name']}"
                in result
            )
            compile(result, "<wrapper>", "exec")


class TestGetCommandFromRuntime:
    """Tests for _get_command_from_runtime helper."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCase(
                name="injects func_file into template element",
                expected_status=SUCCESS,
                config={
                    "command": ["python", "-c {func_file}"],
                    "func_code": "def train(): pass",
                    "func_file": "train.py",
                    "install_snippet": "",
                },
                expected_output=["python", "-c train.py"],
            ),
            TestCase(
                name="prepends install_snippet when present",
                expected_status=SUCCESS,
                config={
                    "command": ["python {func_file}"],
                    "func_code": "def train(): pass",
                    "func_file": "train.py",
                    "install_snippet": "pip install torch\n",
                },
                expected_output=["pip install torch\npython train.py"],
            ),
            TestCase(
                name="passthrough elements without placeholder",
                expected_status=SUCCESS,
                config={
                    "command": ["--verbose", "python {func_file}"],
                    "func_code": "def train(): pass",
                    "func_file": "train.py",
                    "install_snippet": "",
                },
                expected_output=["--verbose", "python train.py"],
            ),
        ],
    )
    def test_get_command_from_runtime(self, test_case: TestCase) -> None:
        print(f"Running test: {test_case.name}")
        runtime = SimpleNamespace(trainer=SimpleNamespace(command=test_case.config["command"]))
        result = _get_command_from_runtime(
            runtime,
            func_code=test_case.config["func_code"],
            func_file=test_case.config["func_file"],
            install_snippet=test_case.config["install_snippet"],
        )
        assert result == test_case.expected_output
