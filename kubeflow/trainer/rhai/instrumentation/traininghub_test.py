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

"""Tests for TrainingHub instrumentation functions."""

import pytest

from kubeflow.trainer.rhai.instrumentation.traininghub_codegen import (
    _render_algorithm_wrapper,
    _render_user_func_code,
)
from kubeflow.trainer.rhai.traininghub import get_training_hub_instrumentation_wrapper
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="algorithm wrapper contains training_func call with kwargs",
            expected_status=SUCCESS,
            config={
                "algorithm_metadata": {
                    "name": "sft",
                    "metrics_file_rank0": "metrics_rank0.json",
                },
                "func_args": {"model_name": "test-model"},
            },
            expected_output="training_func",
        ),
        TestCase(
            name="algorithm wrapper with None func_args uses empty dict",
            expected_status=SUCCESS,
            config={
                "algorithm_metadata": {
                    "name": "osft",
                    "metrics_file_rank0": "metrics_rank0.json",
                },
                "func_args": None,
            },
            expected_output="training_func({})",
        ),
        TestCase(
            name="algorithm wrapper contains termination message function",
            expected_status=SUCCESS,
            config={
                "algorithm_metadata": {
                    "name": "lora_sft",
                    "metrics_file_rank0": "metrics_rank0.json",
                },
                "func_args": None,
            },
            expected_output="_write_termination_message",
        ),
    ],
)
def test_render_algorithm_wrapper(test_case: TestCase) -> None:
    """Verify algorithm wrappers contain expected training_hub calls."""
    print(f"Executing test: {test_case.name}")
    result = _render_algorithm_wrapper(
        test_case.config["algorithm_metadata"],
        test_case.config["func_args"],
    )
    assert test_case.expected_output in result


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="algorithm wrapper includes metrics file from metadata",
            expected_status=SUCCESS,
            config={
                "algorithm_metadata": {
                    "name": "sft",
                    "metrics_file_rank0": "custom_metrics.json",
                },
                "func_args": None,
            },
            expected_output="custom_metrics.json",
        ),
        TestCase(
            name="algorithm wrapper with None metrics_file_rank0 still generates script",
            expected_status=SUCCESS,
            config={
                "algorithm_metadata": {
                    "name": "sft",
                    "metrics_file_rank0": None,
                },
                "func_args": None,
            },
            expected_output="metrics_file_rank0 = None",
        ),
    ],
)
def test_render_algorithm_wrapper_metadata(test_case: TestCase) -> None:
    """Verify algorithm wrappers use metadata fields correctly."""
    print(f"Executing test: {test_case.name}")
    result = _render_algorithm_wrapper(
        test_case.config["algorithm_metadata"],
        test_case.config["func_args"],
    )
    assert test_case.expected_output in result


def _sample_train_func():
    """Sample training function for testing."""
    print("training")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="user func code with no args calls function directly",
            expected_status=SUCCESS,
            config={"func": _sample_train_func, "func_args": None},
            expected_output="_sample_train_func()",
        ),
        TestCase(
            name="user func code with dict args uses kwargs unpacking",
            expected_status=SUCCESS,
            config={"func": _sample_train_func, "func_args": {"lr": 0.01}},
            expected_output="_sample_train_func(**{",
        ),
    ],
)
def test_render_user_func_code(test_case: TestCase) -> None:
    """Verify user function code rendering."""
    print(f"Executing test: {test_case.name}")
    func_code, func_file = _render_user_func_code(
        test_case.config["func"],
        test_case.config["func_args"],
    )
    assert test_case.expected_output in func_code
    assert func_file.endswith(".py")


def test_render_user_func_code_returns_source() -> None:
    """Rendered code must contain the original function body."""
    print("Executing test: render_user_func_code_returns_source")
    func_code, _ = _render_user_func_code(_sample_train_func, None)
    assert 'print("training")' in func_code


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="non-callable raises ValueError",
            expected_status=FAILED,
            config={"func": "not_a_function", "func_args": None, "match": "callable"},
            expected_error=ValueError,
        ),
        TestCase(
            name="integer raises ValueError",
            expected_status=FAILED,
            config={"func": 42, "func_args": None, "match": "callable"},
            expected_error=ValueError,
        ),
    ],
)
def test_render_user_func_code_invalid_input(test_case: TestCase) -> None:
    """Non-callable input must raise ValueError."""
    print(f"Executing test: {test_case.name}")
    with pytest.raises(test_case.expected_error, match=test_case.config["match"]):
        _render_user_func_code(
            test_case.config["func"],
            test_case.config["func_args"],
        )


def test_render_algorithm_wrapper_output_compiles() -> None:
    """Generated algorithm wrapper script must be valid Python."""
    print("Executing test: render_algorithm_wrapper_output_compiles")
    metadata = {"name": "sft", "metrics_file_rank0": "metrics_rank0.json"}
    code = _render_algorithm_wrapper(metadata, {"model_name": "test-model"})
    compile(code, "<generated>", "exec")


def test_get_training_hub_instrumentation_wrapper_compiles() -> None:
    """Generated instrumentation wrapper must be valid Python."""
    print("Executing test: get_training_hub_instrumentation_wrapper_compiles")
    wrapper = get_training_hub_instrumentation_wrapper("sft", "/tmp/ckpts", 28080)
    compile(wrapper, "<generated>", "exec")
