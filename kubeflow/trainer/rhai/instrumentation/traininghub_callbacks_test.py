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

"""Tests for Training Hub callback serialization and pod injection."""

import pytest

from kubeflow.trainer.rhai.instrumentation.traininghub_callbacks import (
    build_training_hub_callback_injection_code,
    validate_callbacks,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase

try:
    from training_hub import TrainingHubCallback, TrainingHubContext
except ImportError:

    class TrainingHubCallback:
        """Test stub when training_hub is not installed."""

    class TrainingHubContext:
        """Test stub when training_hub is not installed."""


class LoggingCallback(TrainingHubCallback):
    """Module-level callback fixture for serialization tests."""

    def on_log(self, context: TrainingHubContext) -> None:
        print(f"Loss at step {context.step}: {context.loss}")

    def on_train_begin(self, context: TrainingHubContext) -> None:
        print("Training started")


class EarlyStopCallback(TrainingHubCallback):
    """Module-level callback fixture for multi-callback serialization tests."""

    def on_step_end(self, context: TrainingHubContext) -> None:
        if context.loss and context.loss < 0.01:
            print("Early stop threshold reached")


def test_validate_callbacks_accepts_classes_and_instances():
    """validate_callbacks accepts callback classes and instances."""
    print("Executing test: validate_callbacks accepts classes and instances")

    validate_callbacks([LoggingCallback])
    validate_callbacks([LoggingCallback()])
    validate_callbacks(None)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="callbacks must be a list",
            expected_status=FAILED,
            config={"callbacks": LoggingCallback},
            expected_error=TypeError,
        ),
        TestCase(
            name="callbacks reject invalid entries",
            expected_status=FAILED,
            config={"callbacks": ["not-a-callback"]},
            expected_error=TypeError,
        ),
    ],
)
def test_validate_callbacks_rejects_invalid_input(test_case):
    """validate_callbacks rejects invalid callback containers."""
    print(f"Executing test: {test_case.name}")

    try:
        validate_callbacks(test_case.config["callbacks"])
        if test_case.expected_status == FAILED:
            raise AssertionError("Expected validation to fail")
    except Exception as exc:
        if test_case.expected_status == SUCCESS:
            raise
        assert type(exc) is test_case.expected_error

    print("test execution complete")


def test_build_callback_injection_code_single_callback():
    """Single callback class is serialized and instantiated in pod preamble."""
    print("Executing test: single callback injection code generation")

    code = build_training_hub_callback_injection_code([LoggingCallback])

    assert "class LoggingCallback" in code
    assert "_KUBEFLOW_HUB_CALLBACKS = [LoggingCallback()]" in code
    assert "from training_hub import TrainingHubCallback, TrainingHubContext" in code
    assert "_kubeflow_wrap_training_hub_api" in code
    assert "lora_grpo" in code

    print("test execution complete")


def test_build_callback_injection_code_multiple_callbacks():
    """Multiple callbacks are serialized and instantiated."""
    print("Executing test: multiple callbacks injection code generation")

    code = build_training_hub_callback_injection_code([LoggingCallback, EarlyStopCallback])

    assert "class LoggingCallback" in code
    assert "class EarlyStopCallback" in code
    assert "LoggingCallback(), EarlyStopCallback()" in code

    print("test execution complete")


def test_build_callback_injection_code_accepts_instance():
    """Passing an instance serializes its class instead."""
    print("Executing test: callback instance serializes class")

    code = build_training_hub_callback_injection_code([LoggingCallback()])

    assert "class LoggingCallback" in code
    assert "LoggingCallback()" in code

    print("test execution complete")


def test_build_callback_injection_code_empty_list():
    """Empty callback list returns empty injection code."""
    print("Executing test: empty callbacks returns empty code")

    assert build_training_hub_callback_injection_code([]) == ""

    print("test execution complete")
