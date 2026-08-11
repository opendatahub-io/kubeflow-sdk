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


class NotACallback:
    """Module-level non-callback class for validation tests."""

    def on_log(self, context: TrainingHubContext) -> None:
        print("not a callback")


_CALLBACK_THRESHOLD = 0.01


class CallbackWithModuleDependency(TrainingHubCallback):
    """Callback that references a module-level constant."""

    def on_log(self, context: TrainingHubContext) -> None:
        if context.loss and context.loss < _CALLBACK_THRESHOLD:
            print("threshold reached")


class CallbackWithUnknownHook(TrainingHubCallback):
    """Callback that defines a hook not in the unified 9."""

    def on_prediction_step(self, context: TrainingHubContext) -> None:
        print("prediction step")


def test_validate_callbacks_accepts_callback_classes():
    """validate_callbacks accepts callback classes."""
    print("Executing test: validate_callbacks accepts callback classes")

    validate_callbacks([LoggingCallback])
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
        TestCase(
            name="callbacks reject instances",
            expected_status=FAILED,
            config={"callbacks": [LoggingCallback()]},
            expected_error=TypeError,
        ),
        TestCase(
            name="callbacks reject non-callback classes",
            expected_status=FAILED,
            config={"callbacks": [NotACallback]},
            expected_error=TypeError,
        ),
        TestCase(
            name="callbacks reject duplicate class names",
            expected_status=FAILED,
            config={"callbacks": [LoggingCallback, LoggingCallback]},
            expected_error=ValueError,
        ),
        TestCase(
            name="callbacks reject module-level dependencies",
            expected_status=FAILED,
            config={"callbacks": [CallbackWithModuleDependency]},
            expected_error=ValueError,
        ),
        TestCase(
            name="callbacks reject unsupported hooks",
            expected_status=FAILED,
            config={"callbacks": [CallbackWithUnknownHook]},
            expected_error=ValueError,
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


def test_generated_callback_injection_code_executes():
    """Generated pod preamble defines and instantiates callbacks without NameError."""
    print("Executing test: generated callback injection code executes")

    import sys
    import types

    training_hub = types.ModuleType("training_hub")

    class _TrainingHubCallback:
        pass

    class _TrainingHubContext:
        pass

    def _noop_api(**kwargs: object) -> None:
        return None

    training_hub.TrainingHubCallback = _TrainingHubCallback
    training_hub.TrainingHubContext = _TrainingHubContext
    training_hub.sft = _noop_api
    training_hub.osft = _noop_api
    training_hub.lora_sft = _noop_api
    training_hub.lora_grpo = _noop_api

    code = build_training_hub_callback_injection_code([LoggingCallback])
    namespace: dict[str, object] = {}
    sys.modules["training_hub"] = training_hub
    try:
        exec(code, namespace)  # noqa: S102
    finally:
        sys.modules.pop("training_hub", None)

    callbacks = namespace["_KUBEFLOW_HUB_CALLBACKS"]
    assert len(callbacks) == 1
    assert callbacks[0].__class__.__name__ == "LoggingCallback"

    print("test execution complete")


def test_build_callback_injection_code_empty_list():
    """Empty callback list returns empty injection code."""
    print("Executing test: empty callbacks returns empty code")

    assert build_training_hub_callback_injection_code([]) == ""

    print("test execution complete")
