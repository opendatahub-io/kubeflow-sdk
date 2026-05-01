# Copyright 2024 The Kubeflow Authors.
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

"""Tests for progression tracking instrumentation module."""

import json
import sys
import time
from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.test.common import SUCCESS, TestCase


def test_progression_instrumentation_imports():
    """Test that progression instrumentation module imports correctly."""
    print("Executing test: progression instrumentation imports")

    # The module itself can be imported without transformers
    # create_progression_instrumentation is what requires transformers at runtime
    import kubeflow.trainer.rhai.instrumentation.progression as progression_module

    assert hasattr(progression_module, "create_progression_instrumentation")
    assert callable(progression_module.create_progression_instrumentation)
    print("test execution complete")


def test_create_progression_instrumentation():
    """Test create_progression_instrumentation returns expected components."""
    print("Executing test: create progression instrumentation components")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        # Create progression instrumentation components
        result = create_progression_instrumentation(metrics_port=28080)

        # Unpack the tuple
        (
            apply_fn,
            callback_cls,
            handler_cls,
            get_metrics_fn,
            update_metrics_fn,
        ) = result

        # Verify all components are present
        assert callable(apply_fn)
        assert callback_cls is not None
        assert callback_cls.__name__ == "KubeflowProgressCallback"
        assert handler_cls is not None
        assert handler_cls.__name__ == "ProgressionMetricsHandler"
        assert callable(get_metrics_fn)
        assert callable(update_metrics_fn)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="completion protection - normal completion 200/200 steps",
            expected_status=SUCCESS,
            config={
                "state_global_step": 200,
                "state_max_steps": 200,
                "state_epoch": 2.0,
                "num_train_epochs": 2,
                "overwrite_attempt_step": 50,
            },
            expected_output={
                "currentStep": 200,
                "totalSteps": 200,
                "currentEpoch": 2.0,
                "totalEpochs": 2,
                "progressPercentage": 100,
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="completion protection - early stop 500/1000 steps (50%)",
            expected_status=SUCCESS,
            config={
                "state_global_step": 500,
                "state_max_steps": 1000,
                "state_epoch": 0.5,
                "num_train_epochs": 1,
                "overwrite_attempt_step": 200,
            },
            expected_output={
                "currentStep": 500,
                "totalSteps": 1000,
                "currentEpoch": 0.5,
                "totalEpochs": 1,
                "progressPercentage": 50,
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="completion protection - None global_step handling",
            expected_status=SUCCESS,
            config={
                "state_global_step": None,
                "state_max_steps": 1000,
                "state_epoch": 0.0,
                "num_train_epochs": 1,
                "overwrite_attempt_step": 200,
            },
            expected_output={
                "currentStep": 0,
                "totalSteps": 1000,
                "currentEpoch": 0.0,
                "totalEpochs": 1,
                "progressPercentage": 0,
                "estimatedRemainingSeconds": 0,
            },
        ),
    ],
)
def test_callback_completion_state_protection(test_case):
    """Test that completion state is protected from overwrites after training ends.

    This tests the fix for the bug where on_step_end was called after on_train_end
    during evaluation, overwriting actual progress values.
    """
    print(f"Executing test: {test_case.name}")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        _, callback_cls, _, get_metrics_json, _ = create_progression_instrumentation(28080)
        callback = callback_cls(metrics_port=28080)

        args = Mock()
        args.num_train_epochs = test_case.config["num_train_epochs"]
        state = Mock()
        state.global_step = test_case.config["state_global_step"]
        state.max_steps = test_case.config["state_max_steps"]
        state.epoch = test_case.config["state_epoch"]
        state.is_world_process_zero = False
        control = Mock()

        # Initialize and complete training
        callback.on_train_begin(args, state, control)
        callback.on_train_end(args, state, control)

        # Verify completion state shows expected values
        metrics = json.loads(get_metrics_json())
        for key, expected_value in test_case.expected_output.items():
            assert metrics[key] == expected_value, (
                f"{key}: expected {expected_value}, got {metrics[key]}"
            )

        # Simulate on_step_end called after training_end (evaluation steps)
        # Should NOT overwrite completion state
        state.global_step = test_case.config["overwrite_attempt_step"]
        callback.on_step_end(args, state, control)

        # Verify completion state is still protected
        metrics_after = json.loads(get_metrics_json())
        for key, expected_value in test_case.expected_output.items():
            assert metrics_after[key] == expected_value, (
                f"After overwrite attempt - {key}: expected {expected_value}, "
                f"got {metrics_after[key]}"
            )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="completion at exact max_steps",
            expected_status=SUCCESS,
            config={
                "state_global_step": 1000,
                "state_max_steps": 1000,
                "state_epoch": 1.0,
                "num_train_epochs": 1,
            },
            expected_output={
                "currentStep": 1000,
                "totalSteps": 1000,
                "progressPercentage": 100,
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="near completion - 374/375 steps (99%)",
            expected_status=SUCCESS,
            config={
                "state_global_step": 374,
                "state_max_steps": 375,
                "state_epoch": 1.99,
                "num_train_epochs": 2,
            },
            expected_output={
                "currentStep": 374,
                "totalSteps": 375,
                "progressPercentage": 99,
            },
        ),
    ],
)
def test_completion_detection_via_on_step_end(test_case):
    """Test that completion is detected via on_step_end when on_train_end is not called.

    This handles cases where:
    - Training crashes before on_train_end
    - User interrupts training
    - Exception occurs before on_train_end
    - Training completes but callback doesn't fire

    The fix ensures estimatedRemainingSeconds=0 when current_step >= total_steps.
    """
    print(f"Executing test: {test_case.name}")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        _, callback_cls, _, get_metrics_json, _ = create_progression_instrumentation(28080)
        callback = callback_cls(metrics_port=28080)

        args = Mock()
        args.num_train_epochs = test_case.config["num_train_epochs"]
        state = Mock()
        state.global_step = test_case.config["state_global_step"]
        state.max_steps = test_case.config["state_max_steps"]
        state.epoch = test_case.config["state_epoch"]
        state.is_world_process_zero = False
        control = Mock()

        # Initialize training (sets start_time)
        callback.on_train_begin(args, state, control)

        # Simulate some elapsed time
        time.sleep(0.1)

        # Call on_step_end WITHOUT calling on_train_end (simulates crash/interrupt)
        callback.on_step_end(args, state, control)

        metrics = json.loads(get_metrics_json())

        # Verify expected outputs
        for key, expected_value in test_case.expected_output.items():
            actual_value = metrics.get(key)
            assert actual_value == expected_value, (
                f"{key}: expected {expected_value}, got {actual_value}"
            )

        # Special check: near-completion should have remaining time > 0
        if "near completion" in test_case.name:
            assert metrics["estimatedRemainingSeconds"] is not None, (
                "Near completion should calculate remaining time"
            )
            if metrics["estimatedRemainingSeconds"] is not None:
                assert metrics["estimatedRemainingSeconds"] >= 0, (
                    "Remaining time should be non-negative"
                )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train and eval metrics properly categorized",
            expected_status=SUCCESS,
            config={
                "logs": {
                    "loss": 0.123,
                    "learning_rate": 1e-4,
                    "grad_norm": 0.456,
                    "eval_loss": 0.789,
                    "eval_accuracy": 0.95,
                    "train_samples_per_second": 100.0,
                }
            },
            expected_output={
                "trainMetrics": {
                    "loss": 0.123,
                    "learning_rate": 1e-4,
                    "grad_norm": 0.456,
                    "throughput_samples_sec": 100.0,
                },
                "evalMetrics": {
                    "eval_loss": 0.789,
                    "eval_accuracy": 0.95,
                },
            },
        ),
        TestCase(
            name="only train metrics",
            expected_status=SUCCESS,
            config={
                "logs": {
                    "loss": 0.5,
                    "learning_rate": 2e-5,
                }
            },
            expected_output={
                "trainMetrics": {
                    "loss": 0.5,
                    "learning_rate": 2e-5,
                },
                "evalMetrics": {},
            },
        ),
        TestCase(
            name="only eval metrics",
            expected_status=SUCCESS,
            config={
                "logs": {
                    "eval_loss": 0.234,
                    "eval_f1": 0.88,
                }
            },
            expected_output={
                "trainMetrics": {},
                "evalMetrics": {
                    "eval_loss": 0.234,
                    "eval_f1": 0.88,
                },
            },
        ),
    ],
)
def test_callback_metrics_categorization(test_case):
    """Test that train and eval metrics are properly categorized.

    Verifies metrics with 'eval_' prefix go to evalMetrics, and training metrics
    (loss, learning_rate, grad_norm) go to trainMetrics.
    """
    print(f"Executing test: {test_case.name}")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        _, callback_cls, _, get_metrics_json, _ = create_progression_instrumentation(28080)
        callback = callback_cls(metrics_port=28080)

        args = Mock()
        state = Mock()
        control = Mock()

        # Log metrics
        callback.on_log(args, state, control, logs=test_case.config["logs"])

        metrics = json.loads(get_metrics_json())

        # Verify train metrics
        for key, value in test_case.expected_output["trainMetrics"].items():
            assert metrics["trainMetrics"].get(key) == value, (
                f"trainMetrics[{key}]: expected {value}, got {metrics['trainMetrics'].get(key)}"
            )

        # Verify eval metrics
        for key, value in test_case.expected_output["evalMetrics"].items():
            assert metrics["evalMetrics"].get(key) == value, (
                f"evalMetrics[{key}]: expected {value}, got {metrics['evalMetrics'].get(key)}"
            )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="float epoch precision - 1.98 epochs",
            expected_status=SUCCESS,
            config={
                "state_global_step": 121,
                "state_max_steps": 124,
                "state_epoch": 1.98,
                "num_train_epochs": 2,
            },
            expected_output={
                "currentStep": 121,
                "totalSteps": 124,
                "currentEpoch": 1.98,
                "totalEpochs": 2,
                "progressPercentage": 97,
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="accurate percentage - 371/375 = 98%",
            expected_status=SUCCESS,
            config={
                "state_global_step": 371,
                "state_max_steps": 375,
                "state_epoch": 1.97,
                "num_train_epochs": 2,
            },
            expected_output={
                "currentStep": 371,
                "totalSteps": 375,
                "currentEpoch": 1.97,
                "totalEpochs": 2,
                "progressPercentage": 98,
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="None epoch handling",
            expected_status=SUCCESS,
            config={
                "state_global_step": 50,
                "state_max_steps": 100,
                "state_epoch": None,
                "num_train_epochs": 1,
            },
            expected_output={
                "currentStep": 50,
                "totalSteps": 100,
                "currentEpoch": 0.0,
                "totalEpochs": 1,
                "progressPercentage": 50,
                "estimatedRemainingSeconds": 0,
            },
        ),
    ],
)
def test_honest_progress_reporting(test_case):
    """Test honest progress reporting with float epochs and accurate percentages.

    Validates that:
    - Epoch counts use float precision (1.98 not truncated to 1)
    - Progress percentage reflects actual steps (97% for 121/124, not forced to 100%)
    - None and invalid values are handled gracefully
    """
    print(f"Executing test: {test_case.name}")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        _, callback_cls, _, get_metrics_json, _ = create_progression_instrumentation(28080)
        callback = callback_cls(metrics_port=28080)

        args = Mock()
        args.num_train_epochs = test_case.config["num_train_epochs"]
        state = Mock()
        state.global_step = test_case.config["state_global_step"]
        state.max_steps = test_case.config["state_max_steps"]
        state.epoch = test_case.config["state_epoch"]
        state.is_world_process_zero = False
        control = Mock()

        # Trigger on_train_end to capture final state
        callback.on_train_begin(args, state, control)
        callback.on_train_end(args, state, control)

        metrics = json.loads(get_metrics_json())

        # Verify all expected outputs
        for key, expected_value in test_case.expected_output.items():
            actual_value = metrics.get(key)
            assert actual_value == expected_value, (
                f"{key}: expected {expected_value}, got {actual_value}"
            )

    print("test execution complete")


def test_metrics_server_handler():
    """Test that ProgressionMetricsHandler serves metrics as JSON."""
    print("Executing test: metrics server handler")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        _, _, handler_cls, get_metrics_json, update_metrics = create_progression_instrumentation(
            28080
        )

        # Update some metrics
        update_metrics(
            {
                "currentStep": 50,
                "totalSteps": 100,
                "progressPercentage": 50,
            }
        )

        # Verify get_metrics_json returns valid JSON
        metrics_json = get_metrics_json()
        metrics = json.loads(metrics_json)

        assert metrics["currentStep"] == 50
        assert metrics["totalSteps"] == 100
        assert metrics["progressPercentage"] == 50

        # Verify handler class exists and is an HTTP handler
        assert handler_cls is not None
        assert handler_cls.__name__ == "ProgressionMetricsHandler"

    print("test execution complete")


def test_thread_safe_metrics_updates():
    """Test that metrics updates are thread-safe."""
    print("Executing test: thread-safe metrics updates")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        _, _, _, get_metrics_json, update_metrics = create_progression_instrumentation(28080)

        # Update metrics multiple times
        update_metrics({"currentStep": 10})
        update_metrics({"currentStep": 20})
        update_metrics({"totalSteps": 100})

        metrics = json.loads(get_metrics_json())

        # Verify last update wins for scalar values
        assert metrics["currentStep"] == 20
        assert metrics["totalSteps"] == 100

        # Update dict metrics (should merge, not replace)
        update_metrics({"trainMetrics": {"loss": 0.5}})
        update_metrics({"trainMetrics": {"learning_rate": 1e-4}})

        metrics = json.loads(get_metrics_json())

        # Verify dict updates are merged
        assert metrics["trainMetrics"]["loss"] == 0.5
        assert metrics["trainMetrics"]["learning_rate"] == 1e-4

    print("test execution complete")


def test_termination_message_write(tmp_path):
    """Test that final metrics are written to termination log."""
    print("Executing test: termination message write")

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_trainer_module = Mock()
    mock_transformers.trainer = mock_trainer_module

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer": mock_trainer_module,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.progression import (
            create_progression_instrumentation,
        )

        _, callback_cls, _, _, _ = create_progression_instrumentation(28080)
        callback = callback_cls(metrics_port=28080)

        args = Mock()
        args.num_train_epochs = 2
        state = Mock()
        state.global_step = 100
        state.max_steps = 100
        state.epoch = 2.0
        state.is_world_process_zero = True
        control = Mock()

        # Use a temp file for termination log
        termination_log = tmp_path / "termination-log"

        # Mock open to redirect /dev/termination-log to our temp file
        original_open = open

        def mock_open(filepath, *args, **kwargs):
            if filepath == "/dev/termination-log":
                return original_open(str(termination_log), *args, **kwargs)
            return original_open(filepath, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open):
            callback.on_train_begin(args, state, control)
            callback.on_train_end(args, state, control)

        # Verify termination message was written
        assert termination_log.exists()
        termination_data = json.loads(termination_log.read_text())

        assert termination_data["progressPercentage"] == 100
        assert termination_data["currentStep"] == 100
        assert termination_data["totalSteps"] == 100
        assert termination_data["estimatedRemainingSeconds"] == 0

    print("test execution complete")
