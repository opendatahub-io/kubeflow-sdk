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

"""Tests for TransformersTrainer and instrumentation wrapper generation."""

import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.transformers import (
    TransformersTrainer,
    get_transformers_instrumentation_wrapper,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


def test_transformers_trainer_initialization():
    """Test TransformersTrainer initialization with default values."""
    print("Executing test: TransformersTrainer initialization with defaults")

    def dummy_train():
        print("Training...")

    trainer = TransformersTrainer(func=dummy_train)

    assert trainer.func == dummy_train
    assert trainer.func_args is None
    assert trainer.packages_to_install is None
    assert trainer.pip_index_urls == list(constants.DEFAULT_PIP_INDEX_URLS)
    assert trainer.num_nodes is None
    assert trainer.resources_per_node is None
    assert trainer.env is None
    assert trainer.enable_progression_tracking is True
    assert trainer.metrics_port == 28080
    assert trainer.metrics_poll_interval_seconds == 30
    assert trainer.custom_metrics == {}

    print("test execution complete")


def test_transformers_trainer_with_custom_config():
    """Test TransformersTrainer with custom configuration."""
    print("Executing test: TransformersTrainer with custom configuration")

    def train_with_args(lr: float, batch_size: int):
        print(f"Training with lr={lr}, batch_size={batch_size}")

    custom_metrics = {
        "eval_accuracy": "accuracy",
        "eval_f1": "f1_score",
    }

    trainer = TransformersTrainer(
        func=train_with_args,
        func_args={"lr": 0.001, "batch_size": 32},
        packages_to_install=["transformers", "datasets"],
        pip_index_urls=["https://custom.pypi.org/simple"],
        num_nodes=2,
        resources_per_node={"gpu": 1, "cpu": 4},
        env={"HF_HOME": "/data/huggingface"},
        enable_progression_tracking=True,
        metrics_port=28090,
        metrics_poll_interval_seconds=60,
        custom_metrics=custom_metrics,
    )

    assert trainer.func == train_with_args
    assert trainer.func_args == {"lr": 0.001, "batch_size": 32}
    assert trainer.packages_to_install == ["transformers", "datasets"]
    assert trainer.pip_index_urls == ["https://custom.pypi.org/simple"]
    assert trainer.num_nodes == 2
    assert trainer.resources_per_node == {"gpu": 1, "cpu": 4}
    assert trainer.env == {"HF_HOME": "/data/huggingface"}
    assert trainer.enable_progression_tracking is True
    assert trainer.metrics_port == 28090
    assert trainer.metrics_poll_interval_seconds == 60
    assert trainer.custom_metrics == custom_metrics

    print("test execution complete")


def test_transformers_trainer_with_progression_disabled():
    """Test TransformersTrainer with progression tracking disabled."""
    print("Executing test: TransformersTrainer with progression tracking disabled")

    def dummy_train():
        print("Training...")

    trainer = TransformersTrainer(
        func=dummy_train,
        enable_progression_tracking=False,
    )

    assert trainer.enable_progression_tracking is False

    print("test execution complete")


def test_instrumentation_wrapper_generation_basic():
    """Test basic instrumentation wrapper generation."""
    print("Executing test: Basic instrumentation wrapper generation")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify wrapper is a string
    assert isinstance(wrapper, str)

    # Verify it imports from SDK (new import-based approach)
    assert (
        "from kubeflow.trainer.rhai.progression_instrumentation import create_instrumentation"
    ) in wrapper

    # Verify it calls create_instrumentation
    assert "create_instrumentation" in wrapper
    assert "enable_tracking" in wrapper

    # Verify user code placeholder
    assert "{{user_func_import_and_call}}" in wrapper

    # Verify port is injected correctly
    assert f"metrics_port={28080}" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_with_custom_metrics():
    """Test instrumentation wrapper with custom metrics."""
    print("Executing test: Instrumentation wrapper with custom metrics")

    custom_metrics = {
        "eval_accuracy": "accuracy",
        "eval_f1": "f1_score",
        "rewards/chosen": "reward_chosen",
    }

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28090,
        custom_metrics=custom_metrics,
    )

    # Verify custom metrics are embedded in the wrapper
    assert "eval_accuracy" in wrapper
    assert "accuracy" in wrapper
    assert "eval_f1" in wrapper
    assert "f1_score" in wrapper
    assert "rewards/chosen" in wrapper
    assert "reward_chosen" in wrapper

    # Verify port is custom
    assert "28090" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_imports_sdk():
    """Test that wrapper imports from SDK (import-based approach)."""
    print("Executing test: Wrapper imports SDK")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify SDK import is present (new import-based approach)
    assert (
        "from kubeflow.trainer.rhai.progression_instrumentation import create_instrumentation"
    ) in wrapper

    # Verify wrapper is concise (no class/function definitions)
    assert "class MetricsServer" not in wrapper
    assert "class ProgressCallback" not in wrapper
    assert "def start_server" not in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_structure():
    """Test wrapper structure and function call."""
    print("Executing test: Wrapper structure")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify wrapper calls create_instrumentation with correct args
    assert "create_instrumentation(" in wrapper
    assert "custom_metrics={}" in wrapper
    assert "metrics_port=28080" in wrapper

    # Verify enable_tracking() is called
    assert "enable_tracking()" in wrapper

    # Verify user code section
    assert "# USER TRAINING CODE" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_conciseness():
    """Test wrapper is concise (no class/function definitions)."""
    print("Executing test: Wrapper conciseness")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify wrapper doesn't contain implementation details (those are in SDK)
    assert "def on_train_begin" not in wrapper
    assert "def on_step_end" not in wrapper
    assert "def on_log" not in wrapper
    assert "def on_train_end" not in wrapper
    assert "_original_init" not in wrapper
    assert "def _instrumented_trainer_init" not in wrapper

    # Wrapper should be very short (just imports and calls)
    assert len(wrapper.split("\n")) < 30, "Wrapper should be concise (< 30 lines)"

    print("test execution complete")


def test_instrumentation_wrapper_user_code_placeholder():
    """Test that wrapper has placeholder for user code injection."""
    print("Executing test: User code placeholder")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify placeholder exists
    assert "{{user_func_import_and_call}}" in wrapper

    # Verify it's in the right section
    assert "# USER TRAINING CODE" in wrapper
    lines = wrapper.split("\n")
    user_code_index = next(i for i, line in enumerate(lines) if "USER TRAINING CODE" in line)
    placeholder_index = next(
        i for i, line in enumerate(lines) if "{{user_func_import_and_call}}" in line
    )
    assert placeholder_index > user_code_index

    print("test execution complete")


def test_instrumentation_wrapper_multiple_ports():
    """Test wrapper generation with different ports."""
    print("Executing test: Multiple port configurations")

    ports = [28080, 28090, 8080, 9000]

    for port in ports:
        wrapper = get_transformers_instrumentation_wrapper(
            metrics_port=port,
            custom_metrics={},
        )
        assert str(port) in wrapper
        print(f"  ✓ Port {port} correctly embedded")

    print("test execution complete")


def test_instrumentation_wrapper_empty_custom_metrics():
    """Test wrapper with empty custom metrics dict."""
    print("Executing test: Empty custom metrics")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Should have empty dict and metrics_port passed to create_instrumentation
    assert "create_instrumentation(" in wrapper
    assert "custom_metrics={}" in wrapper
    assert "metrics_port=28080" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_sdk_delegation():
    """Test that wrapper delegates logic to SDK module."""
    print("Executing test: SDK delegation")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify wrapper delegates to SDK (doesn't contain implementation details)
    # All metrics tracking, progress calculation, etc. are in progression_instrumentation module
    assert "from kubeflow.trainer.rhai.progression_instrumentation" in wrapper

    # These should NOT be in the wrapper (they're in the SDK module)
    assert "state.global_step" not in wrapper
    assert "progress_pct" not in wrapper
    assert "elapsed_sec" not in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_no_syntax_errors():
    """Test that generated wrapper has no obvious syntax errors."""
    print("Executing test: Wrapper syntax validation")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={"eval_loss": "validation_loss"},
    )

    # Try to compile the wrapper (with placeholder removed)
    test_code = wrapper.replace("{{user_func_import_and_call}}", "pass")

    try:
        compile(test_code, "<string>", "exec")
        print("  ✓ Wrapper compiles successfully")
    except SyntaxError as e:
        pytest.fail(f"Generated wrapper has syntax error: {e}")

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="basic trainer without progression tracking",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": False,
            },
        ),
        TestCase(
            name="trainer with custom port",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
                "metrics_port": 8888,
            },
        ),
        TestCase(
            name="trainer with long poll interval",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
                "metrics_poll_interval_seconds": 120,
            },
        ),
        TestCase(
            name="trainer with multiple custom metrics",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
                "custom_metrics": {
                    "eval_accuracy": "accuracy",
                    "eval_precision": "precision",
                    "eval_recall": "recall",
                    "eval_f1": "f1_score",
                },
            },
        ),
    ],
)
def test_transformers_trainer_configurations(test_case):
    """Test various TransformersTrainer configurations."""
    print(f"Executing test: {test_case.name}")

    def dummy_train():
        print("Training...")

    try:
        trainer = TransformersTrainer(func=dummy_train, **test_case.config)

        assert test_case.expected_status == SUCCESS

        # Verify config was applied
        for key, value in test_case.config.items():
            assert getattr(trainer, key) == value

    except Exception as e:
        if test_case.expected_error:
            assert type(e) is test_case.expected_error
        else:
            raise

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
