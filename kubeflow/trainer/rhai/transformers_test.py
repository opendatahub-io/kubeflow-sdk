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

    # Verify it contains all required components
    assert "# METRICS SERVER" in wrapper
    assert "class MetricsServer" in wrapper
    assert "def start_server" in wrapper
    assert "# PROGRESS CALLBACK" in wrapper
    assert "class ProgressCallback" in wrapper
    assert "# TRAINER MONKEY-PATCH" in wrapper
    assert "def enable_tracking" in wrapper
    assert "{{user_func_import_and_call}}" in wrapper

    # Verify port is injected correctly
    assert f"port={28080}" in wrapper or f"port = {28080}" in wrapper

    # Verify no SDK imports at runtime
    assert "from kubeflow.trainer" not in wrapper
    assert "from kubeflow.common" not in wrapper

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


def test_instrumentation_wrapper_self_contained():
    """Test that wrapper is self-contained with no external dependencies."""
    print("Executing test: Wrapper is self-contained")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify no kubeflow imports (SDK shouldn't be needed at runtime)
    assert "from kubeflow" not in wrapper
    assert "import kubeflow" not in wrapper

    # Verify all stdlib imports are present
    assert "import http.server" in wrapper
    assert "import json" in wrapper
    assert "import threading" in wrapper
    assert "import time" in wrapper

    # Verify transformers import (expected)
    assert "from transformers import" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_http_server_structure():
    """Test HTTP server implementation in wrapper."""
    print("Executing test: HTTP server structure")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify HTTP server has required methods
    assert "def do_GET(self):" in wrapper
    assert 'self.path == "/health"' in wrapper
    assert 'self.path in ("/", "/metrics")' in wrapper

    # Verify metrics handler functions (module-level, not class methods)
    assert "def _update_metrics(updates: dict)" in wrapper
    assert "def _get_metrics_json() -> str:" in wrapper

    # Verify metrics structure (KEP-2779 format)
    assert '"progressPercentage":' in wrapper
    assert '"estimatedRemainingSeconds":' in wrapper
    assert '"currentStep":' in wrapper
    assert '"totalSteps":' in wrapper
    assert '"trainMetrics":' in wrapper
    assert '"evalMetrics":' in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_callback_structure():
    """Test progress callback implementation in wrapper."""
    print("Executing test: Progress callback structure")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify callback methods
    assert "def on_train_begin(self, args, state, control, **kwargs)" in wrapper
    assert "def on_step_end(self, args, state, control, **kwargs)" in wrapper
    assert "def on_log(self, args, state, control, logs=None, **kwargs)" in wrapper
    assert "def on_train_end(self, args, state, control, **kwargs)" in wrapper

    # Verify callback initializes with custom metrics map
    assert "self.custom_metrics_map = " in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_trainer_patching():
    """Test Trainer patching logic in wrapper."""

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify trainer patching implementation
    assert "from transformers import trainer as trainer_module" in wrapper
    assert "_original_init = trainer_module.Trainer.__init__" in wrapper
    assert "def _instrumented_trainer_init(self, *args, **kwargs):" in wrapper
    assert "ProgressCallback(custom_metrics, metrics_port)" in wrapper
    assert "trainer_module.Trainer.__init__ = _instrumented_trainer_init" in wrapper

    # Verify enable_tracking function is called
    assert "enable_tracking" in wrapper

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

    # Should have empty dict and metrics_port passed to enable_tracking
    assert "enable_tracking({}, metrics_port=28080)" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_metrics_tracking():
    """Test that wrapper tracks standard metrics."""
    print("Executing test: Standard metrics tracking")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify standard metrics are tracked
    assert '"loss"' in wrapper
    assert '"learning_rate"' in wrapper
    assert '"train_samples_per_second"' in wrapper or '"throughput_samples_sec"' in wrapper

    # Verify metrics update logic (new format uses train_metrics/eval_metrics)
    assert "train_metrics" in wrapper
    assert "eval_metrics" in wrapper
    assert "_update_metrics" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_progress_calculation():
    """Test progress calculation logic in wrapper."""
    print("Executing test: Progress calculation")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
        custom_metrics={},
    )

    # Verify progress calculation
    assert "state.global_step" in wrapper
    assert "state.max_steps" in wrapper
    assert "progress_pct" in wrapper
    assert "/ total_steps * 100" in wrapper or "current_step / total_steps" in wrapper

    # Verify time estimation
    assert "elapsed_sec" in wrapper
    assert "remaining_sec" in wrapper
    assert "estimated_total_time" in wrapper

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
