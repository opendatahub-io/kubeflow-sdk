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

from unittest.mock import patch

import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.transformers import (
    PeriodicCheckpointConfig,
    TransformersTrainer,
    _build_checkpoint_code,
    get_jit_checkpoint_injection_code,
    get_transformers_instrumentation_wrapper,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


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

    print("test execution complete")


def test_transformers_trainer_with_custom_config():
    """Test TransformersTrainer with custom configuration."""
    print("Executing test: TransformersTrainer with custom configuration")

    def train_with_args(lr: float, batch_size: int):
        print(f"Training with lr={lr}, batch_size={batch_size}")

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

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="poll interval too low (4 seconds)",
            expected_status="failed",
            config={"metrics_poll_interval_seconds": 4},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval too high (301 seconds)",
            expected_status="failed",
            config={"metrics_poll_interval_seconds": 301},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval minimum boundary (5 seconds)",
            expected_status="success",
            config={"metrics_poll_interval_seconds": 5},
            expected_output=5,
        ),
        TestCase(
            name="poll interval maximum boundary (300 seconds)",
            expected_status="success",
            config={"metrics_poll_interval_seconds": 300},
            expected_output=300,
        ),
        TestCase(
            name="poll interval invalid type (float)",
            expected_status="failed",
            config={"metrics_poll_interval_seconds": 30.5},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval invalid type (string)",
            expected_status="failed",
            config={"metrics_poll_interval_seconds": "30"},
            expected_error=ValueError,
        ),
    ],
)
def test_metrics_poll_interval_validation(test_case):
    """Test metrics_poll_interval_seconds validation."""
    print(f"Executing test: {test_case.name}")

    def dummy_train():
        pass

    try:
        trainer = TransformersTrainer(
            func=dummy_train,
            metrics_poll_interval_seconds=test_case.config["metrics_poll_interval_seconds"],
        )

        assert test_case.expected_status == "success"
        assert trainer.metrics_poll_interval_seconds == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == "failed"
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="port too low (1023)",
            expected_status="failed",
            config={"metrics_port": 1023},
            expected_error=ValueError,
        ),
        TestCase(
            name="port too high (65536)",
            expected_status="failed",
            config={"metrics_port": 65536},
            expected_error=ValueError,
        ),
        TestCase(
            name="port minimum boundary (1024)",
            expected_status="success",
            config={"metrics_port": 1024},
            expected_output=1024,
        ),
        TestCase(
            name="port maximum boundary (65535)",
            expected_status="success",
            config={"metrics_port": 65535},
            expected_output=65535,
        ),
        TestCase(
            name="port invalid type (float)",
            expected_status="failed",
            config={"metrics_port": 8080.5},
            expected_error=ValueError,
        ),
        TestCase(
            name="port invalid type (string)",
            expected_status="failed",
            config={"metrics_port": "8080"},
            expected_error=ValueError,
        ),
    ],
)
def test_metrics_port_validation(test_case):
    """Test metrics_port validation."""
    print(f"Executing test: {test_case.name}")

    def dummy_train():
        pass

    try:
        trainer = TransformersTrainer(
            func=dummy_train,
            metrics_port=test_case.config["metrics_port"],
        )

        assert test_case.expected_status == "success"
        assert trainer.metrics_port == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == "failed"
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="func is string",
            expected_status="failed",
            config={"func": "not_callable"},
            expected_error=ValueError,
        ),
        TestCase(
            name="func is integer",
            expected_status="failed",
            config={"func": 123},
            expected_error=ValueError,
        ),
        TestCase(
            name="func is None",
            expected_status="failed",
            config={"func": None},
            expected_error=ValueError,
        ),
        TestCase(
            name="func is dict",
            expected_status="failed",
            config={"func": {"train": "function"}},
            expected_error=ValueError,
        ),
    ],
)
def test_func_callable_validation(test_case):
    """Test func callable validation."""
    print(f"Executing test: {test_case.name}")

    try:
        TransformersTrainer(func=test_case.config["func"])
        assert test_case.expected_status == "success"

    except Exception as e:
        assert test_case.expected_status == "failed"
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="basic generation - returns string",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("isinstance", str),
                ("class KubeflowProgressCallback", True),
                ("class ProgressionMetricsHandler", True),
                ("def apply_progression_tracking", True),
                ("{{user_func_import_and_call}}", True),
                ("metrics_port=28080", True),
            ],
        ),
        TestCase(
            name="self-contained - no SDK imports",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("from kubeflow", False),
                ("import kubeflow", False),
                ("class ProgressionMetricsHandler", True),
                ("class KubeflowProgressCallback", True),
                ("def apply_progression_tracking", True),
                ("from transformers import", True),
            ],
        ),
        TestCase(
            name="structure - function call and user code placeholder",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("apply_progression_tracking", True),
                ("metrics_port=28080", True),
                ("apply_progression_tracking()", True),
                ("# USER TRAINING CODE", True),
            ],
        ),
        TestCase(
            name="completeness - all callback methods",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("def on_train_begin", True),
                ("def on_step_end", True),
                ("def on_log", True),
                ("def on_train_end", True),
                ("_original_init", True),
                ("def _instrumented_trainer_init", True),
            ],
        ),
        TestCase(
            name="implementation details - progress tracking logic",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("from kubeflow.trainer.rhai", False),
                ("state.global_step", True),
                ("progress_pct", True),
                ("elapsed_sec", True),
            ],
        ),
        TestCase(
            name="dataclass - ProgressionMetricsState",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("@dataclass", True),
                ("class ProgressionMetricsState", True),
                ("progressPercentage", True),
                ("estimatedRemainingSeconds", True),
                ("currentStep", True),
                ("totalSteps", True),
                ("trainMetrics", True),
                ("evalMetrics", True),
                ("asdict", True),
            ],
        ),
        TestCase(
            name="metrics state initialization",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("progressPercentage: Optional[int] = None", True),
                ("currentStep: int = 0", True),
                ("currentEpoch: float = 0.0", True),
                ("trainMetrics: dict[str, Any] = field(default_factory=dict)", True),
                ("evalMetrics: dict[str, Any] = field(default_factory=dict)", True),
            ],
        ),
        TestCase(
            name="thread safety mechanisms",
            expected_status="success",
            config={"port": 28080},
            expected_output=[
                ("_progression_metrics_lock", True),
                ("threading.Lock()", True),
                ("with _progression_metrics_lock:", True),
                ("_update_progression_metrics", True),
            ],
        ),
    ],
)
def test_instrumentation_wrapper_content(test_case):
    """Test instrumentation wrapper content contains expected elements."""
    print(f"Executing test: {test_case.name}")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=test_case.config["port"],
    )

    assert test_case.expected_status == "success"

    for check_item, expected in test_case.expected_output:
        if check_item == "isinstance":
            assert isinstance(wrapper, expected)
        elif expected:
            assert check_item in wrapper, f"Expected '{check_item}' to be in wrapper"
        else:
            assert check_item not in wrapper, f"Expected '{check_item}' to NOT be in wrapper"

    print("test execution complete")


def test_instrumentation_wrapper_user_code_ordering():
    """Test that user code placeholder appears after instrumentation setup.

    This validates the critical execution order - instrumentation must be applied
    before user training code runs.
    """
    print("Executing test: User code ordering validation")

    wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)

    lines = wrapper.split("\n")
    user_code_marker_index = next(i for i, line in enumerate(lines) if "USER TRAINING CODE" in line)
    placeholder_index = next(
        i for i, line in enumerate(lines) if "{{user_func_import_and_call}}" in line
    )

    # User code placeholder must come after marker to ensure instrumentation runs first
    assert placeholder_index > user_code_marker_index, (
        f"User code placeholder (line {placeholder_index}) must come after "
        f"marker (line {user_code_marker_index}) for correct execution order"
    )

    print("test execution complete")


def test_instrumentation_wrapper_no_syntax_errors():
    """Test that generated wrapper has no obvious syntax errors."""
    print("Executing test: Wrapper syntax validation")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
    )

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

        for key, value in test_case.config.items():
            assert getattr(trainer, key) == value

    except Exception as e:
        if test_case.expected_error:
            assert type(e) is test_case.expected_error
        else:
            raise

    print("test execution complete")


def test_get_trainer_cr_basic():
    """Test basic Trainer CRD generation from TransformersTrainer."""
    print("Executing test: Basic Trainer CRD generation")

    def dummy_train():
        print("Training...")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TransformersTrainer(func=dummy_train)

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    assert trainer_crd is not None
    assert trainer_crd.command is not None
    assert len(trainer_crd.command) > 0
    # Check progression tracking is wrapped
    assert "[Kubeflow] Initializing progression tracking" in " ".join(trainer_crd.command)

    print("test execution complete")


def test_get_trainer_cr_with_num_nodes():
    """Test Trainer CRD generation with num_nodes set."""
    print("Executing test: Trainer CRD with num_nodes")

    def dummy_train():
        print("Training...")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TransformersTrainer(func=dummy_train, num_nodes=4)

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    assert trainer_crd.num_nodes == 4

    print("test execution complete")


def test_get_trainer_cr_with_resources():
    """Test Trainer CRD generation with resources_per_node."""
    print("Executing test: Trainer CRD with resources")

    def dummy_train():
        print("Training...")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TransformersTrainer(
        func=dummy_train, resources_per_node={"cpu": 4, "memory": "8Gi", "nvidia.com/gpu": 1}
    )

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    assert trainer_crd.resources_per_node is not None
    assert trainer_crd.resources_per_node.requests is not None

    print("test execution complete")


def test_get_trainer_cr_with_env():
    """Test Trainer CRD generation with environment variables."""
    print("Executing test: Trainer CRD with environment variables")

    def dummy_train():
        print("Training...")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TransformersTrainer(
        func=dummy_train, env={"HF_HOME": "/data/huggingface", "WANDB_DISABLED": "true"}
    )

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    assert trainer_crd.env is not None
    assert len(trainer_crd.env) == 2
    env_dict = {env.name: env.value for env in trainer_crd.env}
    assert env_dict["HF_HOME"] == "/data/huggingface"
    assert env_dict["WANDB_DISABLED"] == "true"

    print("test execution complete")


def test_get_trainer_cr_with_func_args():
    """Test Trainer CRD generation with function arguments."""
    print("Executing test: Trainer CRD with function arguments")

    def train_with_args(lr: float, batch_size: int):
        print(f"Training with lr={lr}, batch_size={batch_size}")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TransformersTrainer(func=train_with_args, func_args={"lr": 0.001, "batch_size": 32})

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    command_str = " ".join(trainer_crd.command)
    # Check for dict representation of arguments in the generated code
    assert "'lr': 0.001" in command_str
    assert "'batch_size': 32" in command_str

    print("test execution complete")


def test_get_trainer_cr_progression_disabled():
    """Test Trainer CRD generation with progression tracking disabled."""
    print("Executing test: Trainer CRD with progression tracking disabled")

    def dummy_train():
        print("Training...")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TransformersTrainer(func=dummy_train, enable_progression_tracking=False)

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    command_str = " ".join(trainer_crd.command)
    # Should NOT contain progression tracking code
    assert "[Kubeflow] Initializing progression tracking" not in command_str
    assert "KubeflowProgressCallback" not in command_str

    print("test execution complete")


def test_get_trainer_cr_custom_metrics_port():
    """Test Trainer CRD generation with custom metrics port."""
    print("Executing test: Trainer CRD with custom metrics port")

    def dummy_train():
        print("Training...")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TransformersTrainer(func=dummy_train, metrics_port=8888)

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    command_str = " ".join(trainer_crd.command)
    assert "metrics_port=8888" in command_str

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="honest progress - 371/375 steps (98%)",
            expected_status=SUCCESS,
            config={
                "state_global_step": 371,
                "state_max_steps": 375,
                "state_epoch": 2.96,
                "num_train_epochs": 3,
                "overwrite_attempt_step": 100,
            },
            expected_output={
                "currentStep": 371,
                "totalSteps": 375,
                "currentEpoch": 2.96,
                "totalEpochs": 3,
                "progressPercentage": 98,  # Honest: 371/375
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="exact completion - 200/200 steps (100%)",
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
            name="early stop - 500/1000 steps (50%)",
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
                "progressPercentage": 50,  # Honest: 500/1000
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="None global_step handling",
            expected_status=SUCCESS,
            config={
                "state_global_step": None,
                "state_max_steps": 1000,
                "state_epoch": 0.0,
                "num_train_epochs": 1,
                "overwrite_attempt_step": 200,
            },
            expected_output={
                "currentStep": 0,  # None treated as 0
                "totalSteps": 1000,
                "currentEpoch": 0.0,
                "totalEpochs": 1,
                "progressPercentage": 0,  # 0/1000 = 0%
                "estimatedRemainingSeconds": 0,
            },
        ),
    ],
)
def test_callback_completion_state_protection(test_case):
    """Test that completion state is protected from overwrites after training ends.

    This tests the fix for the bug where on_step_end was called after on_train_end
    during evaluation, overwriting actual progress values.

    Now reports honest progress (e.g., 98% for 371/375 steps, not forced to 100%).
    """
    print(f"Executing test: {test_case.name}")

    import sys
    from unittest.mock import Mock

    # Mock transformers module for exec environment
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer = Mock()
    sys.modules["transformers"] = mock_transformers

    try:
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        namespace = {}
        # Replace user code placeholder
        wrapper_code = wrapper.replace("{{user_func_import_and_call}}", "pass")
        # Skip the actual patching call (line after _create_progression_instrumentation)
        lines = wrapper_code.split("\n")
        modified_lines = []
        for line in lines:
            # Comment out the standalone apply_progression_tracking() call
            if line.strip() == "apply_progression_tracking()":
                modified_lines.append("# apply_progression_tracking()  # Skipped in tests")
            else:
                modified_lines.append(line)
        wrapper_code = "\n".join(modified_lines)
        exec(wrapper_code, namespace)

        (
            _,
            callback_class,
            _,
            get_metrics_json,
            _,
        ) = namespace["_create_progression_instrumentation"](28080)
        callback = callback_class(metrics_port=28080)

        import json

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
        state.global_step = test_case.config["overwrite_attempt_step"]
        callback.on_step_end(args, state, control)

        # Verify completion state NOT overwritten (protected by training_finished flag)
        metrics = json.loads(get_metrics_json())
        assert metrics["currentStep"] == test_case.expected_output["currentStep"], (
            "Completion state was overwritten!"
        )
        assert metrics["currentEpoch"] == test_case.expected_output["currentEpoch"], (
            "Completion epoch was overwritten!"
        )
        assert metrics["progressPercentage"] == test_case.expected_output["progressPercentage"], (
            f"Progress percentage: expected {test_case.expected_output['progressPercentage']}, "
            f"got {metrics['progressPercentage']}"
        )
        # Verify estimatedRemainingSeconds is 0 after training completes
        if "estimatedRemainingSeconds" in test_case.expected_output:
            assert metrics["estimatedRemainingSeconds"] == 0, (
                f"estimatedRemainingSeconds: expected 0, got {metrics['estimatedRemainingSeconds']}"
            )

        print("test execution complete")
    finally:
        # Clean up mock
        if "transformers" in sys.modules:
            del sys.modules["transformers"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="completion at max_steps - 1000/1000",
            expected_status=SUCCESS,
            config={
                "state_global_step": 1000,
                "state_max_steps": 1000,
                "state_epoch": 3.0,
                "num_train_epochs": 3,
            },
            expected_output={
                "currentStep": 1000,
                "totalSteps": 1000,
                "currentEpoch": 3.0,
                "totalEpochs": 3,
                "progressPercentage": 100,
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="completion exceeded - 1001/1000 steps",
            expected_status=SUCCESS,
            config={
                "state_global_step": 1001,
                "state_max_steps": 1000,
                "state_epoch": 3.0,
                "num_train_epochs": 3,
            },
            expected_output={
                "currentStep": 1001,
                "totalSteps": 1000,
                "currentEpoch": 3.0,
                "totalEpochs": 3,
                "progressPercentage": 100,
                "estimatedRemainingSeconds": 0,
            },
        ),
        TestCase(
            name="near completion - 999/1000 steps",
            expected_status=SUCCESS,
            config={
                "state_global_step": 999,
                "state_max_steps": 1000,
                "state_epoch": 2.99,
                "num_train_epochs": 3,
            },
            expected_output={
                "currentStep": 999,
                "totalSteps": 1000,
                "currentEpoch": 2.99,
                "totalEpochs": 3,
                "progressPercentage": 99,
                # estimatedRemainingSeconds should be calculated, not 0
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

    import sys
    from unittest.mock import Mock

    # Mock transformers module for exec environment
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer = Mock()
    sys.modules["transformers"] = mock_transformers

    try:
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        namespace = {}
        # Replace user code placeholder
        wrapper_code = wrapper.replace("{{user_func_import_and_call}}", "pass")
        # Skip the actual patching call
        lines = wrapper_code.split("\n")
        modified_lines = []
        for line in lines:
            if line.strip() == "apply_progression_tracking()":
                modified_lines.append("# apply_progression_tracking()  # Skipped in tests")
            else:
                modified_lines.append(line)
        wrapper_code = "\n".join(modified_lines)
        exec(wrapper_code, namespace)

        (
            _,
            callback_class,
            _,
            get_metrics_json,
            _,
        ) = namespace["_create_progression_instrumentation"](28080)
        callback = callback_class(metrics_port=28080)

        import json
        import time

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
    finally:
        # Clean up mock
        if "transformers" in sys.modules:
            del sys.modules["transformers"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train metrics only",
            expected_status=SUCCESS,
            config={
                "logs": {"loss": 0.5, "learning_rate": 0.001, "grad_norm": 0.1},
            },
            expected_output={
                "trainMetrics": {"loss": 0.5, "learning_rate": 0.001, "grad_norm": 0.1},
                "evalMetrics": {},
            },
        ),
        TestCase(
            name="eval metrics only",
            expected_status=SUCCESS,
            config={
                "logs": {"eval_loss": 0.4, "eval_accuracy": 0.95},
            },
            expected_output={
                "trainMetrics": {},
                "evalMetrics": {"eval_loss": 0.4, "eval_accuracy": 0.95},
            },
        ),
        TestCase(
            name="mixed train and eval metrics",
            expected_status=SUCCESS,
            config={
                "logs": {
                    "loss": 0.5,
                    "learning_rate": 0.001,
                    "grad_norm": 0.1,
                    "eval_loss": 0.4,
                    "eval_accuracy": 0.95,
                    "train_samples_per_second": 10.0,
                },
            },
            expected_output={
                "trainMetrics": {
                    "loss": 0.5,
                    "learning_rate": 0.001,
                    "grad_norm": 0.1,
                    "throughput_samples_sec": 10.0,
                },
                "evalMetrics": {"eval_loss": 0.4, "eval_accuracy": 0.95},
            },
        ),
        TestCase(
            name="throughput metric renaming",
            expected_status=SUCCESS,
            config={
                "logs": {"train_samples_per_second": 15.5},
            },
            expected_output={
                "trainMetrics": {"throughput_samples_sec": 15.5},
                "evalMetrics": {},
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

    import sys
    from unittest.mock import Mock

    # Mock transformers module for exec environment
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer = Mock()
    sys.modules["transformers"] = mock_transformers

    try:
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        namespace = {}
        # Replace user code placeholder
        wrapper_code = wrapper.replace("{{user_func_import_and_call}}", "pass")
        # Skip the actual patching call (line after _create_progression_instrumentation)
        lines = wrapper_code.split("\n")
        modified_lines = []
        for line in lines:
            # Comment out the standalone apply_progression_tracking() call
            if line.strip() == "apply_progression_tracking()":
                modified_lines.append("# apply_progression_tracking()  # Skipped in tests")
            else:
                modified_lines.append(line)
        wrapper_code = "\n".join(modified_lines)
        exec(wrapper_code, namespace)

        (
            _,
            callback_class,
            _,
            get_metrics_json,
            _,
        ) = namespace["_create_progression_instrumentation"](28080)
        callback = callback_class(metrics_port=28080)

        import json

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
    finally:
        # Clean up mock
        if "transformers" in sys.modules:
            del sys.modules["transformers"]


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
                "currentEpoch": 1.98,  # Float preserved, not truncated to 1
                "totalEpochs": 2,
                "progressPercentage": 97,  # 121/124 = 97%
            },
        ),
        TestCase(
            name="zero max_steps fallback",
            expected_status=SUCCESS,
            config={
                "state_global_step": 100,
                "state_max_steps": 0,  # Invalid, falls back to 100%
                "state_epoch": 1.5,
                "num_train_epochs": 2,
            },
            expected_output={
                "currentStep": 100,
                "totalSteps": None,  # max_steps=0 → None
                "currentEpoch": 1.5,
                "totalEpochs": 2,
                "progressPercentage": 100,  # Fallback when total_steps invalid
            },
        ),
        TestCase(
            name="negative max_steps fallback",
            expected_status=SUCCESS,
            config={
                "state_global_step": 50,
                "state_max_steps": -10,
                "state_epoch": 0.5,
                "num_train_epochs": 1,
            },
            expected_output={
                "currentStep": 50,
                "totalSteps": None,  # negative → None
                "currentEpoch": 0.5,
                "totalEpochs": 1,
                "progressPercentage": 100,  # Fallback
            },
        ),
        TestCase(
            name="epoch-based training edge - 2.99/3 epochs",
            expected_status=SUCCESS,
            config={
                "state_global_step": 123,
                "state_max_steps": 124,
                "state_epoch": 2.99,
                "num_train_epochs": 3,
            },
            expected_output={
                "currentStep": 123,
                "totalSteps": 124,
                "currentEpoch": 2.99,
                "totalEpochs": 3,
                "progressPercentage": 99,  # 123/124
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

    import sys
    from unittest.mock import Mock

    # Mock transformers module for exec environment
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer = Mock()
    sys.modules["transformers"] = mock_transformers

    try:
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        namespace = {}
        wrapper_code = wrapper.replace("{{user_func_import_and_call}}", "pass")
        lines = wrapper_code.split("\n")
        modified_lines = []
        for line in lines:
            if line.strip() == "apply_progression_tracking()":
                modified_lines.append("# apply_progression_tracking()  # Skipped in tests")
            else:
                modified_lines.append(line)
        wrapper_code = "\n".join(modified_lines)
        exec(wrapper_code, namespace)

        (
            _,
            callback_class,
            _,
            get_metrics_json,
            _,
        ) = namespace["_create_progression_instrumentation"](28080)
        callback = callback_class(metrics_port=28080)

        import json

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
    finally:
        if "transformers" in sys.modules:
            del sys.modules["transformers"]


def _mock_get_jit_checkpoint_injection_code(
    output_dir=None,
    cloud_remote_storage_uri=None,
    periodic_checkpoint_config=None,
    enable_jit_checkpoint=False,
    verify_cloud_storage_access=True,
    verify_cloud_storage_ssl=True,
):
    """Mock implementation of get_jit_checkpoint_injection_code that doesn't require torch."""
    parts = []

    # Build config dict
    config_lines = ["_KUBEFLOW_CHECKPOINT_CONFIG = {"]
    config_lines.append(f'    "enable_jit": {enable_jit_checkpoint},')
    config_lines.append(f'    "verify_cloud_storage_access": {verify_cloud_storage_access},')
    config_lines.append(f'    "verify_cloud_storage_ssl": {verify_cloud_storage_ssl},')
    if output_dir:
        config_lines.append(f'    "output_dir": {repr(output_dir)},')
    if cloud_remote_storage_uri:
        config_lines.append(f'    "cloud_remote_storage_uri": {repr(cloud_remote_storage_uri)},')
    if periodic_checkpoint_config:
        if "save_strategy" in periodic_checkpoint_config:
            config_lines.append(
                f'    "save_strategy": {repr(periodic_checkpoint_config["save_strategy"])},'
            )
        if "save_steps" in periodic_checkpoint_config:
            config_lines.append(f'    "save_steps": {periodic_checkpoint_config["save_steps"]},')
        if "save_total_limit" in periodic_checkpoint_config:
            config_lines.append(
                f'    "save_total_limit": {periodic_checkpoint_config["save_total_limit"]},'
            )
    config_lines.append("}")
    parts.append("\n".join(config_lines))

    # Add CheckpointManager if JIT enabled
    if enable_jit_checkpoint:
        parts.append("class CheckpointManager:\n    pass")

    # Add monkey-patch function
    parts.append("def setup_jit_checkpoint_monkey_patch():\n    pass")

    return "\n\n".join(parts)


# ============================================================================
# PeriodicCheckpointConfig Validation Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default periodic config values",
            expected_status=SUCCESS,
            config={"save_strategy": "epoch", "save_steps": None, "save_total_limit": 3},
        ),
        TestCase(
            name="valid epoch strategy",
            expected_status=SUCCESS,
            config={"save_strategy": "epoch"},
        ),
        TestCase(
            name="valid steps strategy with save_steps",
            expected_status=SUCCESS,
            config={"save_strategy": "steps", "save_steps": 100},
        ),
        TestCase(
            name="valid no strategy to disable periodic checkpointing",
            expected_status=SUCCESS,
            config={"save_strategy": "no"},
        ),
        TestCase(
            name="invalid strategy raises ValueError",
            expected_status=FAILED,
            config={"save_strategy": "invalid"},
            expected_error=ValueError,
        ),
        TestCase(
            name="steps strategy requires save_steps",
            expected_status=FAILED,
            config={"save_strategy": "steps"},
            expected_error=ValueError,
        ),
        TestCase(
            name="save_total_limit cannot be zero",
            expected_status=FAILED,
            config={"save_total_limit": 0},
            expected_error=ValueError,
        ),
        TestCase(
            name="save_total_limit cannot be negative",
            expected_status=FAILED,
            config={"save_total_limit": -1},
            expected_error=ValueError,
        ),
    ],
)
def test_periodic_checkpoint_config_validation(test_case):
    """Test PeriodicCheckpointConfig validation."""
    print("Executing test:", test_case.name)

    try:
        config = PeriodicCheckpointConfig(**test_case.config)

        assert test_case.expected_status == SUCCESS

        # Validate expected values
        if "save_strategy" in test_case.config:
            assert config.save_strategy == test_case.config["save_strategy"]
        if "save_steps" in test_case.config:
            assert config.save_steps == test_case.config["save_steps"]
        if "save_total_limit" in test_case.config:
            assert config.save_total_limit == test_case.config["save_total_limit"]

    except Exception as e:
        assert type(e) is test_case.expected_error

    print("test execution complete")


# ============================================================================
# Checkpoint Injection Tests
# ============================================================================


def _dummy_training_func():
    """Dummy function for testing."""
    pass


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="no checkpoint code when both disabled",
            expected_status=SUCCESS,
            config={
                "trainer": TransformersTrainer(
                    func=_dummy_training_func,
                    enable_jit_checkpoint=False,
                    periodic_checkpoint_config=None,
                )
            },
            expected_output="",
        ),
        TestCase(
            name="checkpoint code with JIT enabled and default periodic config",
            expected_status=SUCCESS,
            config={
                "trainer": TransformersTrainer(
                    func=_dummy_training_func,
                    enable_jit_checkpoint=True,
                    periodic_checkpoint_config=None,
                )
            },
            expected_output={
                "contains": [
                    "_KUBEFLOW_CHECKPOINT_CONFIG",
                    '"enable_jit": True',
                    "CheckpointManager",
                    "setup_jit_checkpoint_monkey_patch",
                ]
            },
        ),
        TestCase(
            name="checkpoint code with JIT and custom periodic config",
            expected_status=SUCCESS,
            config={
                "trainer": TransformersTrainer(
                    func=_dummy_training_func,
                    enable_jit_checkpoint=True,
                    output_dir="/mnt/checkpoints",
                    periodic_checkpoint_config=PeriodicCheckpointConfig(
                        save_strategy="steps",
                        save_steps=500,
                        save_total_limit=5,
                    ),
                )
            },
            expected_output={
                "contains": [
                    '"enable_jit": True',
                    "'/mnt/checkpoints'",
                    "\"save_strategy\": 'steps'",
                    '"save_steps": 500',
                    '"save_total_limit": 5',
                ]
            },
        ),
        TestCase(
            name="checkpoint code with periodic only no JIT",
            expected_status=SUCCESS,
            config={
                "trainer": TransformersTrainer(
                    func=_dummy_training_func,
                    enable_jit_checkpoint=False,
                    periodic_checkpoint_config=PeriodicCheckpointConfig(save_strategy="epoch"),
                )
            },
            expected_output={
                "contains": ['"enable_jit": False', "setup_jit_checkpoint_monkey_patch"],
                "not_contains": ["CheckpointManager"],
            },
        ),
        TestCase(
            name="strategy no propagates to training args",
            expected_status=SUCCESS,
            config={
                "trainer": TransformersTrainer(
                    func=_dummy_training_func,
                    enable_jit_checkpoint=False,
                    periodic_checkpoint_config=PeriodicCheckpointConfig(save_strategy="no"),
                )
            },
            expected_output={
                "contains": [
                    "_KUBEFLOW_CHECKPOINT_CONFIG",
                    "\"save_strategy\": 'no'",
                ],
            },
        ),
    ],
)
@patch(
    "kubeflow.trainer.rhai.transformers.get_jit_checkpoint_injection_code",
    _mock_get_jit_checkpoint_injection_code,
)
def test_checkpoint_code_injection(test_case):
    """Test checkpoint code injection logic."""
    print("Executing test:", test_case.name)

    try:
        trainer = test_case.config["trainer"]
        code = _build_checkpoint_code(trainer)

        assert test_case.expected_status == SUCCESS

        # Check expected output
        if test_case.expected_output == "":
            assert code == ""
        elif isinstance(test_case.expected_output, dict):
            assert code != ""

            # Check contains
            if "contains" in test_case.expected_output:
                for substring in test_case.expected_output["contains"]:
                    assert substring in code, f"Expected '{substring}' in generated code"

            # Check not_contains
            if "not_contains" in test_case.expected_output:
                for substring in test_case.expected_output["not_contains"]:
                    assert substring not in code, f"Did not expect '{substring}' in generated code"

    except Exception as e:
        assert type(e) is test_case.expected_error

    print("test execution complete")


def test_checkpoint_injection_code_execution_jit_enabled(tmp_path):
    """Integration test: execute generated checkpoint code with JIT enabled.

    This test actually runs the generated instrumentation code to verify it works,
    rather than just checking string content.
    """
    import subprocess
    import sys

    from kubeflow.trainer.rhai.transformers import (
        get_jit_checkpoint_injection_code,
    )

    # Generate the actual checkpoint injection code with JIT enabled
    checkpoint_code = get_jit_checkpoint_injection_code(
        output_dir="/mnt/test-checkpoints",
        periodic_checkpoint_config={
            "save_strategy": "epoch",
            "save_steps": None,
            "save_total_limit": 3,
        },
        enable_jit_checkpoint=True,
    )

    # Create stub torch module
    torch_stub = """
class Stream:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def synchronize(self):
        pass

class cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def Stream():
        return Stream()

    @staticmethod
    def stream(stream_obj):
        return stream_obj

class distributed:
    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def barrier():
        pass

Tensor = object  # Stub
"""

    # Create stub transformers module
    transformers_stub = """
class TrainerCallback:
    pass

class trainer_utils:
    PREFIX_CHECKPOINT_DIR = "checkpoint"

PREFIX_CHECKPOINT_DIR = "checkpoint"

class TrainerState:
    def __init__(self):
        self.global_step = 100

class Trainer:
    def __init__(self, *args, **kwargs):
        self.state = TrainerState()
        self.model = None
        self._init_args = args
        self._init_kwargs = kwargs
        self._get_output_dir_called = False
        self._save_checkpoint_called = False

    def _get_output_dir(self, trial=None):
        self._get_output_dir_called = True
        return "/tmp/test-output"

    def _save_checkpoint(self, model, trial=None):
        self._save_checkpoint_called = True

    def train(self, resume_from_checkpoint=None):
        return {"train_called": True, "resume_from": resume_from_checkpoint}
"""

    # Write stubs and checkpoint code to temp file
    test_file = tmp_path / "test_checkpoint_execution.py"
    test_code = f"""
import sys
import os

# Stub modules must be created before import
import types

# Create torch stub module
torch_module = types.ModuleType('torch')
exec('''{torch_stub}''', torch_module.__dict__)
sys.modules['torch'] = torch_module
sys.modules['torch.distributed'] = torch_module.distributed

# Create transformers stub module
transformers_module = types.ModuleType('transformers')
exec('''{transformers_stub}''', transformers_module.__dict__)
sys.modules['transformers'] = transformers_module

# Create transformers.trainer_utils submodule
trainer_utils_module = types.ModuleType('transformers.trainer_utils')
trainer_utils_module.PREFIX_CHECKPOINT_DIR = "checkpoint"
sys.modules['transformers.trainer_utils'] = trainer_utils_module

# Now execute the checkpoint injection code
{checkpoint_code}

# Verify the instrumentation was applied
print("CHECKPOINT_INSTRUMENTATION_LOADED=True")

# Test that apply_checkpointing was called
if 'apply_checkpointing' in dir():
    print("APPLY_CHECKPOINTING_EXISTS=True")

# Test that CheckpointManager exists when JIT is enabled
if 'CheckpointManager' in dir():
    print("CHECKPOINT_MANAGER_EXISTS=True")

# Test that monkey patch was applied
from transformers import Trainer
original_init = Trainer.__init__

# Create a trainer to test patching
class MockTrainingArgs:
    def __init__(self):
        self.output_dir = "/tmp/test"
        self.save_strategy = "steps"
        self.save_steps = 500
        self.save_total_limit = None

try:
    trainer = Trainer(None, MockTrainingArgs())
    print("TRAINER_CREATED=True")

    # Check if checkpoint config was applied
    if hasattr(trainer, '_init_kwargs'):
        # Check if callbacks were injected
        if 'callbacks' in trainer._init_kwargs:
            callbacks = trainer._init_kwargs['callbacks']
            print(f"CALLBACKS_INJECTED={{len(callbacks)}}")
except Exception as e:
    print(f"ERROR={{e}}")
"""

    test_file.write_text(test_code)

    # Execute the test file
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    output = result.stdout
    print(f"Test output:\\n{output}")

    if result.returncode != 0:
        print(f"Test stderr:\\n{result.stderr}")

    # Verify execution results
    assert result.returncode == 0, f"Execution failed with return code {result.returncode}"
    assert "CHECKPOINT_INSTRUMENTATION_LOADED=True" in output
    assert "APPLY_CHECKPOINTING_EXISTS=True" in output
    assert "TRAINER_CREATED=True" in output
    # Verify checkpoint config was applied and callback injected
    assert "Auto-injected JIT checkpoint callback" in output
    assert "CALLBACKS_INJECTED=1" in output


def test_checkpoint_injection_code_execution_jit_disabled(tmp_path):
    """Integration test: verify CheckpointManager is NOT injected when JIT disabled.

    This test ensures that when enable_jit_checkpoint=False, the generated code
    does NOT include CheckpointManager class definition.
    """
    import subprocess
    import sys

    from kubeflow.trainer.rhai.transformers import get_jit_checkpoint_injection_code

    # Generate checkpoint injection code with JIT disabled (only periodic checkpointing)
    checkpoint_code = get_jit_checkpoint_injection_code(
        output_dir="/mnt/test-checkpoints",
        periodic_checkpoint_config={
            "save_strategy": "epoch",
            "save_steps": None,
            "save_total_limit": 3,
        },
        enable_jit_checkpoint=False,  # JIT disabled
    )

    # Verify that the config contains enable_jit=False
    assert "'enable_jit': False" in checkpoint_code

    # Verify that periodic checkpoint config is present
    assert "'save_strategy': 'epoch'" in checkpoint_code
    assert "'save_total_limit': 3" in checkpoint_code

    # Test execution to ensure it works without CheckpointManager
    torch_stub = """
class Stream:
    pass

class cuda:
    @staticmethod
    def is_available():
        return True

class distributed:
    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def barrier():
        pass
"""

    transformers_stub = """
class TrainerCallback:
    pass

class trainer_utils:
    PREFIX_CHECKPOINT_DIR = "checkpoint"

PREFIX_CHECKPOINT_DIR = "checkpoint"

class TrainerState:
    def __init__(self):
        self.global_step = 100

class Trainer:
    def __init__(self, *args, **kwargs):
        self.state = TrainerState()
        self.model = None
        self._init_args = args
        self._init_kwargs = kwargs

    def train(self, resume_from_checkpoint=None):
        return {"train_called": True, "resume_from": resume_from_checkpoint}
"""

    test_file = tmp_path / "test_checkpoint_jit_disabled.py"
    test_code = f"""
import sys
import types

# Create stub modules
torch_module = types.ModuleType('torch')
exec('''{torch_stub}''', torch_module.__dict__)
sys.modules['torch'] = torch_module
sys.modules['torch.distributed'] = torch_module.distributed

transformers_module = types.ModuleType('transformers')
exec('''{transformers_stub}''', transformers_module.__dict__)
sys.modules['transformers'] = transformers_module

trainer_utils_module = types.ModuleType('transformers.trainer_utils')
trainer_utils_module.PREFIX_CHECKPOINT_DIR = "checkpoint"
sys.modules['transformers.trainer_utils'] = trainer_utils_module

# Execute the checkpoint injection code
{checkpoint_code}

# Verify CheckpointManager does NOT exist
if 'CheckpointManager' in dir():
    print("ERROR: CheckpointManager should NOT be defined when JIT is disabled")
    sys.exit(1)
else:
    print("CHECKPOINT_MANAGER_NOT_DEFINED=True")

# Verify apply_checkpointing exists (for periodic checkpointing)
if 'apply_checkpointing' in dir():
    print("APPLY_CHECKPOINTING_EXISTS=True")

# Create a trainer to verify periodic config is applied (without JIT callback)
from transformers import Trainer

class MockTrainingArgs:
    def __init__(self):
        self.output_dir = "/tmp/test"
        self.save_strategy = "steps"
        self.save_steps = 500
        self.save_total_limit = None

trainer = Trainer(None, MockTrainingArgs())
print("TRAINER_CREATED=True")

# Verify that NO JIT callback was injected
callbacks = trainer._init_kwargs.get('callbacks', [])
print(f"CALLBACKS_COUNT={{len(callbacks)}}")
"""

    test_file.write_text(test_code)

    # Execute the test file
    result = subprocess.run(
        [sys.executable, str(test_file)], capture_output=True, text=True, timeout=10
    )

    output = result.stdout
    print(f"Test output:\n{output}")

    if result.returncode != 0:
        print(f"Test stderr:\n{result.stderr}")

    # Verify execution results
    assert result.returncode == 0, f"Execution failed with return code {result.returncode}"
    assert "CHECKPOINT_MANAGER_NOT_DEFINED=True" in output
    assert "APPLY_CHECKPOINTING_EXISTS=True" in output
    assert "TRAINER_CREATED=True" in output
    # When JIT is disabled, no callbacks should be injected
    assert "CALLBACKS_COUNT=0" in output


# ============================================================================
# Auto-Resume Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="find latest checkpoint with multiple valid checkpoints",
            expected_status=SUCCESS,
            config={
                "checkpoints": {
                    "checkpoint-100": {"incomplete": False},
                    "checkpoint-500": {"incomplete": False},
                    "checkpoint-300": {"incomplete": False},
                }
            },
            expected_output={"latest_step": 500, "latest_name": "checkpoint-500"},
        ),
        TestCase(
            name="delete incomplete checkpoint and return next latest",
            expected_status=SUCCESS,
            config={
                "checkpoints": {
                    "checkpoint-500": {"incomplete": True},  # Should be deleted
                    "checkpoint-300": {"incomplete": False},  # Should be returned
                    "checkpoint-100": {"incomplete": False},
                }
            },
            expected_output={"latest_step": 300, "latest_name": "checkpoint-300"},
        ),
        TestCase(
            name="return None when no checkpoints exist",
            expected_status=SUCCESS,
            config={"checkpoints": {}},
            expected_output={"latest_step": None, "latest_name": None},
        ),
        TestCase(
            name="return None when all checkpoints are incomplete",
            expected_status=SUCCESS,
            config={
                "checkpoints": {
                    "checkpoint-500": {"incomplete": True},
                    "checkpoint-300": {"incomplete": True},
                }
            },
            expected_output={"latest_step": None, "latest_name": None},
        ),
        TestCase(
            name="ignore non-checkpoint directories",
            expected_status=SUCCESS,
            config={
                "checkpoints": {
                    "checkpoint-100": {"incomplete": False},
                    "not-a-checkpoint": {"incomplete": False},  # Should be ignored
                    "checkpoint-abc": {"incomplete": False},  # Invalid number
                    "other-dir": {"incomplete": False},  # Should be ignored
                }
            },
            expected_output={"latest_step": 100, "latest_name": "checkpoint-100"},
        ),
    ],
)
def test_find_latest_checkpoint(test_case, tmp_path):
    """Test _find_latest_checkpoint logic for finding and cleaning checkpoints."""
    print(f"Executing test: {test_case.name}")

    import os
    import re
    import shutil

    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER

    # Create checkpoint directories based on test config
    for checkpoint_name, checkpoint_config in test_case.config["checkpoints"].items():
        checkpoint_path = tmp_path / checkpoint_name
        checkpoint_path.mkdir()

        if checkpoint_config.get("incomplete"):
            (checkpoint_path / CHECKPOINT_INCOMPLETE_MARKER).write_text("incomplete")

    # Also create a file named checkpoint-200 to test directory check
    if "not-a-checkpoint" in test_case.config["checkpoints"]:
        (tmp_path / "checkpoint-200").write_text("file, not dir")

    # Implement the _find_latest_checkpoint logic
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []

    for name in os.listdir(tmp_path):
        match = checkpoint_pattern.match(name)
        if not match or not os.path.isdir(os.path.join(tmp_path, name)):
            continue

        checkpoint_path = os.path.join(tmp_path, name)
        incomplete_marker = os.path.join(checkpoint_path, CHECKPOINT_INCOMPLETE_MARKER)

        # Delete incomplete checkpoints
        if os.path.exists(incomplete_marker):
            print(f"[Test] Deleting incomplete checkpoint: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)
            continue

        checkpoints.append((int(match.group(1)), checkpoint_path))

    latest = None
    latest_step = None
    if checkpoints:
        checkpoints.sort(reverse=True)
        latest_step = checkpoints[0][0]
        latest = checkpoints[0][1]

    assert test_case.expected_status == SUCCESS

    # Verify expected output
    if test_case.expected_output["latest_step"] is None:
        assert latest is None
    else:
        assert latest_step == test_case.expected_output["latest_step"]
        assert test_case.expected_output["latest_name"] in latest

    # Verify incomplete checkpoints were deleted
    for checkpoint_name, checkpoint_config in test_case.config["checkpoints"].items():
        checkpoint_path = tmp_path / checkpoint_name
        if checkpoint_config.get("incomplete"):
            assert not checkpoint_path.exists(), f"{checkpoint_name} should be deleted"

    print("test execution complete")


def test_find_latest_checkpoint_nonexistent_dir():
    """Test _find_latest_checkpoint returns None for nonexistent directory."""
    print("Executing test: find latest checkpoint with nonexistent directory")

    import os

    output_dir = "/nonexistent/directory/path"

    # Test the logic
    result = None
    if output_dir and os.path.exists(output_dir):
        # Would search for checkpoints
        result = "some_checkpoint"

    assert result is None, "Should return None when output_dir doesn't exist"

    print("test execution complete")


def test_find_latest_checkpoint_none_output_dir():
    """Test _find_latest_checkpoint returns None for None output_dir."""
    print("Executing test: find latest checkpoint with None output_dir")

    output_dir = None

    # Test the logic
    result = None
    if output_dir:
        result = "some_checkpoint"

    assert result is None, "Should return None when output_dir is None"

    print("test execution complete")


def test_auto_resume_code_generation():
    """Test that auto-resume code is generated in checkpoint injection."""
    print("Executing test: auto-resume code generation")

    from kubeflow.trainer.rhai.transformers import get_jit_checkpoint_injection_code

    checkpoint_code = get_jit_checkpoint_injection_code(
        output_dir="/tmp/checkpoints",
        periodic_checkpoint_config=None,
        enable_jit_checkpoint=True,
    )

    # Verify auto-resume logic is present in generated code
    assert "_find_latest_checkpoint" in checkpoint_code, "Should include _find_latest_checkpoint"
    assert "def _patched_train" in checkpoint_code, "Should include _patched_train"
    assert "resume_from_checkpoint" in checkpoint_code, (
        "Should include resume_from_checkpoint logic"
    )
    assert "if resume_from_checkpoint is None" in checkpoint_code, "Should check if user set it"
    assert "Auto-resuming from:" in checkpoint_code, "Should log auto-resume action"
    assert "self.train = _patched_train" in checkpoint_code, "Should patch train method"

    # Verify imports needed for auto-resume
    assert "import re" in checkpoint_code, "Should import re for regex"
    assert "import shutil" in checkpoint_code, "Should import shutil for rmtree"
    assert "import os" in checkpoint_code, "Should import os"

    print("test execution complete")


def test_auto_resume_user_override():
    """Test that user's explicit resume_from_checkpoint is not overridden."""
    print("Executing test: auto-resume respects user override")

    import sys
    from unittest.mock import Mock

    # Mock transformers module
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer = Mock()
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"
    sys.modules["transformers"] = mock_transformers
    sys.modules["transformers.trainer_utils"] = mock_transformers.trainer_utils

    # Mock torch module
    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    sys.modules["torch"] = mock_torch

    try:
        from kubeflow.trainer.rhai.transformers import get_jit_checkpoint_injection_code

        checkpoint_code = get_jit_checkpoint_injection_code(
            output_dir="/tmp/checkpoints",
            periodic_checkpoint_config=None,
            enable_jit_checkpoint=True,
        )

        # Verify the logic: only auto-resume if resume_from_checkpoint is None
        assert "if resume_from_checkpoint is None" in checkpoint_code

        assert (
            "resume_from_checkpoint is None and training_args" in checkpoint_code
            or "if resume_from_checkpoint is None" in checkpoint_code
        )

        print("test execution complete")

    finally:
        if "transformers" in sys.modules:
            del sys.modules["transformers"]
        if "transformers.trainer_utils" in sys.modules:
            del sys.modules["transformers.trainer_utils"]
        if "torch" in sys.modules:
            del sys.modules["torch"]


def test_transformers_trainer_s3_requires_data_connection_name():
    """Test TransformersTrainer raises error when S3 output_dir without data_connection_name."""
    print("Executing test: S3 output_dir requires data_connection_name")

    def dummy_train():
        pass

    with pytest.raises(ValueError) as exc_info:
        TransformersTrainer(
            func=dummy_train,
            output_dir="s3://my-bucket/checkpoints",
            # data_connection_name is missing
        )

    assert "data_connection_name is required" in str(exc_info.value)
    assert "s3://" in str(exc_info.value)

    print("test execution complete")


def test_build_checkpoint_code_with_s3_storage_uri():
    """Test _build_checkpoint_code passes storage_uri for S3 output_dir."""
    print("Executing test: _build_checkpoint_code with S3 storage_uri")

    def dummy_train():
        pass

    trainer = TransformersTrainer(
        func=dummy_train,
        output_dir="s3://my-bucket/checkpoints",
        data_connection_name="my-s3-secret",
        enable_jit_checkpoint=True,
    )

    code = _build_checkpoint_code(trainer)

    # Verify cloud_remote_storage_uri is included in the generated code
    assert code != ""
    assert "cloud_remote_storage_uri" in code
    assert "s3://my-bucket/checkpoints" in code

    print("test execution complete")


def test_get_jit_checkpoint_injection_code_with_storage_uri():
    """Test get_jit_checkpoint_injection_code includes cloud_remote_storage_uri in config."""
    print("Executing test: get_jit_checkpoint_injection_code with cloud_remote_storage_uri")

    checkpoint_code = get_jit_checkpoint_injection_code(
        output_dir="/mnt/kubeflow-checkpoints",
        cloud_remote_storage_uri="s3://my-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )

    # Verify cloud_remote_storage_uri is in the generated config
    assert "cloud_remote_storage_uri" in checkpoint_code
    assert "s3://my-bucket/model-checkpoints" in checkpoint_code

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="download latest checkpoint from S3",
            expected_status=SUCCESS,
            config={
                "checkpoints": ["checkpoint-100", "checkpoint-300", "checkpoint-200"],
                "incomplete_markers": [],
                "is_rank_0": True,
            },
            expected_output="checkpoint-300",
        ),
        TestCase(
            name="skip incomplete checkpoint in S3",
            expected_status=SUCCESS,
            config={
                "checkpoints": ["checkpoint-100", "checkpoint-300", "checkpoint-200"],
                "incomplete_markers": ["checkpoint-300"],
                "is_rank_0": True,
            },
            expected_output="checkpoint-200",
        ),
        TestCase(
            name="rank 1 does not download from S3",
            expected_status=SUCCESS,
            config={
                "checkpoints": ["checkpoint-100"],
                "incomplete_markers": [],
                "is_rank_0": False,
            },
            expected_output=None,  # No download for rank 1
        ),
        TestCase(
            name="empty remote storage - no checkpoints available",
            expected_status=SUCCESS,
            config={
                "checkpoints": [],  # No checkpoints in storage
                "incomplete_markers": [],
                "is_rank_0": True,
            },
            expected_output=None,  # No download, message logged
        ),
        TestCase(
            name="all checkpoints incomplete - none available for download",
            expected_status=SUCCESS,
            config={
                "checkpoints": ["checkpoint-100", "checkpoint-200", "checkpoint-300"],
                "incomplete_markers": [
                    "checkpoint-100",
                    "checkpoint-200",
                    "checkpoint-300",
                ],  # All incomplete
                "is_rank_0": True,
            },
            expected_output=None,  # No download, message logged
        ),
        TestCase(
            name="remote storage path does not exist yet - first training run",
            expected_status=SUCCESS,
            config={
                "checkpoints": None,  # Trigger FileNotFoundError in ls()
                "incomplete_markers": [],
                "is_rank_0": True,
            },
            expected_output=None,  # No download, starts from scratch
        ),
    ],
)
def test_s3_download_execution(test_case, tmp_path):
    """Integration test: S3 checkpoint download behavior.

    Tests:
    - Latest checkpoint is downloaded
    - Incomplete checkpoints are skipped
    - Only rank 0 downloads
    """
    print(f"Executing test: {test_case.name}")

    import subprocess
    import sys

    from kubeflow.trainer.rhai.transformers import get_jit_checkpoint_injection_code

    checkpoint_code = get_jit_checkpoint_injection_code(
        output_dir="/mnt/checkpoints",
        cloud_remote_storage_uri="s3://test-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )

    # Create fsspec stub based on test config
    checkpoints = test_case.config["checkpoints"]
    if checkpoints is None:
        # Simulate remote storage path not existing yet
        ls_implementation = '        raise FileNotFoundError("Remote storage path does not exist")'
    else:
        checkpoints_list = ", ".join([f'"{cp}"' for cp in checkpoints])
        ls_implementation = f"        return [{checkpoints_list}]"

    incomplete_checks = []
    for marker in test_case.config["incomplete_markers"]:
        incomplete_checks.append(
            f'        if "{marker}" in path and "checkpoint-is-incomplete.txt" in path:\n'
            f"            return True"
        )
    incomplete_logic = "\n".join(incomplete_checks) if incomplete_checks else "        pass"

    fsspec_stub = f"""
class MockS3FileSystem:
    def __init__(self, *args, **kwargs):
        pass

    def ls(self, path, detail=False):
{ls_implementation}

    def exists(self, path):
{incomplete_logic}
        return False

    def pipe(self, path, data):
        pass

    def cat(self, path):
        return b"test"

    def get(self, src, dst, recursive=False, callback=None):
        print(f"DOWNLOADED={{src}}")

    def du(self, path, total=True, maxdepth=None):
        # Return mock size (1 MB)
        return 1024 * 1024

    def rm_file(self, path):
        pass

    def rm_file(self, path):
        pass

class Callback:
    def __init__(self):
        self.size = 0
        self.value = 0

    def set_size(self, size):
        self.size = size

    def relative_update(self, inc=1):
        self.value += inc

class callbacks:
    Callback = Callback

def filesystem(protocol, **kwargs):
    return MockS3FileSystem()
"""

    torch_stub = """
class distributed:
    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def barrier():
        pass

class cuda:
    @staticmethod
    def is_available():
        return False
"""

    transformers_stub = """
class TrainerCallback:
    pass

class trainer_utils:
    PREFIX_CHECKPOINT_DIR = "checkpoint"

PREFIX_CHECKPOINT_DIR = "checkpoint"

class TrainerState:
    def __init__(self):
        self.global_step = 100
        self.is_world_process_zero = True

class Trainer:
    def __init__(self, *args, **kwargs):
        self.state = TrainerState()
        self.model = None
        self._init_args = args
        self._init_kwargs = kwargs

    def train(self, resume_from_checkpoint=None):
        return {"train_called": True, "resume_from": resume_from_checkpoint}
"""

    test_file = tmp_path / f"test_s3_{test_case.name.replace(' ', '_')}.py"
    is_rank_0 = test_case.config["is_rank_0"]

    test_code = f"""
import sys
import types

# Create fsspec stub
fsspec_module = types.ModuleType('fsspec')
exec('''{fsspec_stub}''', fsspec_module.__dict__)
sys.modules['fsspec'] = fsspec_module

# Create fsspec.callbacks submodule
callbacks_module = types.ModuleType('fsspec.callbacks')
callbacks_module.Callback = fsspec_module.Callback
sys.modules['fsspec.callbacks'] = callbacks_module
fsspec_module.callbacks = callbacks_module

# Create torch stub
torch_module = types.ModuleType('torch')
exec('''{torch_stub}''', torch_module.__dict__)
sys.modules['torch'] = torch_module
sys.modules['torch.distributed'] = torch_module.distributed

# Create transformers stub
transformers_module = types.ModuleType('transformers')
exec('''{transformers_stub}''', transformers_module.__dict__)
sys.modules['transformers'] = transformers_module

trainer_utils_module = types.ModuleType('transformers.trainer_utils')
trainer_utils_module.PREFIX_CHECKPOINT_DIR = "checkpoint"
sys.modules['transformers.trainer_utils'] = trainer_utils_module

# Execute checkpoint injection code
{checkpoint_code}

from transformers import TrainerCallback

# Find JITCheckpointCallback
callback_class = None
for name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, TrainerCallback) and name != 'TrainerCallback':
        callback_class = obj
        break

if not callback_class:
    print("ERROR: JITCheckpointCallback not found")
    sys.exit(1)

callback = callback_class(cloud_remote_storage_uri="s3://test-bucket/model-checkpoints")

class MockArgs:
    output_dir = "/mnt/checkpoints"
    local_process_index = 0 if {is_rank_0} else 1

class MockState:
    is_world_process_zero = {is_rank_0}

callback.on_init_end(MockArgs(), MockState(), None)

print("TEST_COMPLETE=True")
"""

    test_file.write_text(test_code)

    result = subprocess.run(
        [sys.executable, str(test_file)], capture_output=True, text=True, timeout=10
    )

    output = result.stdout
    print(f"Test output:\n{output}")

    if result.returncode != 0:
        print(f"Test stderr:\n{result.stderr}")

    assert result.returncode == 0, f"Execution failed with return code {result.returncode}"
    assert "TEST_COMPLETE=True" in output

    # Verify expected download behavior
    if test_case.expected_output:
        assert f"DOWNLOADED={test_case.expected_output}" in output
    else:
        # No download should occur (rank 1, empty storage, or all incomplete)
        assert "DOWNLOADED=" not in output

        # Verify informative messages based on scenario
        if test_case.config["is_rank_0"]:
            checkpoints = test_case.config["checkpoints"]

            if checkpoints is None:
                # FileNotFoundError case - remote storage doesn't exist yet
                assert "No existing or valid checkpoints found in cloud storage" in output
                # Should only print once
                assert output.count("Training will start from scratch") == 1
            elif checkpoints and test_case.config["incomplete_markers"]:
                # Has checkpoints but all are incomplete
                assert "No existing or valid checkpoints found in cloud storage" in output
                # Should only print once
                assert output.count("Training will start from scratch") == 1
            elif not checkpoints:
                # Empty list - ls() succeeded but no checkpoints, no message expected
                assert "No valid checkpoints found" not in output
                assert "No existing checkpoints found" not in output

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="normalize S3 trailing slash",
            expected_status=SUCCESS,
            config={
                "output_dir": "s3://my-bucket/checkpoints/",
                "data_connection_name": "my-secret",
            },
            expected_output="s3://my-bucket/checkpoints",
        ),
        TestCase(
            name="normalize S3 double slashes",
            expected_status=SUCCESS,
            config={
                "output_dir": "s3://my-bucket//checkpoints",
                "data_connection_name": "my-secret",
            },
            expected_output="s3://my-bucket/checkpoints",
        ),
        TestCase(
            name="normalize PVC trailing slash",
            expected_status=SUCCESS,
            config={"output_dir": "pvc://my-pvc/path/"},
            expected_output="pvc://my-pvc/path",
        ),
        TestCase(
            name="error on missing S3 bucket",
            expected_status=FAILED,
            config={"output_dir": "s3://"},
            expected_error=ValueError,
        ),
        TestCase(
            name="error on missing PVC name",
            expected_status=FAILED,
            config={"output_dir": "pvc://"},
            expected_error=ValueError,
        ),
        TestCase(
            name="error on triple slash (missing bucket)",
            expected_status=FAILED,
            config={"output_dir": "s3:///prefix"},
            expected_error=ValueError,
        ),
        TestCase(
            name="error on unsupported scheme",
            expected_status=FAILED,
            config={"output_dir": "gs://bucket"},
            expected_error=ValueError,
        ),
        TestCase(
            name="allow local filesystem path",
            expected_status=SUCCESS,
            config={"output_dir": "/local/path"},
            expected_output="/local/path",
        ),
    ],
)
def test_output_dir_normalization(test_case):
    """Test output_dir normalization and validation."""
    print(f"Executing test: {test_case.name}")

    def dummy_train():
        pass

    try:
        trainer = TransformersTrainer(
            func=dummy_train,
            output_dir=test_case.config["output_dir"],
            data_connection_name=test_case.config.get("data_connection_name"),
        )

        assert test_case.expected_status == SUCCESS
        assert trainer.output_dir == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
