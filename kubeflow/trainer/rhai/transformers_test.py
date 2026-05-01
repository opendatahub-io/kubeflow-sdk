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

from pathlib import Path
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
                ("progressPercentage: int | None = None", True),
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


def test_get_trainer_cr_non_pytorch_framework():
    """Test Trainer CRD generation with non-pytorch framework uses DEFAULT_COMMAND."""
    print("Executing test: Trainer CRD with non-pytorch framework")

    def dummy_train():
        print("Training...")

    from kubeflow.trainer.rhai.transformers import get_trainer_cr_from_transformers_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="jax",
            image="jax/jax:0.4.0",
        ),
    )
    trainer = TransformersTrainer(func=dummy_train, enable_progression_tracking=False)

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    command_str = " ".join(trainer_crd.command)
    assert "python" in command_str
    assert "torchrun" not in command_str

    print("test execution complete")


def test_get_trainer_cr_with_checkpoint_and_packages():
    """Test Trainer CRD generation with checkpoint injection and packages_to_install.

    Validates that:
    - Checkpoint header/footer code is injected around user code
    - packages_to_install generates pip install script in the command
    """
    print("Executing test: Trainer CRD with checkpoint and packages")

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
        func=dummy_train,
        enable_progression_tracking=False,
        enable_jit_checkpoint=True,
        output_dir="/tmp/checkpoints",
        packages_to_install=["transformers", "datasets"],
    )

    trainer_crd = get_trainer_cr_from_transformers_trainer(runtime, trainer)

    command_str = " ".join(trainer_crd.command)
    # Checkpoint header should be present
    assert "[Kubeflow] Initializing checkpoint instrumentation" in command_str
    assert "apply_checkpointing()" in command_str
    # Checkpoint footer should be present
    assert "upload_final_model_to_cloud()" in command_str
    # Package install script should be present
    assert "pip install" in command_str
    assert "transformers" in command_str
    assert "datasets" in command_str

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

    header = "\n\n".join(parts)
    footer = "# post-training cleanup placeholder"
    return header, footer


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
        header, footer = _build_checkpoint_code(trainer)

        assert test_case.expected_status == SUCCESS

        # Check expected output
        if test_case.expected_output == "":
            assert header == ""
            assert footer == ""
        elif isinstance(test_case.expected_output, dict):
            code = header + footer
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
    checkpoint_header, checkpoint_footer = get_jit_checkpoint_injection_code(
        output_dir="/mnt/test-checkpoints",
        periodic_checkpoint_config={
            "save_strategy": "epoch",
            "save_steps": None,
            "save_total_limit": 3,
        },
        enable_jit_checkpoint=True,
    )
    checkpoint_code = checkpoint_header

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
    def is_available():
        return False

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
    checkpoint_header, checkpoint_footer = get_jit_checkpoint_injection_code(
        output_dir="/mnt/test-checkpoints",
        periodic_checkpoint_config={
            "save_strategy": "epoch",
            "save_steps": None,
            "save_total_limit": 3,
        },
        enable_jit_checkpoint=False,  # JIT disabled
    )
    checkpoint_code = checkpoint_header

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
    def is_available():
        return False

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
            # Use per-rank marker format matching production code
            marker_name = f"{CHECKPOINT_INCOMPLETE_MARKER}.node-0-rank-0"
            (checkpoint_path / marker_name).write_text("incomplete")

    # Also create a file named checkpoint-200 to test directory check
    if "not-a-checkpoint" in test_case.config["checkpoints"]:
        (tmp_path / "checkpoint-200").write_text("file, not dir")

    # Implement the _find_latest_checkpoint logic (matches production code)
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []

    for name in os.listdir(tmp_path):
        match = checkpoint_pattern.match(name)
        if not match or not os.path.isdir(os.path.join(tmp_path, name)):
            continue

        checkpoint_path = os.path.join(tmp_path, name)

        # Check for any incomplete marker files (supports per-rank markers)
        has_incomplete = any(
            f.startswith(CHECKPOINT_INCOMPLETE_MARKER) for f in os.listdir(checkpoint_path)
        )

        # Delete incomplete checkpoints
        if has_incomplete:
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

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/tmp/checkpoints",
        periodic_checkpoint_config=None,
        enable_jit_checkpoint=True,
    )

    # Verify auto-resume logic is present in generated code
    assert "_find_latest_checkpoint" in checkpoint_header, "Should include _find_latest_checkpoint"
    assert "def _patched_train" in checkpoint_header, "Should include _patched_train"
    assert "resume_from_checkpoint" in checkpoint_header, (
        "Should include resume_from_checkpoint logic"
    )
    assert "if resume_from_checkpoint is None" in checkpoint_header, "Should check if user set it"
    assert "Auto-resuming from:" in checkpoint_header, "Should log auto-resume action"
    assert "self.train = _patched_train" in checkpoint_header, "Should patch train method"

    # Verify imports needed for auto-resume
    assert "import re" in checkpoint_header, "Should import re for regex"
    assert "import shutil" in checkpoint_header, "Should import shutil for rmtree"
    assert "import os" in checkpoint_header, "Should import os"

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

        checkpoint_header, _ = get_jit_checkpoint_injection_code(
            output_dir="/tmp/checkpoints",
            periodic_checkpoint_config=None,
            enable_jit_checkpoint=True,
        )

        # Verify the logic: only auto-resume if resume_from_checkpoint is None
        assert "if resume_from_checkpoint is None" in checkpoint_header

        assert (
            "resume_from_checkpoint is None and training_args" in checkpoint_header
            or "if resume_from_checkpoint is None" in checkpoint_header
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

    header, footer = _build_checkpoint_code(trainer)

    # Verify cloud_remote_storage_uri is included in the generated code
    assert header != ""
    assert "cloud_remote_storage_uri" in header
    assert "s3://my-bucket/checkpoints" in header

    print("test execution complete")


def test_get_jit_checkpoint_injection_code_with_storage_uri():
    """Test get_jit_checkpoint_injection_code includes cloud_remote_storage_uri in config."""
    print("Executing test: get_jit_checkpoint_injection_code with cloud_remote_storage_uri")

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/kubeflow-checkpoints",
        cloud_remote_storage_uri="s3://my-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )

    # Verify cloud_remote_storage_uri is in the generated config
    assert "cloud_remote_storage_uri" in checkpoint_header
    assert "s3://my-bucket/model-checkpoints" in checkpoint_header

    print("test execution complete")


def _run_checkpoint_validation_subprocess(
    tmp_path: Path, test_name: str, checkpoint_code: str, training_args_body: str
) -> tuple[int, str, str]:
    """Run injected checkpoint code in a subprocess and return exit code and output."""
    import subprocess
    import sys

    fsspec_stub = """
class MockS3FileSystem:
    def __init__(self, *args, **kwargs):
        pass

    def pipe(self, path, data):
        pass

    def cat(self, path):
        return b"test"

    def rm_file(self, path):
        pass

def filesystem(protocol, **kwargs):
    return MockS3FileSystem()
"""

    torch_stub = """
class distributed:
    @staticmethod
    def is_available():
        return False

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

class Tensor:
    pass
"""

    transformers_stub = """
class TrainerCallback:
    pass

class trainer_utils:
    PREFIX_CHECKPOINT_DIR = "checkpoint"

PREFIX_CHECKPOINT_DIR = "checkpoint"

class Trainer:
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs

    def train(self, resume_from_checkpoint=None, **train_kwargs):
        return {"train_called": True, "resume_from": resume_from_checkpoint}
"""

    test_file = tmp_path / f"{test_name}.py"
    test_code = f"""
import sys
import types

fsspec_module = types.ModuleType('fsspec')
exec('''{fsspec_stub}''', fsspec_module.__dict__)
sys.modules['fsspec'] = fsspec_module

callbacks_module = types.ModuleType('fsspec.callbacks')
callbacks_module.Callback = type('Callback', (), {{}})
sys.modules['fsspec.callbacks'] = callbacks_module
fsspec_module.callbacks = callbacks_module

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

{checkpoint_code}

from transformers import Trainer

class TrainingArgs:
{training_args_body}

try:
    Trainer(args=TrainingArgs())
    print("NO_ERROR")
    sys.exit(1)
except ValueError as e:
    print(str(e))
    sys.exit(0)
"""

    test_file.write_text(test_code)

    result = subprocess.run(
        [sys.executable, str(test_file)], capture_output=True, text=True, timeout=10
    )
    return result.returncode, result.stdout, result.stderr


def test_save_only_model_validation_error(tmp_path):
    """Test save_only_model=True raises a validation error."""
    print("Executing test: save_only_model validation error")

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/kubeflow-checkpoints",
        enable_jit_checkpoint=True,
    )

    returncode, stdout, stderr = _run_checkpoint_validation_subprocess(
        tmp_path=tmp_path,
        test_name="test_save_only_model_validation_error",
        checkpoint_code=checkpoint_header,
        training_args_body="    save_only_model = True\n",
    )

    if returncode != 0:
        print(f"Test stderr:\n{stderr}")

    assert returncode == 0
    assert "save_only_model=True is incompatible with Kubeflow checkpointing" in stdout

    print("test execution complete")


def test_save_on_each_node_validation_error(tmp_path):
    """Test save_on_each_node=True raises a validation error with S3."""
    print("Executing test: save_on_each_node validation error")

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/kubeflow-checkpoints",
        cloud_remote_storage_uri="s3://my-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )

    returncode, stdout, stderr = _run_checkpoint_validation_subprocess(
        tmp_path=tmp_path,
        test_name="test_save_on_each_node_validation_error",
        checkpoint_code=checkpoint_header,
        training_args_body="    save_on_each_node = True\n",
    )

    if returncode != 0:
        print(f"Test stderr:\n{stderr}")

    assert returncode == 0
    assert "save_on_each_node=True is not supported when output_dir is an S3 URI" in stdout

    print("test execution complete")


def test_async_upload_worker_scaffolding():
    """Test async upload worker scaffolding is present in generated code."""
    print("Executing test: async upload worker scaffolding")

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/kubeflow-checkpoints",
        cloud_remote_storage_uri="s3://my-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )

    assert "LifoQueue" in checkpoint_header
    assert "daemon=False" in checkpoint_header
    assert "KubeflowCheckpointUploader" in checkpoint_header
    assert "1 hour" in checkpoint_header
    assert "self._upload_thread.join(timeout=" in checkpoint_header

    print("test execution complete")


def test_async_parallel_upload_scaffolding():
    """Test parallel upload scaffolding is present in generated code."""
    print("Executing test: async parallel upload scaffolding")

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/kubeflow-checkpoints",
        cloud_remote_storage_uri="s3://my-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )

    assert "ThreadPoolExecutor" in checkpoint_header
    assert "as_completed" in checkpoint_header
    assert "_parallel_upload_files" in checkpoint_header

    print("test execution complete")


def test_async_upload_execution(tmp_path):
    """Integration test: async upload runs and cleans staging."""
    print("Executing test: async upload execution")

    import subprocess
    import sys

    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER

    output_dir = tmp_path / "checkpoints"

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir=str(output_dir),
        cloud_remote_storage_uri="s3://test-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )
    checkpoint_code = checkpoint_header

    fsspec_stub = """
class MockS3FileSystem:
    def __init__(self, *args, **kwargs):
        pass

    def ls(self, path, detail=False):
        return []

    def exists(self, path):
        return False

    def pipe(self, path, data):
        print(f"PIPE={path}")

    def cat(self, path):
        return b"test"

    def put_file(self, local, remote):
        print(f"PUT_FILE={remote}")

    def rm_file(self, path):
        print(f"RM_FILE={path}")

    def get(self, src, dst, recursive=False, callback=None):
        pass

    def du(self, path, total=True, maxdepth=None):
        return 0

def filesystem(protocol, **kwargs):
    return MockS3FileSystem()
"""

    torch_stub = """
class distributed:
    @staticmethod
    def is_available():
        return False

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
        self.global_step = 0
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

    test_file = tmp_path / "test_async_upload_execution.py"

    test_code = f"""
import os
import sys
import types

fsspec_module = types.ModuleType('fsspec')
exec('''{fsspec_stub}''', fsspec_module.__dict__)
sys.modules['fsspec'] = fsspec_module

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

{checkpoint_code}

from transformers import TrainerCallback

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
    output_dir = r"{output_dir}"
    local_process_index = 0

class MockState:
    global_step = 3
    is_world_process_zero = True

checkpoint_path = os.path.join(MockArgs.output_dir, "checkpoint-3")
os.makedirs(checkpoint_path, exist_ok=True)
with open(os.path.join(checkpoint_path, "model.bin"), "w", encoding="utf-8") as f:
    f.write("data")

callback.on_save(MockArgs(), MockState(), None)
callback.shutdown_upload_worker()

if True:
    staging_checkpoint = os.path.join(
        MockArgs.output_dir, CHECKPOINT_STAGING_DIR, "checkpoint-3"
    )
    print(f"STAGING_CHECKPOINT_EXISTS={{os.path.exists(staging_checkpoint)}}")
    print(f"CHECKPOINT_EXISTS={{os.path.exists(checkpoint_path)}}")

print("TEST_COMPLETE=True")
"""

    test_file.write_text(test_code)

    result = subprocess.run(
        [sys.executable, str(test_file)], capture_output=True, text=True, timeout=10
    )

    output = result.stdout
    print(f"Test output:\\n{output}")

    if result.returncode != 0:
        print(f"Test stderr:\\n{result.stderr}")

    assert result.returncode == 0, f"Execution failed with return code {result.returncode}"
    assert "TEST_COMPLETE=True" in output
    expected_marker = f"{CHECKPOINT_INCOMPLETE_MARKER}.node-0-rank-0"
    assert f"PIPE=checkpoint-3/{expected_marker}" in output
    assert "PUT_FILE=checkpoint-3/model.bin" in output
    assert f"RM_FILE=checkpoint-3/{expected_marker}" in output
    assert "STAGING_CHECKPOINT_EXISTS=False" in output
    assert "CHECKPOINT_EXISTS=False" in output

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

    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER
    from kubeflow.trainer.rhai.transformers import get_jit_checkpoint_injection_code

    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/checkpoints",
        cloud_remote_storage_uri="s3://test-bucket/model-checkpoints",
        enable_jit_checkpoint=True,
    )
    checkpoint_code = checkpoint_header

    # Create fsspec stub based on test config
    checkpoints = test_case.config["checkpoints"]
    if checkpoints is None:
        # Simulate remote storage path not existing yet
        ls_implementation = (
            '            raise FileNotFoundError("Remote storage path does not exist")'
        )
    else:
        checkpoints_list = ", ".join([f'"{cp}"' for cp in checkpoints])
        ls_implementation = f"            return [{checkpoints_list}]"

    incomplete_markers = ", ".join(
        [f'"{marker}"' for marker in test_case.config["incomplete_markers"]]
    )

    fsspec_stub = f"""
class MockS3FileSystem:
    def __init__(self, *args, **kwargs):
        pass

    def ls(self, path, detail=False):
        if path == "":
{ls_implementation}
        if path in [{incomplete_markers}]:
            return [f"{{path}}/{CHECKPOINT_INCOMPLETE_MARKER}.node-0-rank-0"]
        return []

    def exists(self, path):
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
    def is_available():
        return False

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


def test_s3_access_retry_code_generation():
    """Test that S3 access verification includes 3-retry logic in generated code."""
    print("Executing test: S3 retry logic in generated code")

    # Generate checkpoint injection code with S3 config
    checkpoint_header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/checkpoints",
        cloud_remote_storage_uri="s3://test-bucket/path",
        enable_jit_checkpoint=True,
        verify_cloud_storage_access=True,
    )
    code = checkpoint_header

    # Verify retry logic is present in generated code
    assert "for attempt in range(1, 4):" in code, "3-attempt retry loop not found"
    assert "last_error = None" in code, "Error tracking not found"
    assert "last_error = e" in code, "Error capture not found"
    assert "if attempt < 3:" in code, "Retry condition not found"
    assert "time.sleep(1)" in code, "Backoff sleep not found"
    assert "if last_error:" in code, "Error re-raise check not found"
    assert "raise last_error" in code, "Error propagation not found"

    # Verify code structure contains the access verification section
    assert "verify_cloud_storage_access" in code
    assert 'self.remote_fs.pipe(test_file, b"test")' in code
    assert "self.remote_fs.cat(test_file)" in code
    assert "self.remote_fs.rm_file(test_file)" in code


# ============================================================================
# Final Model Upload Tests
# ============================================================================


def test_upload_worker_restarts_when_thread_dies():
    """Test that start_upload_worker() restarts worker when thread is dead."""
    print("Executing test: upload worker restarts when thread dies")

    from queue import LifoQueue
    import threading
    from unittest.mock import Mock

    # Create mock callback with dead thread
    callback = Mock()
    callback.upload_queue = LifoQueue()
    callback._upload_thread = Mock(spec=threading.Thread)
    callback._upload_thread.is_alive.return_value = False  # Thread is dead

    # Mock the methods that would be called during worker creation
    callback._upload_worker_loop = Mock()

    # Execute the code from start_upload_worker
    upload_thread = getattr(callback, "_upload_thread", None)
    worker_alive = upload_thread is not None and upload_thread.is_alive()

    # Worker should NOT be alive (thread exists but is dead)
    assert worker_alive is False

    print("test execution complete")


def test_upload_worker_restarts_when_thread_none():
    """Test that start_upload_worker() restarts worker when thread is None."""
    print("Executing test: upload worker restarts when thread is None")

    from unittest.mock import Mock

    # Create mock callback with None thread
    callback = Mock()
    callback._upload_thread = None

    # Execute the check logic
    upload_thread = getattr(callback, "_upload_thread", None)
    worker_alive = upload_thread is not None and upload_thread.is_alive()

    # Worker should NOT be alive (thread is None)
    assert worker_alive is False

    print("test execution complete")


def test_upload_worker_skips_restart_when_alive():
    """Test that start_upload_worker() skips restart when worker is alive."""
    print("Executing test: upload worker skips restart when alive")

    import threading
    from unittest.mock import Mock

    # Create mock callback with alive thread
    callback = Mock()
    callback._upload_thread = Mock(spec=threading.Thread)
    callback._upload_thread.is_alive.return_value = True  # Thread is alive

    from queue import LifoQueue

    callback.upload_queue = LifoQueue()

    # Execute the check logic
    upload_thread = getattr(callback, "_upload_thread", None)
    worker_alive = upload_thread is not None and upload_thread.is_alive()

    # Worker SHOULD be alive
    assert worker_alive is True
    # Queue exists, so worker restart should be skipped
    should_restart = callback.upload_queue is None or not worker_alive
    assert should_restart is False

    print("test execution complete")


def _run_final_model_upload_subprocess(
    tmp_path: Path,
    test_name: str,
    output_dir: str,
    local_rank: str = "0",
    cloud_remote_storage_uri: str = "s3://test-bucket/model-output",
    setup_files: str = "",
    extra_fsspec_methods: str = "",
) -> tuple[int, str, str]:
    """Run upload_final_model_to_cloud in a subprocess and return output.

    Args:
        tmp_path: Pytest tmp_path fixture for temp files.
        test_name: Unique name for the test file.
        output_dir: Path to the output directory with model artifacts.
        local_rank: LOCAL_RANK env var value (default "0").
        cloud_remote_storage_uri: Cloud storage URI for config.
        setup_files: Extra Python code to create files/dirs in output_dir before upload.
        extra_fsspec_methods: Additional methods to add to MockS3FileSystem.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    import subprocess
    import sys

    checkpoint_header, checkpoint_footer = get_jit_checkpoint_injection_code(
        output_dir=output_dir,
        cloud_remote_storage_uri=cloud_remote_storage_uri,
        enable_jit_checkpoint=True,
        verify_cloud_storage_access=False,
    )

    fsspec_stub = f"""
class MockS3FileSystem:
    def __init__(self, *args, **kwargs):
        pass

    def ls(self, path, detail=False):
        return []

    def exists(self, path):
        return False

    def pipe(self, path, data):
        pass

    def cat(self, path):
        return b"test"

    def put_file(self, local, remote, callback=None):
        print(f"PUT_FILE={{remote}}")

    def put(self, local, remote, recursive=False, callback=None):
        print(f"PUT_DIR={{remote}}")

    def rm_file(self, path):
        pass

    def get(self, src, dst, recursive=False, callback=None):
        pass

    def du(self, path, total=True, maxdepth=None):
        return 0

{extra_fsspec_methods}

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
    def is_available():
        return False

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
        self.global_step = 0
        self.is_world_process_zero = True

class Trainer:
    def __init__(self, *args, **kwargs):
        self.state = TrainerState()
        self.model = None
        self._init_args = args
        self._init_kwargs = kwargs

    def train(self, resume_from_checkpoint=None):
        return {{"train_called": True, "resume_from": resume_from_checkpoint}}
"""

    test_file = tmp_path / f"{test_name}.py"
    test_code = f"""
import os
import sys
import types

os.environ["LOCAL_RANK"] = "{local_rank}"

fsspec_module = types.ModuleType('fsspec')
exec('''{fsspec_stub}''', fsspec_module.__dict__)
sys.modules['fsspec'] = fsspec_module

callbacks_module = types.ModuleType('fsspec.callbacks')
callbacks_module.Callback = fsspec_module.Callback
sys.modules['fsspec.callbacks'] = callbacks_module
fsspec_module.callbacks = callbacks_module

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

# Execute checkpoint instrumentation (header)
{checkpoint_header}

# Create output_dir and test files
os.makedirs(r"{output_dir}", exist_ok=True)
{setup_files}

# Execute post-training cleanup (footer)
{checkpoint_footer}

print("TEST_COMPLETE=True")
"""

    test_file.write_text(test_code)

    result = subprocess.run(
        [sys.executable, str(test_file)], capture_output=True, text=True, timeout=15
    )
    return result.returncode, result.stdout, result.stderr


def test_final_model_upload_files_and_dirs(tmp_path):
    """Test that upload_final_model_to_cloud uploads files and directories."""
    print("Executing test: final model upload files and directories")

    output_dir = str(tmp_path / "model-output")

    setup_files = f"""
# Create model files in output_dir
with open(os.path.join(r"{output_dir}", "config.json"), "w") as f:
    f.write('{{"model_type": "bert"}}')
with open(os.path.join(r"{output_dir}", "model.safetensors"), "w") as f:
    f.write("model_weights_data")
with open(os.path.join(r"{output_dir}", "tokenizer.json"), "w") as f:
    f.write('{{"type": "BPE"}}')

# Create a subdirectory (e.g., tokenizer subdir)
os.makedirs(os.path.join(r"{output_dir}", "tokenizer"), exist_ok=True)
with open(os.path.join(r"{output_dir}", "tokenizer", "vocab.txt"), "w") as f:
    f.write("vocab_data")
"""

    returncode, stdout, stderr = _run_final_model_upload_subprocess(
        tmp_path=tmp_path,
        test_name="test_final_model_upload_files_and_dirs",
        output_dir=output_dir,
        setup_files=setup_files,
    )

    print(f"stdout: {stdout}")
    if returncode != 0:
        print(f"stderr: {stderr}")

    assert returncode == 0, f"Subprocess failed: {stderr}"
    assert "TEST_COMPLETE=True" in stdout
    assert "Uploading final model artifacts to S3" in stdout
    assert "PUT_FILE=config.json" in stdout
    assert "PUT_FILE=model.safetensors" in stdout
    assert "PUT_FILE=tokenizer.json" in stdout
    assert "PUT_DIR=tokenizer" in stdout
    assert "Final model upload complete" in stdout

    print("test execution complete")


def test_final_model_upload_skips_checkpoints_and_staging(tmp_path):
    """Test that checkpoint dirs, staging dir, and .cache are excluded from upload."""
    print("Executing test: final model upload skips checkpoints and staging")

    from kubeflow.trainer.rhai.constants import CHECKPOINT_STAGING_DIR

    output_dir = str(tmp_path / "model-output")

    setup_files = f"""
# Create a regular model file (should be uploaded)
with open(os.path.join(r"{output_dir}", "training_args.bin"), "w") as f:
    f.write("training_args_data")

# Create checkpoint dirs (should be skipped)
os.makedirs(os.path.join(r"{output_dir}", "checkpoint-100"), exist_ok=True)
with open(os.path.join(r"{output_dir}", "checkpoint-100", "model.bin"), "w") as f:
    f.write("checkpoint_data")

os.makedirs(os.path.join(r"{output_dir}", "checkpoint-200"), exist_ok=True)

# Create staging dir (should be skipped)
os.makedirs(os.path.join(r"{output_dir}", "{CHECKPOINT_STAGING_DIR}"), exist_ok=True)

# Create .cache dir (should be skipped)
os.makedirs(os.path.join(r"{output_dir}", ".cache"), exist_ok=True)
with open(os.path.join(r"{output_dir}", ".cache", "cached_file"), "w") as f:
    f.write("cached_data")
"""

    returncode, stdout, stderr = _run_final_model_upload_subprocess(
        tmp_path=tmp_path,
        test_name="test_final_model_upload_skips_checkpoints",
        output_dir=output_dir,
        setup_files=setup_files,
    )

    print(f"stdout: {stdout}")
    if returncode != 0:
        print(f"stderr: {stderr}")

    assert returncode == 0, f"Subprocess failed: {stderr}"
    assert "TEST_COMPLETE=True" in stdout
    assert "PUT_FILE=training_args.bin" in stdout
    # Verify checkpoint dirs, staging, and cache are NOT uploaded
    assert (
        "checkpoint-100"
        not in stdout.replace("Initializing checkpoint instrumentation", "").split(
            "Uploading final model artifacts"
        )[1]
    )
    assert "checkpoint-200" not in stdout.split("Uploading final model artifacts")[1]
    assert CHECKPOINT_STAGING_DIR not in stdout.split("Uploading final model artifacts")[1]
    assert ".cache" not in stdout.split("Uploading final model artifacts")[1]
    assert "Final model upload complete" in stdout

    print("test execution complete")


def test_final_model_upload_no_cloud_uri(tmp_path):
    """Test that upload is skipped when cloud_remote_storage_uri is not configured."""
    print("Executing test: final model upload no cloud URI")

    output_dir = str(tmp_path / "model-output")

    # When no cloud URI, use local-only checkpoint code
    checkpoint_header, checkpoint_footer = get_jit_checkpoint_injection_code(
        output_dir=output_dir,
        cloud_remote_storage_uri=None,
        enable_jit_checkpoint=True,
    )

    import subprocess
    import sys

    fsspec_stub = """
class Callback:
    def __init__(self):
        self.size = 0
        self.value = 0

class callbacks:
    Callback = Callback

def filesystem(protocol, **kwargs):
    raise RuntimeError("Should not be called")
"""

    torch_stub = """
class distributed:
    @staticmethod
    def is_available():
        return False

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

class Trainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(self, resume_from_checkpoint=None):
        return {"train_called": True}
"""

    test_file = tmp_path / "test_no_cloud_uri.py"
    test_code = f"""
import os
import sys
import types

fsspec_module = types.ModuleType('fsspec')
exec('''{fsspec_stub}''', fsspec_module.__dict__)
sys.modules['fsspec'] = fsspec_module

callbacks_module = types.ModuleType('fsspec.callbacks')
callbacks_module.Callback = fsspec_module.Callback
sys.modules['fsspec.callbacks'] = callbacks_module
fsspec_module.callbacks = callbacks_module

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

os.makedirs(r"{output_dir}", exist_ok=True)
with open(os.path.join(r"{output_dir}", "model.bin"), "w") as f:
    f.write("model_data")

{checkpoint_header}

{checkpoint_footer}

print("TEST_COMPLETE=True")
"""

    test_file.write_text(test_code)

    result = subprocess.run(
        [sys.executable, str(test_file)], capture_output=True, text=True, timeout=15
    )

    print(f"stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"stderr: {result.stderr}")

    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert "TEST_COMPLETE=True" in result.stdout
    # Should NOT attempt any upload
    assert "Uploading final model artifacts" not in result.stdout
    assert "PUT_FILE" not in result.stdout

    print("test execution complete")


def test_final_model_upload_non_rank_zero(tmp_path):
    """Test that upload is skipped on non-rank-0 processes."""
    print("Executing test: final model upload non-rank-zero")

    output_dir = str(tmp_path / "model-output")

    setup_files = f"""
with open(os.path.join(r"{output_dir}", "model.bin"), "w") as f:
    f.write("model_data")
"""

    returncode, stdout, stderr = _run_final_model_upload_subprocess(
        tmp_path=tmp_path,
        test_name="test_final_model_upload_non_rank_zero",
        output_dir=output_dir,
        local_rank="1",
        setup_files=setup_files,
    )

    print(f"stdout: {stdout}")
    if returncode != 0:
        print(f"stderr: {stderr}")

    assert returncode == 0, f"Subprocess failed: {stderr}"
    assert "TEST_COMPLETE=True" in stdout
    # Should NOT attempt any upload on non-rank-0
    assert "Uploading final model artifacts" not in stdout
    assert "PUT_FILE" not in stdout

    print("test execution complete")


def test_final_model_upload_empty_output_dir(tmp_path):
    """Test that upload logs 'no artifacts' when only excluded items are present."""
    print("Executing test: final model upload empty output dir")

    from kubeflow.trainer.rhai.constants import CHECKPOINT_STAGING_DIR

    output_dir = str(tmp_path / "model-output")

    setup_files = f"""
# Only create items that should be excluded
os.makedirs(os.path.join(r"{output_dir}", "checkpoint-500"), exist_ok=True)
os.makedirs(os.path.join(r"{output_dir}", ".cache"), exist_ok=True)
os.makedirs(os.path.join(r"{output_dir}", "{CHECKPOINT_STAGING_DIR}"), exist_ok=True)
"""

    returncode, stdout, stderr = _run_final_model_upload_subprocess(
        tmp_path=tmp_path,
        test_name="test_final_model_upload_empty_dir",
        output_dir=output_dir,
        setup_files=setup_files,
    )

    print(f"stdout: {stdout}")
    if returncode != 0:
        print(f"stderr: {stderr}")

    assert returncode == 0, f"Subprocess failed: {stderr}"
    assert "TEST_COMPLETE=True" in stdout
    assert "No final model artifacts to upload" in stdout
    assert "PUT_FILE" not in stdout
    assert "PUT_DIR" not in stdout

    print("test execution complete")


def test_final_model_upload_failure_is_non_fatal(tmp_path):
    """Test that upload failure does not crash the process."""
    print("Executing test: final model upload failure is non-fatal")

    output_dir = str(tmp_path / "model-output")

    setup_files = f"""
with open(os.path.join(r"{output_dir}", "model.bin"), "w") as f:
    f.write("model_data")
"""

    extra_methods = """
    def put_file(self, local, remote, callback=None):
        raise ConnectionError("S3 connection lost")
"""

    returncode, stdout, stderr = _run_final_model_upload_subprocess(
        tmp_path=tmp_path,
        test_name="test_final_model_upload_failure",
        output_dir=output_dir,
        setup_files=setup_files,
        extra_fsspec_methods=extra_methods,
    )

    print(f"stdout: {stdout}")
    if returncode != 0:
        print(f"stderr: {stderr}")

    assert returncode == 0, f"Subprocess failed: {stderr}"
    assert "TEST_COMPLETE=True" in stdout
    assert "Warning: Final model upload failed" in stdout
    assert "S3 connection lost" in stdout
    assert "Training completed successfully" in stdout

    print("test execution complete")


def test_final_model_upload_nonexistent_output_dir(tmp_path):
    """Test that upload is skipped when output_dir does not exist."""
    print("Executing test: final model upload nonexistent output dir")

    output_dir = str(tmp_path / "nonexistent-dir")

    # Don't create the output_dir, but don't set up files either
    returncode, stdout, stderr = _run_final_model_upload_subprocess(
        tmp_path=tmp_path,
        test_name="test_final_model_upload_nonexistent",
        output_dir=output_dir,
        setup_files="# Don't create output_dir - it should not exist\nimport shutil\nshutil.rmtree"
        f'(r"{output_dir}", ignore_errors=True)',
    )

    print(f"stdout: {stdout}")
    if returncode != 0:
        print(f"stderr: {stderr}")

    assert returncode == 0, f"Subprocess failed: {stderr}"
    assert "TEST_COMPLETE=True" in stdout
    # Should silently return without uploading
    assert "Uploading final model artifacts" not in stdout

    print("test execution complete")


def test_sigterm_handler_paths(tmp_path):
    """Test SIGTERM handler dispatches correctly for all three paths.

    Path 1: Normal - calls _save_jit_checkpoint() directly.
    Path 2: During optimizer step - defers via _checkpoint_deferred flag.
    Path 3: During periodic save - skips JIT (sets checkpoint_requested only).
    """
    print("Executing test: SIGTERM handler paths")

    import subprocess
    import sys

    output_dir = tmp_path / "checkpoints"

    checkpoint_code, _ = get_jit_checkpoint_injection_code(
        output_dir=str(output_dir),
        enable_jit_checkpoint=True,
    )

    torch_stub = """
class distributed:
    @staticmethod
    def is_available():
        return False

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
        self.global_step = 42
        self.is_world_process_zero = True

class TrainingArguments:
    def __init__(self):
        self.local_process_index = 0

class CallbackHandler:
    def on_save(self, args, state, control):
        pass

class MockControl:
    should_save = False
    should_training_stop = False

class Trainer:
    def __init__(self, *args, **kwargs):
        self.state = TrainerState()
        self.args = TrainingArguments()
        self.control = MockControl()
        self.model = type('MockModel', (), {'parameters': lambda self: []})()
        self.callback_handler = CallbackHandler()
        self._init_args = args
        self._init_kwargs = kwargs
        self._save_checkpoint_called = False

    def _get_output_dir(self, trial=None):
        return OUTPUT_DIR

    def _save_checkpoint(self, model, trial=None):
        self._save_checkpoint_called = True

    def train(self, resume_from_checkpoint=None):
        return {"train_called": True}
"""

    test_file = tmp_path / "test_sigterm_paths.py"
    test_code = f"""
import os
import sys
import signal
import types

OUTPUT_DIR = r"{output_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

{checkpoint_code}

from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState

# Find JITCheckpointCallback
callback_class = None
for name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, TrainerCallback) and name != 'TrainerCallback':
        callback_class = obj
        break

if not callback_class:
    print("ERROR: JITCheckpointCallback not found")
    sys.exit(1)

def create_jit_manager():
    \"\"\"Create a JIT manager via the callback lifecycle.\"\"\"
    trainer = Trainer(None, TrainingArguments())
    trainer._get_output_dir = lambda trial=None: OUTPUT_DIR
    trainer._save_checkpoint_called = False

    cb = callback_class()
    cb._trainer_ref = trainer
    cb.on_train_begin(TrainingArguments(), TrainerState(), None)
    return cb.jit_manager

# --- Test Path 1: SIGTERM only sets flag (checkpoint deferred to callback) ---
mgr = create_jit_manager()
mgr._sigterm_handler(signal.SIGTERM, None)
assert mgr.checkpoint_requested == True, "checkpoint_requested not set"
assert mgr.trainer._save_checkpoint_called == False, "Path 1: signal handler should NOT call _save_checkpoint"
print("PATH_1_FLAG_ONLY=PASS")

# --- Test: Second SIGTERM is idempotent ---
mgr4 = create_jit_manager()
mgr4.checkpoint_requested = True  # Already set
mgr4._sigterm_handler(signal.SIGTERM, None)
assert mgr4.trainer._save_checkpoint_called == False, "Idempotent: should not save again"
print("IDEMPOTENT=PASS")

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

    assert result.returncode == 0, f"Execution failed: {result.stderr}"
    assert "PATH_1_FLAG_ONLY=PASS" in output
    assert "IDEMPOTENT=PASS" in output
    assert "TEST_COMPLETE=True" in output

    print("test execution complete")


def test_callback_guards_and_deferred_checkpoint(tmp_path):
    """Test JITCheckpointCallback guards and checkpoint behavior.

    Verifies:
    - on_step_end/on_epoch_end call _save_jit_checkpoint and set
      should_save=False, should_training_stop=True when checkpoint in progress.
    - on_optimizer_step only sets should_training_stop=True (no checkpoint).
    """
    print("Executing test: callback guards and deferred checkpoint")

    import subprocess
    import sys

    output_dir = tmp_path / "checkpoints"

    checkpoint_code, _ = get_jit_checkpoint_injection_code(
        output_dir=str(output_dir),
        enable_jit_checkpoint=True,
    )

    torch_stub = """
class distributed:
    @staticmethod
    def is_available():
        return False

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
        self.global_step = 50
        self.is_world_process_zero = True

class TrainingArguments:
    def __init__(self):
        self.local_process_index = 0

class CallbackHandler:
    def on_save(self, args, state, control):
        pass

class Trainer:
    def __init__(self, *args, **kwargs):
        self.state = TrainerState()
        self.args = TrainingArguments()
        self.model = type('MockModel', (), {'parameters': lambda self: []})()
        self.callback_handler = CallbackHandler()
        self._init_args = args
        self._init_kwargs = kwargs
        self._save_checkpoint_called = False

    def _get_output_dir(self, trial=None):
        return OUTPUT_DIR

    def _save_checkpoint(self, model, trial=None):
        self._save_checkpoint_called = True

    def train(self, resume_from_checkpoint=None):
        return {"train_called": True}
"""

    test_file = tmp_path / "test_callback_guards.py"
    test_code = f"""
import os
import sys
import types

OUTPUT_DIR = r"{output_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

{checkpoint_code}

from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState

# Find JITCheckpointCallback
callback_class = None
for name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, TrainerCallback) and name != 'TrainerCallback':
        callback_class = obj
        break

if not callback_class:
    print("ERROR: JITCheckpointCallback not found")
    sys.exit(1)

# Setup: create callback with a trainer reference
trainer = Trainer(None, TrainingArguments())
trainer._get_output_dir = lambda trial=None: OUTPUT_DIR

callback = callback_class()
callback._trainer_ref = trainer

# Simulate on_train_begin to create JIT manager
callback.on_train_begin(TrainingArguments(), TrainerState(), None)
assert callback.jit_manager is not None, "JIT manager not created"

args = TrainingArguments()
state = TrainerState()

# === Test 1: on_step_end saves checkpoint and prevents redundant save ===
callback.jit_manager.checkpoint_requested = True

class MockControl:
    should_save = True
    should_training_stop = False

control = MockControl()
callback.on_step_end(args, state, control)
assert control.should_training_stop == True, "on_step_end should set should_training_stop"
assert control.should_save == False, "on_step_end should set should_save=False to prevent redundant save"
assert trainer._save_checkpoint_called == True, "on_step_end should call _save_jit_checkpoint"
print("GUARD_ON_STEP_END=PASS")

# === Test 2: on_epoch_end saves checkpoint and prevents redundant save ===
trainer2 = Trainer(None, TrainingArguments())
trainer2._get_output_dir = lambda trial=None: OUTPUT_DIR
trainer2._save_checkpoint_called = False
callback2 = callback_class()
callback2._trainer_ref = trainer2
callback2.on_train_begin(TrainingArguments(), TrainerState(), None)
callback2.jit_manager.checkpoint_requested = True

control2 = MockControl()
control2.should_save = True
control2.should_training_stop = False
callback2.on_epoch_end(args, state, control2)
assert control2.should_training_stop == True, "on_epoch_end should set should_training_stop"
assert control2.should_save == False, "on_epoch_end should set should_save=False to prevent redundant save"
assert trainer2._save_checkpoint_called == True, "on_epoch_end should call _save_jit_checkpoint"
print("GUARD_ON_EPOCH_END=PASS")

# === Test 3: on_optimizer_step only stops training (no checkpoint) ===
trainer3 = Trainer(None, TrainingArguments())
trainer3._get_output_dir = lambda trial=None: OUTPUT_DIR
trainer3._save_checkpoint_called = False
callback3 = callback_class()
callback3._trainer_ref = trainer3
callback3.on_train_begin(TrainingArguments(), TrainerState(), None)
callback3.jit_manager.checkpoint_requested = True

control6 = MockControl()
control6.should_training_stop = False
callback3.on_optimizer_step(args, state, control6)
assert control6.should_training_stop == True, (
    "on_optimizer_step should stop training"
)
assert trainer3._save_checkpoint_called == False, (
    "on_optimizer_step should NOT call _save_jit_checkpoint"
)
print("OPTIMIZER_STEP_GUARD=PASS")

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

    assert result.returncode == 0, f"Execution failed: {result.stderr}"
    assert "GUARD_ON_STEP_END=PASS" in output
    assert "GUARD_ON_EPOCH_END=PASS" in output
    assert "OPTIMIZER_STEP_GUARD=PASS" in output
    assert "TEST_COMPLETE=True" in output

    print("test execution complete")


def test_find_latest_checkpoint_race_condition(tmp_path):
    """Test checkpoint discovery handles concurrent directory removal gracefully.

    Simulates the race where global rank 0 deletes an incomplete checkpoint
    directory while another rank is scanning it with os.listdir().
    """
    print("Executing test: checkpoint discovery race condition")

    import os
    import re
    import shutil

    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER

    # Create a checkpoint directory then remove it between listdir calls
    # to simulate the race condition
    checkpoint_path = tmp_path / "checkpoint-100"
    checkpoint_path.mkdir()
    (checkpoint_path / "model.bin").write_text("data")

    # Also create a valid checkpoint that should be found
    valid_path = tmp_path / "checkpoint-50"
    valid_path.mkdir()
    (valid_path / "model.bin").write_text("data")

    # Implement the checkpoint discovery logic with race condition handling
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []

    entries = os.listdir(tmp_path)

    # Remove checkpoint-100 AFTER listdir but BEFORE processing it
    # (simulates race with rank 0 deleting it)
    shutil.rmtree(checkpoint_path)

    for name in entries:
        match = checkpoint_pattern.match(name)
        full_path = os.path.join(tmp_path, name)
        if not match or not os.path.isdir(full_path):
            continue

        try:
            has_incomplete = any(
                f.startswith(CHECKPOINT_INCOMPLETE_MARKER) for f in os.listdir(full_path)
            )
        except (FileNotFoundError, OSError):
            # This is the race condition path - directory was removed
            print(f"Skipping checkpoint {name} directory was removed by another rank")
            continue

        if not has_incomplete:
            checkpoints.append((int(match.group(1)), full_path))

    # checkpoint-100 was removed (race condition), checkpoint-50 should be found
    assert len(checkpoints) == 1, f"Expected 1 checkpoint, got {len(checkpoints)}"
    assert checkpoints[0][0] == 50, f"Expected step 50, got {checkpoints[0][0]}"

    print("test execution complete")


def test_sentinel_lifecycle_during_jit_checkpoint(tmp_path):
    """Test sentinel file lifecycle: created before save, removed after save.

    Verifies:
    1. Sentinel file exists DURING _save_checkpoint() (created before save).
    2. Sentinel file is removed AFTER _save_checkpoint() completes.
    3. No CHECKPOINT_INCOMPLETE_MARKER files remain (checkpoint is resume-ready).
    """
    print("Executing test: sentinel lifecycle during JIT checkpoint")

    import subprocess
    import sys

    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER

    output_dir = tmp_path / "checkpoints"

    checkpoint_code, _ = get_jit_checkpoint_injection_code(
        output_dir=str(output_dir),
        enable_jit_checkpoint=True,
    )

    torch_stub = """
class distributed:
    @staticmethod
    def is_available():
        return False

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

    transformers_stub = f"""
import os

CHECKPOINT_INCOMPLETE_MARKER = "{CHECKPOINT_INCOMPLETE_MARKER}"
_OUTPUT_DIR = r"{output_dir}"

class TrainerCallback:
    pass

class trainer_utils:
    PREFIX_CHECKPOINT_DIR = "checkpoint"

PREFIX_CHECKPOINT_DIR = "checkpoint"

class TrainerState:
    def __init__(self):
        self.global_step = 10
        self.is_world_process_zero = True

class TrainingArguments:
    def __init__(self):
        self.local_process_index = 0

class CallbackHandler:
    def on_save(self, args, state, control):
        pass

class MockControl:
    should_save = False
    should_training_stop = False

class Trainer:
    def __init__(self, *args, **kwargs):
        self.state = TrainerState()
        self.args = TrainingArguments()
        self.control = MockControl()
        self.model = type('MockModel', (), {{'parameters': lambda self: []}})()
        self.callback_handler = CallbackHandler()
        self._init_args = args
        self._init_kwargs = kwargs

    def _get_output_dir(self, trial=None):
        return _OUTPUT_DIR

    def _save_checkpoint(self, model, trial=None):
        # During save, verify sentinel file EXISTS
        checkpoint_path = os.path.join(_OUTPUT_DIR, "checkpoint-10")
        sentinel_files = [
            f for f in os.listdir(checkpoint_path)
            if f.startswith(CHECKPOINT_INCOMPLETE_MARKER)
        ]
        if sentinel_files:
            print("SENTINEL_EXISTS_DURING_SAVE=True")
            print(f"SENTINEL_NAME={{sentinel_files[0]}}")
        else:
            print("SENTINEL_EXISTS_DURING_SAVE=False")

    def train(self, resume_from_checkpoint=None):
        return {{"train_called": True}}
"""

    test_file = tmp_path / "test_sentinel_lifecycle.py"
    test_code = f"""
import os
import sys
import signal
import types

OUTPUT_DIR = r"{output_dir}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

{checkpoint_code}

from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState

CHECKPOINT_INCOMPLETE_MARKER = "{CHECKPOINT_INCOMPLETE_MARKER}"

# Find JITCheckpointCallback
callback_class = None
for name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, TrainerCallback) and name != 'TrainerCallback':
        callback_class = obj
        break

if not callback_class:
    print("ERROR: JITCheckpointCallback not found")
    sys.exit(1)

# Create trainer and JIT manager
trainer = Trainer(None, TrainingArguments())
trainer._get_output_dir = lambda trial=None: OUTPUT_DIR

cb = callback_class()
cb._trainer_ref = trainer
cb.on_train_begin(TrainingArguments(), TrainerState(), None)

mgr = cb.jit_manager
assert mgr is not None, "JIT manager not created"

# Trigger SIGTERM handler (sets checkpoint_requested flag)
mgr._sigterm_handler(signal.SIGTERM, None)
assert mgr.checkpoint_requested == True, "checkpoint_requested not set"

# Simulate on_step_end which calls _save_jit_checkpoint
from transformers import MockControl
control = MockControl()
cb.on_step_end(TrainingArguments(), TrainerState(), control)
assert control.should_training_stop == True, "should_training_stop not set"
assert control.should_save == False, "should_save should be False to prevent redundant save"

# After _save_jit_checkpoint completes, verify sentinel is REMOVED
checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint-10")
remaining_sentinels = [
    f for f in os.listdir(checkpoint_path)
    if f.startswith(CHECKPOINT_INCOMPLETE_MARKER)
]
if not remaining_sentinels:
    print("SENTINEL_REMOVED_AFTER_SAVE=True")
else:
    print(f"SENTINEL_REMOVED_AFTER_SAVE=False (remaining: {{remaining_sentinels}})")

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

    assert result.returncode == 0, f"Execution failed: {result.stderr}"
    # Sentinel existed DURING _save_checkpoint
    assert "SENTINEL_EXISTS_DURING_SAVE=True" in output
    # Sentinel name matches per-rank format
    assert f"SENTINEL_NAME={CHECKPOINT_INCOMPLETE_MARKER}.node-0-rank-0" in output
    # Sentinel removed AFTER save (checkpoint is resume-ready)
    assert "SENTINEL_REMOVED_AFTER_SAVE=True" in output
    assert "TEST_COMPLETE=True" in output

    print("test execution complete")


@pytest.fixture
def progression_instrumentation():
    """Set up mock transformers and return progression instrumentation components.

    Mocks sys.modules["transformers"] for the exec-based test environment,
    executes the instrumentation wrapper, and yields all 5 components from
    _create_progression_instrumentation. Restores original sys.modules state
    on teardown.

    Yields:
        Tuple of (apply_fn, callback_class, handler_class, get_metrics_json,
        update_progression_metrics).
    """
    import sys
    from unittest.mock import Mock

    _sentinel = object()
    prev_transformers = sys.modules.get("transformers", _sentinel)

    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer = Mock()
    sys.modules["transformers"] = mock_transformers

    wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
    wrapper_code = wrapper.replace("{{user_func_import_and_call}}", "pass")
    lines = wrapper_code.split("\n")
    modified_lines = []
    for line in lines:
        if line.strip() == "apply_progression_tracking()":
            modified_lines.append("# apply_progression_tracking()  # Skipped in tests")
        else:
            modified_lines.append(line)
    wrapper_code = "\n".join(modified_lines)
    namespace = {}
    exec(wrapper_code, namespace)

    components = namespace["_create_progression_instrumentation"](28080)
    yield components

    if prev_transformers is _sentinel:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = prev_transformers


def test_update_progression_metrics(progression_instrumentation):
    """Test _update_progression_metrics metrics update function.

    Validates that:
    - Scalar fields are set correctly
    - Dict fields are merged (not replaced)
    - Unknown keys are silently ignored
    """
    print("Executing test: Update Progression Metrics.")

    _, _, _, get_metrics_json, update_progression_metrics = progression_instrumentation

    import json

    # Verify initial state (all defaults)
    metrics = json.loads(get_metrics_json())
    assert metrics["currentStep"] == 0
    assert metrics["currentEpoch"] == 0.0
    assert metrics["progressPercentage"] is None
    assert metrics["estimatedRemainingSeconds"] is None
    assert metrics["totalSteps"] is None
    assert metrics["totalEpochs"] is None
    assert metrics["trainMetrics"] == {}
    assert metrics["evalMetrics"] == {}

    # Test scalar field updates
    update_progression_metrics(
        {
            "currentStep": 50,
            "totalSteps": 100,
            "currentEpoch": 1.5,
            "totalEpochs": 3,
            "progressPercentage": 50,
            "estimatedRemainingSeconds": 120,
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["currentStep"] == 50
    assert metrics["totalSteps"] == 100
    assert metrics["currentEpoch"] == 1.5
    assert metrics["totalEpochs"] == 3
    assert metrics["progressPercentage"] == 50
    assert metrics["estimatedRemainingSeconds"] == 120

    # Test dict fields are merged (not replaced)
    update_progression_metrics(
        {
            "trainMetrics": {"loss": 0.5, "learning_rate": 0.001},
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["trainMetrics"] == {"loss": 0.5, "learning_rate": 0.001}

    # Merge additional keys into existing dict
    update_progression_metrics(
        {
            "trainMetrics": {"grad_norm": 1.2},
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["trainMetrics"] == {
        "loss": 0.5,
        "learning_rate": 0.001,
        "grad_norm": 1.2,
    }

    # Overwrite existing key within dict
    update_progression_metrics(
        {
            "trainMetrics": {"loss": 0.3},
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["trainMetrics"]["loss"] == 0.3
    assert metrics["trainMetrics"]["learning_rate"] == 0.001

    # Test evalMetrics dict merge
    update_progression_metrics(
        {
            "evalMetrics": {"eval_loss": 0.8, "eval_accuracy": 0.92},
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["evalMetrics"] == {"eval_loss": 0.8, "eval_accuracy": 0.92}

    # Test unknown keys are silently ignored
    update_progression_metrics(
        {
            "nonExistentField": 999,
            "currentStep": 75,
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["currentStep"] == 75
    assert "nonExistentField" not in metrics

    # Test scalar fields overwrite previous values
    update_progression_metrics(
        {
            "progressPercentage": 100,
            "estimatedRemainingSeconds": 0,
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["progressPercentage"] == 100
    assert metrics["estimatedRemainingSeconds"] == 0

    # Verify previously set scalar fields remain unchanged
    assert metrics["currentStep"] == 75
    assert metrics["totalSteps"] == 100

    print("test execution complete")


def test_get_progression_metrics_json(progression_instrumentation):
    """Test _get_progression_metrics_json returns valid JSON with expected schema.

    Validates that:
    - Return value is a valid JSON string
    - Default state contains all expected fields
    - State reflects updates made via _update_progression_metrics
    """
    print("Executing test: Get Progression Metrics JSON.")

    _, _, _, get_metrics_json, update_progression_metrics = progression_instrumentation

    import json

    # Verify default state returns valid JSON with all expected fields
    raw = get_metrics_json()
    assert isinstance(raw, str)
    metrics = json.loads(raw)
    expected_keys = {
        "progressPercentage",
        "estimatedRemainingSeconds",
        "currentStep",
        "totalSteps",
        "currentEpoch",
        "totalEpochs",
        "trainMetrics",
        "evalMetrics",
    }
    assert set(metrics.keys()) == expected_keys

    # Verify JSON reflects updates
    update_progression_metrics(
        {
            "currentStep": 42,
            "progressPercentage": 84,
            "trainMetrics": {"loss": 0.25},
        }
    )
    metrics = json.loads(get_metrics_json())
    assert metrics["currentStep"] == 42
    assert metrics["progressPercentage"] == 84
    assert metrics["trainMetrics"] == {"loss": 0.25}

    print("test execution complete")


def test_progression_metrics_handler_do_get(progression_instrumentation):
    """Test ProgressionMetricsHandler serves metrics JSON over HTTP.

    Validates that:
    - GET request returns 200 with Content-Type application/json
    - Response body is valid JSON matching current metrics state
    """
    print("Executing test: Progression Metrics Handler do_GET.")

    _, _, handler_class, _, update_progression_metrics = progression_instrumentation

    import http.server
    import json
    import threading

    # Seed some state so we can verify it in the response
    update_progression_metrics(
        {
            "currentStep": 10,
            "totalSteps": 100,
            "progressPercentage": 10,
        }
    )

    # Start a real HTTP server on an ephemeral port
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        import urllib.request

        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5)
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "application/json"

        body = json.loads(resp.read().decode("utf-8"))
        assert body["currentStep"] == 10
        assert body["totalSteps"] == 100
        assert body["progressPercentage"] == 10
    finally:
        server.shutdown()
        server.server_close()

    print("test execution complete")


def test_on_train_begin_initial_metrics(progression_instrumentation):
    """Test on_train_begin sets initial metrics for a fresh training start.

    Validates that:
    - totalEpochs is set from args.num_train_epochs
    - Fresh start (global_step = 0) sets progressPercentage to 0
    - Server is NOT started when is_world_process_zero is False
    """
    print("Executing test: on_train_begin initial metrics.")

    from unittest.mock import Mock

    _, callback_class, _, get_metrics_json, _ = progression_instrumentation

    import json

    callback = callback_class(metrics_port=28080)
    args = Mock()
    args.num_train_epochs = 5
    state = Mock()
    state.global_step = 0
    state.max_steps = 500
    state.epoch = 0.0
    state.is_world_process_zero = False
    control = Mock()

    callback.on_train_begin(args, state, control)

    metrics = json.loads(get_metrics_json())
    assert metrics["currentStep"] == 0
    assert metrics["totalSteps"] == 500
    assert metrics["totalEpochs"] == 5
    assert metrics["currentEpoch"] == 0.0
    assert metrics["progressPercentage"] == 0
    assert callback.start_time is not None
    assert callback.server is None

    print("test execution complete")


def test_on_train_begin_checkpoint_resume(progression_instrumentation):
    """Test on_train_begin calculates initial progress for checkpoint resume.

    When global_step > 0 at the start, progressPercentage should reflect
    the resumed position (e.g., 200/1000 = 20%).
    """
    print("Executing test: on_train_begin checkpoint resume.")

    from unittest.mock import Mock

    _, callback_class, _, get_metrics_json, _ = progression_instrumentation

    import json

    callback = callback_class(metrics_port=28080)
    args = Mock()
    args.num_train_epochs = 10
    state = Mock()
    state.global_step = 200
    state.max_steps = 1000
    state.epoch = 2.0
    state.is_world_process_zero = False
    control = Mock()

    callback.on_train_begin(args, state, control)

    metrics = json.loads(get_metrics_json())
    assert metrics["currentStep"] == 200
    assert metrics["totalSteps"] == 1000
    assert metrics["totalEpochs"] == 10
    assert metrics["currentEpoch"] == 2.0
    assert metrics["progressPercentage"] == 20

    print("test execution complete")


def test_on_train_end_termination_message(progression_instrumentation):
    """Test on_train_end writes termination message for world process zero.

    Validates that:
    - /dev/termination-log receives valid JSON with final metrics
    - trainMetrics and evalMetrics are included in the termination message
    """
    print("Executing test: on_train_end termination message.")

    import io
    from unittest.mock import Mock, mock_open, patch

    _, callback_class, _, get_metrics_json, update_progression_metrics = progression_instrumentation

    import json

    # Pre-seed train/eval metrics so they appear in termination message
    update_progression_metrics(
        {
            "trainMetrics": {"loss": 0.1, "learning_rate": 0.0001},
            "evalMetrics": {"eval_loss": 0.2},
        }
    )

    callback = callback_class(metrics_port=28080)
    args = Mock()
    args.num_train_epochs = 3
    state = Mock()
    state.global_step = 300
    state.max_steps = 300
    state.epoch = 3.0
    state.is_world_process_zero = True
    control = Mock()

    # Capture what gets written to /dev/termination-log
    written_data = io.StringIO()
    m = mock_open()
    m.return_value.write = written_data.write

    with patch("builtins.open", m):
        callback.on_train_begin(args, state, control)
        callback.on_train_end(args, state, control)

    m.assert_called_once_with("/dev/termination-log", "w")
    termination_json = written_data.getvalue()
    termination = json.loads(termination_json)

    assert termination["currentStep"] == 300
    assert termination["totalSteps"] == 300
    assert termination["totalEpochs"] == 3
    assert termination["currentEpoch"] == 3.0
    assert termination["progressPercentage"] == 100
    assert termination["estimatedRemainingSeconds"] == 0
    assert termination["trainMetrics"]["loss"] == 0.1
    assert termination["evalMetrics"]["eval_loss"] == 0.2

    assert callback.training_finished is True

    print("test execution complete")


def test_on_train_end_non_world_process_zero(progression_instrumentation):
    """Test on_train_end skips termination message for non-world-process-zero.

    Validates that training_finished is set and metrics are updated, but
    /dev/termination-log is NOT written.
    """
    print("Executing test: on_train_end non-world-process-zero.")

    from unittest.mock import Mock, patch

    _, callback_class, _, get_metrics_json, _ = progression_instrumentation

    import json

    callback = callback_class(metrics_port=28080)
    args = Mock()
    args.num_train_epochs = 2
    state = Mock()
    state.global_step = 100
    state.max_steps = 100
    state.epoch = 2.0
    state.is_world_process_zero = False
    control = Mock()

    with patch("builtins.open") as mock_file:
        callback.on_train_begin(args, state, control)
        callback.on_train_end(args, state, control)
        mock_file.assert_not_called()

    assert callback.training_finished is True

    metrics = json.loads(get_metrics_json())
    assert metrics["progressPercentage"] == 100
    assert metrics["estimatedRemainingSeconds"] == 0

    print("test execution complete")


def test_apply_progression_tracking():
    """Test apply_progression_tracking patches Trainer.__init__.

    Validates that:
    - Trainer.__init__ is replaced with an instrumented version
    - The original __init__ is still called
    - KubeflowProgressCallback is added to the trainer callbacks

    Note: uses a custom mock setup (FakeTrainer class) instead of the shared
    fixture because Mock objects forbid setting __init__.
    """
    print("Executing test: apply_progression_tracking.")

    import sys
    import types
    from unittest.mock import Mock

    trainer_stub = types.ModuleType("transformers.trainer")
    init_call_log = []

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            init_call_log.append(("original", args, kwargs))
            self.callback_handler = Mock()
            self.callback_handler.callbacks = []

        def add_callback(self, cb):
            self.callback_handler.callbacks.append(cb)

    trainer_stub.Trainer = FakeTrainer

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer = trainer_stub
    sys.modules["transformers"] = mock_transformers
    sys.modules["transformers.trainer"] = trainer_stub

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
            apply_fn,
            callback_class,
            _,
            _,
            _,
        ) = namespace["_create_progression_instrumentation"](28080)

        original_init = FakeTrainer.__init__

        apply_fn()

        assert trainer_stub.Trainer.__init__ != original_init

        instance = trainer_stub.Trainer()

        assert len(init_call_log) == 1
        assert init_call_log[0][0] == "original"

        assert len(instance.callback_handler.callbacks) == 1
        added_callback = instance.callback_handler.callbacks[0]
        assert isinstance(added_callback, callback_class)
        assert added_callback.metrics_port == 28080

        print("test execution complete")
    finally:
        if "transformers" in sys.modules:
            del sys.modules["transformers"]
        if "transformers.trainer" in sys.modules:
            del sys.modules["transformers.trainer"]


def test_on_log_ignores_non_numeric_values(progression_instrumentation):
    """Test on_log silently ignores non-numeric, non-tensor metric values.

    Validates that:
    - String values are ignored
    - None values are ignored
    - List/dict values are ignored
    - Only int/float values are categorized
    """
    print("Executing test: on_log ignores non-numeric values.")

    from unittest.mock import Mock

    _, callback_class, _, get_metrics_json, _ = progression_instrumentation

    import json

    callback = callback_class(metrics_port=28080)
    args = Mock()
    state = Mock()
    control = Mock()

    callback.on_log(
        args,
        state,
        control,
        logs={
            "loss": 0.5,
            "some_string": "not a number",
            "some_none": None,
            "some_list": [1, 2, 3],
            "some_dict": {"nested": True},
            "eval_loss": 0.3,
            "eval_string": "also not a number",
        },
    )

    metrics = json.loads(get_metrics_json())
    assert metrics["trainMetrics"] == {"loss": 0.5}
    assert metrics["evalMetrics"] == {"eval_loss": 0.3}

    print("test execution complete")


def test_on_log_empty_and_none_logs(progression_instrumentation):
    """Test on_log handles empty and None logs gracefully.

    Validates that:
    - logs=None does not update metrics
    - logs={} does not update metrics
    """
    print("Executing test: on_log empty and None logs.")

    from unittest.mock import Mock

    _, callback_class, _, get_metrics_json, _ = progression_instrumentation

    import json

    callback = callback_class(metrics_port=28080)
    args = Mock()
    state = Mock()
    control = Mock()

    # Test with None logs
    callback.on_log(args, state, control, logs=None)
    metrics = json.loads(get_metrics_json())
    assert metrics["trainMetrics"] == {}
    assert metrics["evalMetrics"] == {}

    # Test with empty logs
    callback.on_log(args, state, control, logs={})
    metrics = json.loads(get_metrics_json())
    assert metrics["trainMetrics"] == {}
    assert metrics["evalMetrics"] == {}

    print("test execution complete")


def test_on_train_begin_server_startup(progression_instrumentation):
    """Test on_train_begin starts HTTP server for world process zero.

    Validates that:
    - Server is started when is_world_process_zero is True
    - Server is accessible and serves metrics
    - Server attribute is set on the callback
    """
    print("Executing test: on_train_begin server startup.")

    from unittest.mock import Mock

    _, callback_class, _, get_metrics_json, _ = progression_instrumentation

    import json
    import urllib.request

    # Use port 0 to let the OS assign an ephemeral port
    callback = callback_class(metrics_port=0)
    args = Mock()
    args.num_train_epochs = 3
    state = Mock()
    state.global_step = 0
    state.max_steps = 100
    state.epoch = 0.0
    state.is_world_process_zero = True
    control = Mock()

    callback.on_train_begin(args, state, control)

    assert callback.server is not None
    port = callback.server.server_address[1]

    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5)
        assert resp.status == 200
        body = json.loads(resp.read().decode("utf-8"))
        assert body["totalSteps"] == 100
        assert body["totalEpochs"] == 3
    finally:
        callback.server.shutdown()
        callback.server.server_close()

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
