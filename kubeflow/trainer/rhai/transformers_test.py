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
                ("currentEpoch: int = 0", True),
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
    user_code_marker_index = next(
        i for i, line in enumerate(lines) if "USER TRAINING CODE" in line
    )
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
    assert "lr=0.001" in command_str
    assert "batch_size=32" in command_str

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
