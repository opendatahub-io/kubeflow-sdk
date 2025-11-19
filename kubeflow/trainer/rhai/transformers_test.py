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
    )

    assert isinstance(wrapper, str)
    assert "from kubeflow.trainer.rhai" not in wrapper
    assert "import kubeflow" not in wrapper
    assert "class KubeflowProgressCallback" in wrapper
    assert "class ProgressionMetricsHandler" in wrapper
    assert "def apply_progression_tracking" in wrapper
    assert "{{user_func_import_and_call}}" in wrapper
    assert f"metrics_port = {28080}" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_self_contained():
    """Test that wrapper is self-contained (no SDK dependency)."""
    print("Executing test: Wrapper is self-contained")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
    )

    assert "from kubeflow" not in wrapper
    assert "import kubeflow" not in wrapper
    assert "class ProgressionMetricsHandler" in wrapper
    assert "class KubeflowProgressCallback" in wrapper
    assert "def apply_progression_tracking" in wrapper
    assert "from transformers import" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_structure():
    """Test wrapper structure and function call."""
    print("Executing test: Wrapper structure")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
    )

    assert "apply_progression_tracking" in wrapper
    assert "metrics_port = 28080" in wrapper
    assert "apply_progression_tracking()" in wrapper
    assert "# USER TRAINING CODE" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_completeness():
    """Test wrapper contains all implementation details (self-contained)."""
    print("Executing test: Wrapper completeness")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
    )

    assert "def on_train_begin" in wrapper
    assert "def on_step_end" in wrapper
    assert "def on_log" in wrapper
    assert "def on_train_end" in wrapper
    assert "_original_init" in wrapper
    assert "def _instrumented_trainer_init" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_user_code_placeholder():
    """Test that wrapper has placeholder for user code injection."""
    print("Executing test: User code placeholder")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
    )

    assert "{{user_func_import_and_call}}" in wrapper
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
        )
        assert str(port) in wrapper
        print(f"  ✓ Port {port} correctly embedded")

    print("test execution complete")


def test_instrumentation_wrapper_contains_implementation():
    """Test that wrapper contains all implementation details (self-contained)."""
    print("Executing test: Wrapper contains implementation")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
    )

    assert "from kubeflow.trainer.rhai" not in wrapper
    assert "state.global_step" in wrapper
    assert "progress_pct" in wrapper
    assert "elapsed_sec" in wrapper

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


def test_instrumentation_wrapper_contains_dataclass():
    """Test that wrapper contains the ProgressionMetricsState dataclass definition."""
    print("Executing test: Wrapper contains dataclass")

    wrapper = get_transformers_instrumentation_wrapper(
        metrics_port=28080,
    )

    assert "@dataclass" in wrapper
    assert "class ProgressionMetricsState" in wrapper
    assert "progressPercentage" in wrapper
    assert "estimatedRemainingSeconds" in wrapper
    assert "currentStep" in wrapper
    assert "totalSteps" in wrapper
    assert "trainMetrics" in wrapper
    assert "evalMetrics" in wrapper
    assert "asdict" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_different_ports():
    """Test that different ports are correctly embedded in wrapper."""
    print("Executing test: Different port values")

    test_ports = [8080, 9090, 28080, 28090, 30000]

    for port in test_ports:
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=port)
        assert f"metrics_port = {port}" in wrapper
        print(f"  ✓ Port {port} correctly set")

    print("test execution complete")


def test_instrumentation_wrapper_metrics_state_initialization():
    """Test that metrics state is properly initialized in wrapper."""
    print("Executing test: Metrics state initialization")

    wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)

    assert "progressPercentage: Optional[int] = None" in wrapper
    assert "currentStep: int = 0" in wrapper
    assert "currentEpoch: int = 0" in wrapper
    assert "trainMetrics: dict[str, Any] = field(default_factory=dict)" in wrapper
    assert "evalMetrics: dict[str, Any] = field(default_factory=dict)" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_thread_safety():
    """Test that wrapper includes thread-safe update mechanisms."""
    print("Executing test: Thread safety in wrapper")

    wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)

    assert "_progression_metrics_lock" in wrapper
    assert "threading.Lock()" in wrapper
    assert "with _progression_metrics_lock:" in wrapper
    assert "_update_progression_metrics" in wrapper

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
    assert "metrics_port = 8888" in command_str

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
