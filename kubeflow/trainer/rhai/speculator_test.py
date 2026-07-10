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

"""Tests for SpeculativeDecodingTrainer and CRD conversion."""

import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.speculator import (
    SpeculativeDecodingTrainer,
    SpeculatorMode,
    SpeculatorType,
    _create_speculator_progression_instrumentation,
    _render_speculator_training_script,
    get_trainer_cr_from_speculator_trainer,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


def test_speculator_trainer_initialization():
    """Test SpeculativeDecodingTrainer initialization with default values."""
    print("Executing test: SpeculativeDecodingTrainer initialization with defaults")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
    )

    assert trainer.verifier_name_or_path == "Qwen/Qwen3-8B"
    assert trainer.speculator_type == SpeculatorType.EAGLE3
    assert trainer.mode == SpeculatorMode.TRAIN_ONLY
    assert trainer.hidden_states_path == "/data/hidden_states"
    assert trainer.draft_vocab_size is None
    assert trainer.epochs == 3
    assert trainer.lr == 1e-4
    assert trainer.total_seq_len == 8192
    assert trainer.hidden_states_dtype == "bfloat16"
    assert trainer.num_nodes is None
    assert trainer.resources_per_node is None
    assert trainer.packages_to_install is None
    assert trainer.pip_index_urls == list(constants.DEFAULT_PIP_INDEX_URLS)
    assert trainer.env is None
    assert trainer.output_dir == "pvc://test-pvc/output"
    assert trainer.enable_progression_tracking is True
    assert trainer.metrics_port == 28080
    assert trainer.metrics_poll_interval_seconds == 30

    print("test execution complete")


def test_speculator_trainer_with_custom_config():
    """Test SpeculativeDecodingTrainer with custom configuration."""
    print("Executing test: SpeculativeDecodingTrainer with custom configuration")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="meta-llama/Llama-3.1-70B",
        speculator_type=SpeculatorType.EAGLE3,
        mode=SpeculatorMode.TRAIN_ONLY,
        hidden_states_path="/mnt/pvc/hidden_states",
        output_dir="pvc://shared/checkpoints/eagle3",
        draft_vocab_size=32000,
        epochs=5,
        lr=5e-5,
        total_seq_len=4096,
        num_nodes=2,
        resources_per_node={"nvidia.com/gpu": 4},
        packages_to_install=["speculators"],
        pip_index_urls=["https://custom.pypi.org/simple"],
        env={"WANDB_DISABLED": "true"},
        enable_progression_tracking=True,
        metrics_port=28090,
        metrics_poll_interval_seconds=60,
    )

    assert trainer.verifier_name_or_path == "meta-llama/Llama-3.1-70B"
    assert trainer.speculator_type == SpeculatorType.EAGLE3
    assert trainer.hidden_states_path == "/mnt/pvc/hidden_states"
    assert trainer.output_dir == "pvc://shared/checkpoints/eagle3"
    assert trainer.draft_vocab_size == 32000
    assert trainer.epochs == 5
    assert trainer.lr == 5e-5
    assert trainer.total_seq_len == 4096
    assert trainer.num_nodes == 2
    assert trainer.resources_per_node == {"nvidia.com/gpu": 4}
    assert trainer.packages_to_install == ["speculators"]
    assert trainer.pip_index_urls == ["https://custom.pypi.org/simple"]
    assert trainer.env == {"WANDB_DISABLED": "true"}
    assert trainer.metrics_port == 28090
    assert trainer.metrics_poll_interval_seconds == 60

    print("test execution complete")


def test_speculator_mode_train_only_requires_hidden_states():
    """Test that TRAIN_ONLY mode requires hidden_states_path."""
    print("Executing test: TRAIN_ONLY mode requires hidden_states_path")

    with pytest.raises(ValueError, match="hidden_states_path is required for TRAIN_ONLY mode"):
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
        )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="DATA_ONLY mode not yet supported",
            expected_status=FAILED,
            config={"mode": SpeculatorMode.DATA_ONLY},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="OFFLINE mode not yet supported",
            expected_status=FAILED,
            config={"mode": SpeculatorMode.OFFLINE},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="ONLINE mode not yet supported",
            expected_status=FAILED,
            config={"mode": SpeculatorMode.ONLINE},
            expected_error=NotImplementedError,
        ),
    ],
)
def test_unsupported_mode_validation(test_case):
    """Test that unsupported modes raise NotImplementedError."""
    print(f"Executing test: {test_case.name}")

    try:
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="pvc://test-pvc/output",
            **test_case.config,
        )

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="port too low (1023)",
            expected_status=FAILED,
            config={"metrics_port": 1023},
            expected_error=ValueError,
        ),
        TestCase(
            name="port too high (65536)",
            expected_status=FAILED,
            config={"metrics_port": 65536},
            expected_error=ValueError,
        ),
        TestCase(
            name="port minimum boundary (1024)",
            expected_status=SUCCESS,
            config={"metrics_port": 1024},
            expected_output=1024,
        ),
        TestCase(
            name="port maximum boundary (65535)",
            expected_status=SUCCESS,
            config={"metrics_port": 65535},
            expected_output=65535,
        ),
    ],
)
def test_metrics_port_validation(test_case):
    """Test metrics_port validation."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="pvc://test-pvc/output",
            metrics_port=test_case.config["metrics_port"],
        )

        assert test_case.expected_status == SUCCESS
        assert trainer.metrics_port == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="poll interval too low (4 seconds)",
            expected_status=FAILED,
            config={"metrics_poll_interval_seconds": 4},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval too high (301 seconds)",
            expected_status=FAILED,
            config={"metrics_poll_interval_seconds": 301},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval minimum boundary (5 seconds)",
            expected_status=SUCCESS,
            config={"metrics_poll_interval_seconds": 5},
            expected_output=5,
        ),
        TestCase(
            name="poll interval maximum boundary (300 seconds)",
            expected_status=SUCCESS,
            config={"metrics_poll_interval_seconds": 300},
            expected_output=300,
        ),
    ],
)
def test_metrics_poll_interval_validation(test_case):
    """Test metrics_poll_interval_seconds validation."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="pvc://test-pvc/output",
            metrics_poll_interval_seconds=test_case.config["metrics_poll_interval_seconds"],
        )

        assert test_case.expected_status == SUCCESS
        assert trainer.metrics_poll_interval_seconds == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


def test_non_pvc_output_dir_not_supported():
    """Test that non-PVC output_dir raises NotImplementedError."""
    print("Executing test: non-PVC output_dir not supported")

    with pytest.raises(NotImplementedError, match="is not yet supported"):
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="s3://my-bucket/checkpoints",
        )

    with pytest.raises(ValueError, match="Unsupported storage URI scheme"):
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="gcs://my-bucket/checkpoints",
        )

    print("test execution complete")


def test_pvc_output_dir_normalized():
    """Test that PVC output_dir is normalized."""
    print("Executing test: PVC output_dir normalization")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://my-pvc/checkpoints/",
    )

    assert trainer.output_dir == "pvc://my-pvc/checkpoints"

    print("test execution complete")


def test_training_script_content():
    """Test that generated training script contains expected CLI arguments."""
    print("Executing test: Training script content")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        draft_vocab_size=32000,
        epochs=5,
        lr=1e-5,
    )

    script = _render_speculator_training_script(trainer)

    assert "from speculators.models.eagle3.core import Eagle3DraftModel" in script
    assert "from speculators.train.trainer import Trainer, TrainerConfig" in script
    assert "from speculators.train.noise_transforms import AddUniformNoise" in script
    assert "ArrowDataset" in script
    assert "SampleFileDataset" not in script
    assert "split_files" not in script
    assert "Eagle3DraftModel.from_training_args" in script
    assert "Qwen/Qwen3-8B" in script
    assert "/data/hidden_states" in script
    assert "draft_vocab_size=32000" in script
    assert "norm_before_residual=True" in script
    assert "ttt_steps=3" in script
    assert "max_len = total_seq_len" in script
    assert "total_seq_len=8192" in script
    assert "create_collate_fn(max_len, verifier_config.hidden_size)" in script
    assert "AddUniformNoise()" in script
    assert 'on_missing="skip"' in script
    assert "split_ratio=0.9" in script
    assert "split_ratio=-0.1" in script
    assert "hidden_states_dtype='bfloat16'" in script
    assert "getattr(torch, hidden_states_dtype)" in script
    assert "MultipackDistributedBatchSamplerV2" in script
    assert "trainer.run_training()" in script

    print("test execution complete")


def test_training_script_with_pvc_output_dir():
    """Test that output_dir PVC URI resolves to correct save_path in training script."""
    print("Executing test: Training script with PVC output_dir")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://shared/speculator_output",
    )

    script = _render_speculator_training_script(trainer)

    assert "/mnt/kubeflow-checkpoints/speculator_output" in script
    assert "/tmp/checkpoints" not in script

    print("test execution complete")


def test_train_only_requires_output_dir():
    """Test that TRAIN_ONLY mode requires output_dir."""
    print("Executing test: TRAIN_ONLY mode requires output_dir")

    with pytest.raises(ValueError, match="output_dir is required for TRAIN_ONLY mode"):
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
        )

    print("test execution complete")


def test_training_script_custom_total_seq_len():
    """Test that generated script uses custom total_seq_len value."""
    print("Executing test: Training script custom total_seq_len")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        total_seq_len=2048,
    )

    script = _render_speculator_training_script(trainer)

    assert "total_seq_len=2048" in script
    assert "total_seq_len=8192" not in script

    print("test execution complete")


def test_training_script_custom_hidden_states_dtype():
    """Test that generated script uses custom hidden_states_dtype value."""
    print("Executing test: Training script custom hidden_states_dtype")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        hidden_states_dtype="float16",
    )

    script = _render_speculator_training_script(trainer)

    assert "hidden_states_dtype='float16'" in script
    assert "hidden_states_dtype='bfloat16'" not in script

    print("test execution complete")


def test_hidden_states_dtype_validation():
    """Test that invalid hidden_states_dtype raises ValueError."""
    print("Executing test: hidden_states_dtype validation")

    with pytest.raises(ValueError, match="hidden_states_dtype must be one of"):
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="pvc://test-pvc/output",
            hidden_states_dtype="float64",
        )

    print("test execution complete")


def test_training_script_distributed_batch_sampler():
    """Test that generated script uses MultipackDistributedBatchSamplerV2 for data sharding."""
    print("Executing test: Training script distributed batch sampler")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
    )

    script = _render_speculator_training_script(trainer)

    assert "MultipackDistributedBatchSamplerV2" in script
    assert "batch_max_length=max_len" in script
    assert "approx_lengths" in script
    assert "num_replicas=world_size" in script
    assert "rank=rank" in script
    assert "batch_sampler=train_batch_sampler" in script
    assert "batch_sampler=val_batch_sampler" in script

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="epochs must be positive integer",
            expected_status=FAILED,
            config={"epochs": 0},
            expected_error=ValueError,
        ),
        TestCase(
            name="lr must be positive number",
            expected_status=FAILED,
            config={"lr": -1e-4},
            expected_error=ValueError,
        ),
        TestCase(
            name="total_seq_len must be positive integer",
            expected_status=FAILED,
            config={"total_seq_len": 0},
            expected_error=ValueError,
        ),
        TestCase(
            name="draft_vocab_size must be positive integer",
            expected_status=FAILED,
            config={"draft_vocab_size": -1},
            expected_error=ValueError,
        ),
    ],
)
def test_numeric_field_validation(test_case):
    """Test that numeric fields are validated for type and range."""
    print(f"Executing test: {test_case.name}")

    try:
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="pvc://test-pvc/output",
            **test_case.config,
        )

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


def test_training_script_uses_verifier_vocab_when_draft_unset():
    """Test that generated script uses verifier_config.vocab_size when draft_vocab_size is None."""
    print("Executing test: Training script uses verifier vocab size when unset")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
    )

    script = _render_speculator_training_script(trainer)

    assert "draft_vocab_size = verifier_config.vocab_size" in script

    print("test execution complete")


def test_crd_conversion_train_only():
    """Test CRD conversion for TRAIN_ONLY mode."""
    print("Executing test: CRD conversion for TRAIN_ONLY")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        num_nodes=2,
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.num_nodes == 2
    assert trainer_crd.command is not None

    script = " ".join(trainer_crd.command) if trainer_crd.command else ""
    assert "speculator_train.py" in script
    assert "Qwen/Qwen3-8B" in script

    print("test execution complete")


def test_crd_conversion_with_env_vars():
    """Test that user env vars appear in CRD."""
    print("Executing test: CRD conversion with env vars")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        env={"WANDB_DISABLED": "true", "NCCL_DEBUG": "INFO"},
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.env is not None
    env_names = {e.name for e in trainer_crd.env}
    assert "WANDB_DISABLED" in env_names
    assert "NCCL_DEBUG" in env_names

    wandb_env = next(e for e in trainer_crd.env if e.name == "WANDB_DISABLED")
    assert wandb_env.value == "true"

    print("test execution complete")


def test_crd_conversion_no_env_vars():
    """Test CRD conversion when no env vars are provided."""
    print("Executing test: CRD conversion without env vars")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.env is None

    print("test execution complete")


def test_crd_conversion_with_resources():
    """Test CRD conversion with resources_per_node."""
    print("Executing test: CRD conversion with resources_per_node")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        resources_per_node={"nvidia.com/gpu": 2, "cpu": 8, "memory": "32Gi"},
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.resources_per_node is not None

    print("test execution complete")


def test_crd_uses_torchrun_entrypoint():
    """Test that CRD conversion sets torchrun as the entrypoint."""
    print("Executing test: CRD uses torchrun entrypoint")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
    )

    get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert runtime.trainer.command == constants.TORCH_COMMAND

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="trainer with progression tracking disabled",
            expected_status=SUCCESS,
            config={"enable_progression_tracking": False},
        ),
        TestCase(
            name="trainer with progression tracking enabled (defaults)",
            expected_status=SUCCESS,
            config={"enable_progression_tracking": True},
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
def test_speculator_trainer_configurations(test_case):
    """Test various SpeculativeDecodingTrainer configurations."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="/data/hidden_states",
            output_dir="pvc://test-pvc/output",
            **test_case.config,
        )

        assert test_case.expected_status == SUCCESS

        for key, value in test_case.config.items():
            assert getattr(trainer, key) == value

    except Exception as e:
        if test_case.expected_error:
            assert type(e) is test_case.expected_error
        else:
            raise

    print("test execution complete")


def test_speculator_enum_values():
    """Test enum values for SpeculatorMode and SpeculatorType."""
    print("Executing test: Enum values")

    assert SpeculatorMode.TRAIN_ONLY.value == "train_only"
    assert SpeculatorMode.DATA_ONLY.value == "data_only"
    assert SpeculatorMode.OFFLINE.value == "offline"
    assert SpeculatorMode.ONLINE.value == "online"
    assert SpeculatorType.EAGLE3.value == "eagle3"
    assert SpeculatorType.DFLASH.value == "dflash"
    assert SpeculatorType.MTP.value == "mtp"
    assert SpeculatorType.PEAGLE.value == "peagle"

    print("test execution complete")


def test_progression_tracking_injected_when_enabled():
    """Test that progression tracking code is injected when enabled."""
    print("Executing test: Progression tracking injected when enabled")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        enable_progression_tracking=True,
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    script = " ".join(trainer_crd.command) if trainer_crd.command else ""
    assert "Speculator Progression Tracking" in script
    assert "apply_progression_tracking" in script
    assert "MetricsHandler" in script
    assert "SpeculatorMetricsHTTPHandler" in script

    print("test execution complete")


def test_progression_tracking_not_injected_when_disabled():
    """Test that progression tracking code is NOT injected when disabled."""
    print("Executing test: Progression tracking not injected when disabled")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/data/hidden_states",
        output_dir="pvc://test-pvc/output",
        enable_progression_tracking=False,
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    script = " ".join(trainer_crd.command) if trainer_crd.command else ""
    assert "Speculator Progression Tracking" not in script
    assert "apply_progression_tracking" not in script

    print("test execution complete")


def test_progression_instrumentation_returns_callable():
    """Test that _create_speculator_progression_instrumentation returns valid tuple."""
    print("Executing test: Progression instrumentation returns callable")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        apply_fn, handler_class, _ = _create_speculator_progression_instrumentation(
            metrics_port=28080,
            num_epochs=3,
            save_path=tmpdir,
        )

        assert callable(apply_fn)
        assert handler_class is not None

    print("test execution complete")


def test_progression_instrumentation_schema_transform():
    """Test that the HTTP handler transforms speculators metrics to controller schema."""
    print("Executing test: Progression instrumentation schema transform")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        _, handler_class, _ = _create_speculator_progression_instrumentation(
            metrics_port=28080,
            num_epochs=3,
            save_path=tmpdir,
        )

        handler = handler_class.__new__(handler_class)
        result = handler._transform(
            {"train": {"loss": 2.5}, "epoch": 0, "lr": 1e-4, "global_step": 5}
        )

        assert result["currentEpoch"] == 1
        assert result["totalEpochs"] == 3
        assert result["currentStep"] == 5
        assert result["trainMetrics"]["loss"] == "2.5000"
        assert result["trainMetrics"]["learning_rate"] == "0.000100"

    print("test execution complete")


def test_hidden_states_path_normalization():
    """Test that hidden_states_path is normalized."""
    print("Executing test: hidden_states_path normalization")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="pvc://shared//hidden_states/",
        output_dir="pvc://shared/output",
    )

    assert trainer.hidden_states_path == "pvc://shared/hidden_states"

    print("test execution complete")


def test_hidden_states_path_unsupported_scheme():
    """Test that non-PVC URI schemes for hidden_states_path raise NotImplementedError."""
    print("Executing test: hidden_states_path unsupported scheme")

    with pytest.raises(NotImplementedError, match="hidden_states_path scheme"):
        SpeculativeDecodingTrainer(
            verifier_name_or_path="Qwen/Qwen3-8B",
            hidden_states_path="s3://bucket/hidden_states",
            output_dir="pvc://shared/output",
        )

    print("test execution complete")


def test_hidden_states_path_direct_path_allowed():
    """Test that direct filesystem paths for hidden_states_path are allowed."""
    print("Executing test: hidden_states_path direct path allowed")

    trainer = SpeculativeDecodingTrainer(
        verifier_name_or_path="Qwen/Qwen3-8B",
        hidden_states_path="/mnt/data/hidden_states",
        output_dir="pvc://shared/output",
    )

    assert trainer.hidden_states_path == "/mnt/data/hidden_states"

    print("test execution complete")
