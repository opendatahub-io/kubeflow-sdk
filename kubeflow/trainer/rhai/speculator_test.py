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
    SpeculatorConfig,
    SpeculatorMode,
    SpeculatorType,
    _create_speculator_progression_instrumentation,
    _render_speculator_training_script,
    apply_speculator_sidecar_overrides,
    get_trainer_cr_from_speculator_trainer,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


def test_speculator_trainer_initialization():
    """Test SpeculativeDecodingTrainer initialization with default values."""
    print("Executing test: SpeculativeDecodingTrainer initialization with defaults")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    assert trainer.verifier_model == "Qwen/Qwen3-8B"
    assert trainer.speculator_type == SpeculatorType.EAGLE3
    assert trainer.mode == SpeculatorMode.TRAIN_ONLY
    assert trainer.hidden_states_path == "pvc://test-pvc/hidden_states"
    assert trainer.epochs == 3
    assert trainer.lr == 1e-4
    assert trainer.total_seq_len == 2048
    assert trainer.training_resources == {"nvidia.com/gpu": 1}
    assert trainer.vllm_resources is None
    assert trainer.vllm_gpu_memory_utilization == 0.9
    assert trainer.config.target_layer_ids == [2, 16, 29, 31]
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
        verifier_model="meta-llama/Llama-3.1-70B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 2},
        speculator_type=SpeculatorType.EAGLE3,
        hidden_states_path="pvc://shared/hidden_states",
        data_path="pvc://shared/arrow_dataset",
        output_dir="pvc://shared/checkpoints/eagle3",
        epochs=5,
        lr=5e-5,
        total_seq_len=4096,
        packages_to_install=["speculators"],
        pip_index_urls=["https://custom.pypi.org/simple"],
        env={"WANDB_DISABLED": "true"},
        enable_progression_tracking=True,
        metrics_port=28090,
        metrics_poll_interval_seconds=60,
        config=SpeculatorConfig(
            num_layers=1,
            ttt_steps=5,
            hidden_states_dtype="float16",
            target_layer_ids=[2, 16, 29, 31],
        ),
    )

    assert trainer.verifier_model == "meta-llama/Llama-3.1-70B"
    assert trainer.speculator_type == SpeculatorType.EAGLE3
    assert trainer.hidden_states_path == "pvc://shared/hidden_states"
    assert trainer.output_dir == "pvc://shared/checkpoints/eagle3"
    assert trainer.epochs == 5
    assert trainer.lr == 5e-5
    assert trainer.total_seq_len == 4096
    assert trainer.training_resources == {"nvidia.com/gpu": 2}
    assert trainer.packages_to_install == ["speculators"]
    assert trainer.pip_index_urls == ["https://custom.pypi.org/simple"]
    assert trainer.env == {"WANDB_DISABLED": "true"}
    assert trainer.metrics_port == 28090
    assert trainer.metrics_poll_interval_seconds == 60
    assert trainer.config.ttt_steps == 5
    assert trainer.config.hidden_states_dtype == "float16"

    print("test execution complete")


def test_speculator_mode_train_only_requires_hidden_states():
    """Test that TRAIN_ONLY mode requires hidden_states_path."""
    print("Executing test: TRAIN_ONLY mode requires hidden_states_path")

    with pytest.raises(ValueError, match="hidden_states_path is required for TRAIN_ONLY"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_speculator_mode_train_only_requires_data_path():
    """Test that TRAIN_ONLY mode requires data_path."""
    print("Executing test: TRAIN_ONLY mode requires data_path")

    with pytest.raises(ValueError, match="data_path is required for TRAIN_ONLY mode"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            output_dir="pvc://test-pvc/output",
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

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
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="pvc://test-pvc/output",
            metrics_port=test_case.config["metrics_port"],
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
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
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="pvc://test-pvc/output",
            metrics_poll_interval_seconds=test_case.config["metrics_poll_interval_seconds"],
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

        assert test_case.expected_status == SUCCESS
        assert trainer.metrics_poll_interval_seconds == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


def test_non_pvc_output_dir_not_supported():
    """Test that non-PVC output_dir raises ValueError."""
    print("Executing test: non-PVC output_dir not supported")

    with pytest.raises(ValueError, match="output_dir must use a PVC URI"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="s3://my-bucket/checkpoints",
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    with pytest.raises(ValueError, match="output_dir must use a PVC URI"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="gcs://my-bucket/checkpoints",
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_pvc_output_dir_normalized():
    """Test that PVC output_dir is normalized."""
    print("Executing test: PVC output_dir normalization")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://my-pvc/hidden_states",
        data_path="pvc://my-pvc/arrow_dataset",
        output_dir="pvc://my-pvc/checkpoints/",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    assert trainer.output_dir == "pvc://my-pvc/checkpoints"

    print("test execution complete")


def test_training_script_content():
    """Test that generated training script contains expected CLI arguments."""
    print("Executing test: Training script content")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        epochs=5,
        lr=1e-5,
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "_speculator_train_only" in script
    assert "verifier_model='Qwen/Qwen3-8B'" in script
    assert "hidden_states_path='/mnt/kubeflow-checkpoints/hidden_states'" in script
    assert "save_path='/mnt/kubeflow-checkpoints/output'" in script
    assert "epochs=5" in script
    assert "lr=1e-05" in script
    assert "total_seq_len=2048" in script
    assert "hidden_states_dtype='bfloat16'" in script

    print("test execution complete")


def test_training_script_trainer_config_fields():
    """Test that generated script passes all TrainerConfig fields from SpeculatorConfig."""
    print("Executing test: Training script TrainerConfig fields")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(
            target_layer_ids=[2, 16, 29, 31],
            scheduler_type="cosine",
            scheduler_warmup_steps=100,
            scheduler_total_steps=5000,
            scheduler_num_cosine_cycles=1.0,
            checkpoint_freq=0.5,
            save_best=True,
            log_freq=10,
            resume_from_checkpoint=True,
        ),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "scheduler_type='cosine'" in script
    assert "scheduler_warmup_steps=100" in script
    assert "scheduler_total_steps=5000" in script
    assert "scheduler_num_cosine_cycles=1.0" in script
    assert "checkpoint_freq=0.5" in script
    assert "save_best=True" in script
    assert "log_freq=10" in script
    assert "resume_from_checkpoint=True" in script

    print("test execution complete")


def test_training_script_trainer_config_defaults():
    """Test that default SpeculatorConfig passes correct defaults for TrainerConfig fields."""
    print("Executing test: Training script TrainerConfig defaults")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "scheduler_type='linear'" in script
    assert "scheduler_warmup_steps=None" in script
    assert "scheduler_total_steps=None" in script
    assert "scheduler_num_cosine_cycles=0.5" in script
    assert "checkpoint_freq=1.0" in script
    assert "save_best=False" in script
    assert "log_freq=1" in script
    assert "resume_from_checkpoint=False" in script

    print("test execution complete")


def test_training_script_with_pvc_output_dir():
    """Test that output_dir PVC URI resolves to correct save_path in training script."""
    print("Executing test: Training script with PVC output_dir")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://shared/hidden_states",
        data_path="pvc://shared/arrow_dataset",
        output_dir="pvc://shared/speculator_output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "/mnt/kubeflow-checkpoints/speculator_output" in script

    print("test execution complete")


def test_train_only_requires_output_dir():
    """Test that TRAIN_ONLY mode requires output_dir."""
    print("Executing test: TRAIN_ONLY mode requires output_dir")

    with pytest.raises(ValueError, match="output_dir is required for TRAIN_ONLY mode"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_training_script_custom_total_seq_len():
    """Test that generated script uses custom total_seq_len value."""
    print("Executing test: Training script custom total_seq_len")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        total_seq_len=2048,
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "total_seq_len=2048" in script
    assert "total_seq_len=8192" not in script

    print("test execution complete")


def test_training_script_custom_hidden_states_dtype():
    """Test that generated script uses custom hidden_states_dtype value."""
    print("Executing test: Training script custom hidden_states_dtype")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(hidden_states_dtype="float16", target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "hidden_states_dtype='float16'" in script
    assert "hidden_states_dtype='bfloat16'" not in script

    print("test execution complete")


def test_hidden_states_dtype_validation():
    """Test that invalid hidden_states_dtype raises ValueError."""
    print("Executing test: hidden_states_dtype validation")

    with pytest.raises(ValueError, match="config.hidden_states_dtype must be one of"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="pvc://test-pvc/output",
            config=SpeculatorConfig(
                hidden_states_dtype="float64", target_layer_ids=[2, 16, 29, 31]
            ),
        )

    print("test execution complete")


def test_training_script_distributed_batch_sampler():
    """Test that in-process training function is used for distributed training."""
    print("Executing test: Training script distributed batch sampler")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "_speculator_train_only" in script
    assert "MultipackDistributedBatchSamplerV2" in script
    assert "init_process_group" in script

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
    ],
)
def test_numeric_field_validation(test_case):
    """Test that numeric fields are validated for type and range."""
    print(f"Executing test: {test_case.name}")

    try:
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="pvc://test-pvc/output",
            **test_case.config,
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

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
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

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
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        env={"WANDB_DISABLED": "true", "NCCL_DEBUG": "INFO"},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
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
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.env is None

    print("test execution complete")


def test_crd_conversion_with_training_resources():
    """Test CRD conversion with training_resources."""
    print("Executing test: CRD conversion with training_resources")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 2},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.resources_per_node is not None

    print("test execution complete")


def test_crd_uses_torchrun_entrypoint_for_train_only():
    """Test that CRD conversion sets torchrun as the entrypoint for TRAIN_ONLY."""
    print("Executing test: CRD uses torchrun entrypoint for TRAIN_ONLY")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/model-opt-cuda-rhel9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert runtime.trainer.command == constants.TORCH_COMMAND

    print("test execution complete")


def test_crd_uses_python_entrypoint_for_data_only():
    """Test that CRD conversion sets plain python as the entrypoint for DATA_ONLY."""
    print("Executing test: CRD uses python entrypoint for DATA_ONLY")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/cuda-ubi9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_model="meta-llama/Llama-3.1-8B-Instruct",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert runtime.trainer.command == constants.DEFAULT_COMMAND

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
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="pvc://test-pvc/output",
            **test_case.config,
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
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
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        enable_progression_tracking=True,
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
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
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        enable_progression_tracking=False,
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    script = " ".join(trainer_crd.command) if trainer_crd.command else ""
    assert "Speculator Progression Tracking" not in script
    assert "apply_progression_tracking" not in script

    print("test execution complete")


def test_progression_instrumentation_returns_callable():
    """Test that _create_speculator_progression_instrumentation returns valid tuple."""
    print("Executing test: Progression instrumentation returns callable")

    apply_fn, start_data_fn, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="train_only",
        num_epochs=3,
    )

    assert callable(apply_fn)
    assert callable(start_data_fn)
    assert handler_class is not None

    print("test execution complete")


def test_progression_instrumentation_schema_transform():
    """Test that the HTTP handler transforms speculators metrics to controller schema."""
    print("Executing test: Progression instrumentation schema transform")

    _, _, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="train_only",
        num_epochs=3,
    )

    handler = handler_class.__new__(handler_class)
    result = handler._training_progress(scale=100, offset=0)

    assert result["trainMetrics"] is None

    print("test execution complete")


def test_hidden_states_path_normalization():
    """Test that hidden_states_path is normalized."""
    print("Executing test: hidden_states_path normalization")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://shared//hidden_states/",
        data_path="pvc://shared/arrow_dataset",
        output_dir="pvc://shared/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    assert trainer.hidden_states_path == "pvc://shared/hidden_states"

    print("test execution complete")


def test_hidden_states_path_unsupported_scheme():
    """Test that non-PVC URI schemes for hidden_states_path raise NotImplementedError."""
    print("Executing test: hidden_states_path unsupported scheme")

    with pytest.raises(ValueError, match="hidden_states_path must use a PVC URI"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="s3://bucket/hidden_states",
            data_path="pvc://shared/arrow_dataset",
            output_dir="pvc://shared/output",
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_hidden_states_path_bare_path_rejected():
    """Test that bare paths for hidden_states_path are rejected."""
    print("Executing test: hidden_states_path bare path rejected")

    with pytest.raises(ValueError, match="hidden_states_path must use a PVC URI"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1},
            hidden_states_path="/mnt/data/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="pvc://test-pvc/output",
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_data_only_requires_dataset_name():
    """Test that DATA_ONLY mode requires dataset_name."""
    print("Executing test: DATA_ONLY mode requires dataset_name")

    with pytest.raises(ValueError, match="dataset_name is required for DATA_ONLY mode"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.DATA_ONLY,
            vllm_resources={"nvidia.com/gpu": 1},
            training_resources={"nvidia.com/gpu": 1},
            output_dir="pvc://test-pvc/output",
        )

    print("test execution complete")


def test_data_only_requires_output_dir():
    """Test that DATA_ONLY mode requires output_dir."""
    print("Executing test: DATA_ONLY mode requires output_dir")

    with pytest.raises(ValueError, match="output_dir is required for DATA_ONLY mode"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.DATA_ONLY,
            vllm_resources={"nvidia.com/gpu": 1},
            training_resources={"nvidia.com/gpu": 1},
            dataset_name="ultrachat",
        )

    print("test execution complete")


def test_data_only_valid():
    """Test that DATA_ONLY mode with required fields succeeds."""
    print("Executing test: DATA_ONLY mode valid configuration")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    assert trainer.mode == SpeculatorMode.DATA_ONLY
    assert trainer.dataset_name == "ultrachat"
    assert trainer.output_dir == "pvc://test-pvc/output"
    assert trainer.hidden_states_path is None

    print("test execution complete")


def test_data_only_renders_correct_script():
    """Test that DATA_ONLY generates script with _speculator_data_only call."""
    print("Executing test: DATA_ONLY renders correct script")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="meta-llama/Llama-3.1-8B-Instruct",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://shared/datagen_output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "_speculator_data_only(" in script
    assert "_speculator_train_only(" not in script
    assert "verifier_model='meta-llama/Llama-3.1-8B-Instruct'" in script
    assert "dataset_name='ultrachat'" in script
    assert "/mnt/kubeflow-checkpoints/datagen_output" in script
    assert "EXTRACTION_INCOMPLETE_MARKER" in script

    print("test execution complete")


def test_data_only_script_passes_world_size_and_rank():
    """Test that datagen command passes --world-size and --rank from env vars."""
    print("Executing test: DATA_ONLY script passes world-size and rank")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert '"--world-size"' in script
    assert '"--rank"' in script

    print("test execution complete")


def test_data_only_script_embeds_datagen_script():
    """Test that DATA_ONLY generated script embeds data_generation_offline.py as base64."""
    print("Executing test: DATA_ONLY script embeds datagen script")

    import base64

    trainer = SpeculativeDecodingTrainer(
        verifier_model="meta-llama/Llama-3.1-8B-Instruct",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://shared/datagen_output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "_DATAGEN_SCRIPT_B64" in script
    assert "base64.b64decode(_DATAGEN_SCRIPT_B64)" in script
    assert "/tmp/data_generation_offline.py" in script

    b64_line = [line for line in script.split("\n") if line.startswith("_DATAGEN_SCRIPT_B64")][0]
    b64_value = b64_line.split('"')[1]
    decoded = base64.b64decode(b64_value).decode("utf-8")
    assert "generate_hidden_states_async" in decoded
    assert "def main():" in decoded

    print("test execution complete")


def test_data_progression_returns_callable():
    """Test that unified instrumentation in data_only mode returns valid tuple."""
    print("Executing test: Data progression instrumentation returns callable")

    apply_fn, start_fn, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="data_only",
    )

    assert callable(apply_fn)
    assert callable(start_fn)
    assert handler_class is not None

    print("test execution complete")


def test_data_progression_handler_counts_files(tmp_path):
    """Test that handler counts hs_*.safetensors files in data_only mode."""
    print("Executing test: Data progression handler counts files")

    _, start_fn, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="data_only",
    )

    hs_dir = tmp_path / "hidden_states"
    hs_dir.mkdir()
    (hs_dir / "hs_0.safetensors").write_text("")
    (hs_dir / "hs_1.safetensors").write_text("")
    (hs_dir / "hs_2.safetensors").write_text("")
    (hs_dir / "other_file.txt").write_text("")

    start_fn(str(hs_dir), 10)

    handler = handler_class.__new__(handler_class)
    result = handler._data_progress(scale=100, offset=0)

    assert result["currentStep"] == 3
    assert result["totalSteps"] == 10
    assert result["progressPercentage"] == 30
    assert result["currentEpoch"] is None
    assert result["totalEpochs"] is None
    assert result["trainMetrics"] is None

    print("test execution complete")


def test_data_progression_handler_empty_dir(tmp_path):
    """Test that handler returns 0% when no hidden states files exist."""
    print("Executing test: Data progression handler empty directory")

    _, start_fn, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="data_only",
    )

    hs_dir = tmp_path / "hidden_states"
    hs_dir.mkdir()

    start_fn(str(hs_dir), 50)

    handler = handler_class.__new__(handler_class)
    result = handler._data_progress(scale=100, offset=0)

    assert result["currentStep"] == 0
    assert result["totalSteps"] == 50
    assert result["progressPercentage"] == 0
    assert result["estimatedRemainingSeconds"] is None

    print("test execution complete")


def test_data_progression_handler_complete(tmp_path):
    """Test that handler returns 100% when all files are generated."""
    print("Executing test: Data progression handler complete")

    _, start_fn, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="data_only",
    )

    hs_dir = tmp_path / "hidden_states"
    hs_dir.mkdir()
    for i in range(5):
        (hs_dir / f"hs_{i}.safetensors").write_text("")

    start_fn(str(hs_dir), 5)

    handler = handler_class.__new__(handler_class)
    result = handler._data_progress(scale=100, offset=0)

    assert result["currentStep"] == 5
    assert result["totalSteps"] == 5
    assert result["progressPercentage"] == 100

    print("test execution complete")


def test_data_progression_handler_not_started():
    """Test that handler returns nulls when data tracking not yet started."""
    print("Executing test: Data progression handler not started")

    _, _, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="data_only",
    )

    handler = handler_class.__new__(handler_class)
    result = handler._data_progress(scale=100, offset=0)

    assert result["progressPercentage"] is None
    assert result["currentStep"] is None
    assert result["totalSteps"] is None

    print("test execution complete")


def test_data_only_script_contains_progress_server_call():
    """Test that DATA_ONLY rendered script contains _start_data_progress_server call."""
    print("Executing test: DATA_ONLY script contains progress server call")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    assert "_start_data_progress_server" in script

    print("test execution complete")


def test_data_only_progression_tracking_injected():
    """Test that DATA_ONLY mode injects unified progression instrumentation."""
    print("Executing test: DATA_ONLY progression tracking injected")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        enable_progression_tracking=True,
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="test-image:latest",
        ),
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)
    bash_script = trainer_crd.command[2]

    assert "Speculator Progression Tracking" in bash_script
    assert "_create_speculator_progression_instrumentation" in bash_script
    assert "mode='data_only'" in bash_script

    print("test execution complete")


def test_data_only_script_contains_marker_logic():
    """Test that DATA_ONLY script contains per-rank incomplete marker check and cleanup."""
    print("Executing test: DATA_ONLY script contains marker logic")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "EXTRACTION_INCOMPLETE_MARKER" in script
    assert "Previous data extraction" in script
    assert "os.remove(incomplete_marker)" in script

    print("test execution complete")


def test_train_only_script_contains_training_function():
    """Test that TRAIN_ONLY script includes training function."""
    print("Executing test: TRAIN_ONLY script contains training function")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "EXTRACTION_INCOMPLETE_MARKER" in script
    assert "_speculator_train_only" in script

    print("test execution complete")


def test_data_only_script_contains_vllm_health_check():
    """Test that DATA_ONLY script contains vLLM health check for external endpoint."""
    print("Executing test: DATA_ONLY script contains vLLM health check")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "/health" in script
    assert "vLLM sidecar is ready" in script
    assert "vLLM endpoint not reachable" in script

    print("test execution complete")


def test_data_only_script_skips_completed_extraction():
    """Test that DATA_ONLY script skips extraction when hidden states already exist."""
    print("Executing test: DATA_ONLY script skips completed extraction")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert ".safetensors" in script
    assert "Data extraction already completed. Skipping." in script

    print("test execution complete")


def test_apply_speculator_sidecar_overrides():
    """Test sidecar overrides add correct init container config."""
    print("Executing test: apply_speculator_sidecar_overrides")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="pvc://shared/models/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://shared/speculator/run1",
        vllm_resources={"nvidia.com/gpu": 1},
        vllm_gpu_memory_utilization=0.85,
        config=SpeculatorConfig(target_layer_ids=[2, 18, 33, 35]),
    )

    result = apply_speculator_sidecar_overrides(trainer, [])

    assert len(result) == 1
    override = result[0]
    assert override["targetJobs"] == [{"name": "node"}]

    init_containers = override["spec"]["initContainers"]
    assert len(init_containers) == 1

    sidecar = init_containers[0]
    assert sidecar["name"] == "vllm-sidecar"

    env_dict = {e["name"]: e["value"] for e in sidecar["env"]}
    assert env_dict["SPECULATOR_VERIFIER_MODEL"] == "/mnt/kubeflow-checkpoints/models/Qwen3-8B"
    assert env_dict["SPECULATOR_HS_PATH"] == (
        "/mnt/kubeflow-checkpoints/speculator/run1/hidden_states"
    )
    assert env_dict["SPECULATOR_GPU_MEM_UTIL"] == "0.85"
    assert env_dict["SPECULATOR_VLLM_GPU_COUNT"] == "1"
    assert env_dict["SPECULATOR_TARGET_LAYER_IDS"] == "2,18,33,35"

    assert sidecar["volumeMounts"][0]["name"] == "checkpoint-storage"
    assert sidecar["volumeMounts"][0]["mountPath"] == "/mnt/kubeflow-checkpoints"

    assert sidecar["resources"]["limits"]["nvidia.com/gpu"] == "1"

    print("test execution complete")


def test_apply_speculator_sidecar_overrides_preserves_existing():
    """Test sidecar overrides preserve existing pod template overrides."""
    print("Executing test: sidecar overrides preserve existing overrides")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://shared/output",
        config=SpeculatorConfig(target_layer_ids=[2, 18, 33, 35]),
    )

    existing = [
        {
            "targetJobs": [{"name": "node"}],
            "spec": {
                "volumes": [
                    {"name": "checkpoint-storage", "persistentVolumeClaim": {"claimName": "shared"}}
                ],
                "containers": [
                    {
                        "name": "node",
                        "volumeMounts": [
                            {"name": "checkpoint-storage", "mountPath": "/mnt/kubeflow-checkpoints"}
                        ],
                    }
                ],
            },
        }
    ]

    result = apply_speculator_sidecar_overrides(trainer, existing)

    assert len(result) == 1
    spec = result[0]["spec"]
    assert len(spec["volumes"]) == 1
    assert len(spec["containers"]) == 1
    assert len(spec["initContainers"]) == 1
    assert spec["initContainers"][0]["name"] == "vllm-sidecar"

    print("test execution complete")


def test_data_only_script_uses_sidecar_endpoint():
    """Test DATA_ONLY script renders sidecar endpoint."""
    print("Executing test: DATA_ONLY script uses sidecar endpoint")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "http://localhost:8234/v1" in script
    assert "gpu_memory_utilization" not in script
    assert "vllm_gpu_count" not in script

    print("test execution complete")


def test_data_only_script_resolves_pvc_verifier_model():
    """Test that pvc:// verifier_model is resolved to local path in rendered script."""
    print("Executing test: DATA_ONLY script resolves pvc:// verifier_model")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="pvc://shared/models/meta-llama/Llama-3.1-8B-Instruct",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://shared/datagen_output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert (
        "verifier_model='/mnt/kubeflow-checkpoints/models/meta-llama/Llama-3.1-8B-Instruct'"
        in script
    )
    assert "pvc://" not in script

    print("test execution complete")


def test_verifier_model_hf_id_auto_detects_target_layer_ids():
    """Test that HF model ID triggers AutoConfig to compute target_layer_ids."""
    print("Executing test: HF verifier_model auto-detects target_layer_ids")

    import sys
    from unittest.mock import MagicMock

    mock_model_config = MagicMock()
    mock_model_config.num_hidden_layers = 36
    mock_model_config.text_config = mock_model_config
    del mock_model_config.text_config

    mock_auto_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_model_config

    mock_transformers = MagicMock()
    mock_transformers.AutoConfig = mock_auto_config

    sys.modules["transformers"] = mock_transformers
    try:
        trainer = SpeculativeDecodingTrainer(
            verifier_model="meta-llama/Llama-3.1-8B-Instruct",
            mode=SpeculatorMode.DATA_ONLY,
            vllm_resources={"nvidia.com/gpu": 1},
            training_resources={"nvidia.com/gpu": 1},
            dataset_name="ultrachat",
            output_dir="pvc://shared/datagen_output",
        )
    finally:
        del sys.modules["transformers"]

    assert trainer.config is not None
    assert trainer.config.target_layer_ids == [2, 18, 33, 36]

    script = _render_speculator_training_script(trainer)
    assert "verifier_model='meta-llama/Llama-3.1-8B-Instruct'" in script

    print("test execution complete")


def test_verifier_model_invalid_id_fails_autoconfig():
    """Test that invalid verifier_model raises ValueError with clear message."""
    print("Executing test: invalid verifier_model fails AutoConfig")

    import sys
    from unittest.mock import MagicMock

    mock_auto_config = MagicMock()
    mock_auto_config.from_pretrained.side_effect = OSError(
        "not-a-real-model is not a local folder and is not a valid model identifier"
    )

    mock_transformers = MagicMock()
    mock_transformers.AutoConfig = mock_auto_config

    sys.modules["transformers"] = mock_transformers
    try:
        with pytest.raises(ValueError, match="is not a valid HuggingFace model ID"):
            SpeculativeDecodingTrainer(
                verifier_model="not-a-real-model",
                mode=SpeculatorMode.DATA_ONLY,
                vllm_resources={"nvidia.com/gpu": 1},
                training_resources={"nvidia.com/gpu": 1},
                dataset_name="ultrachat",
                output_dir="pvc://shared/datagen_output",
            )
    finally:
        del sys.modules["transformers"]

    print("test execution complete")


def test_sidecar_overrides_resolves_pvc_verifier_model():
    """Test that pvc:// verifier_model is resolved in sidecar env var."""
    print("Executing test: sidecar resolves pvc:// verifier_model")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="pvc://shared/models/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://shared/datagen_output",
        config=SpeculatorConfig(target_layer_ids=[2, 18, 33, 35]),
    )

    result = apply_speculator_sidecar_overrides(trainer, [])

    sidecar = result[0]["spec"]["initContainers"][0]
    env_dict = {e["name"]: e["value"] for e in sidecar["env"]}
    assert env_dict["SPECULATOR_VERIFIER_MODEL"] == "/mnt/kubeflow-checkpoints/models/Qwen3-8B"

    print("test execution complete")


def test_sidecar_overrides_passes_target_layer_ids():
    """Test that target_layer_ids is passed as SPECULATOR_TARGET_LAYER_IDS env var."""
    print("Executing test: sidecar passes target_layer_ids")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="pvc://shared/models/Qwen3-8B",
        mode=SpeculatorMode.DATA_ONLY,
        vllm_resources={"nvidia.com/gpu": 1},
        training_resources={"nvidia.com/gpu": 1},
        dataset_name="ultrachat",
        output_dir="pvc://shared/datagen_output",
        config=SpeculatorConfig(target_layer_ids=[2, 18, 33, 35]),
    )

    result = apply_speculator_sidecar_overrides(trainer, [])

    sidecar = result[0]["spec"]["initContainers"][0]
    env_dict = {e["name"]: e["value"] for e in sidecar["env"]}
    assert env_dict["SPECULATOR_TARGET_LAYER_IDS"] == "2,18,33,35"

    print("test execution complete")


def test_train_only_script_resolves_pvc_data_path():
    """Test that TRAIN_ONLY script resolves pvc:// data_path to local mount."""
    print("Executing test: TRAIN_ONLY resolves pvc:// data_path")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://shared/hidden_states",
        data_path="pvc://shared/arrow_dataset",
        output_dir="pvc://shared/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "data_path='/mnt/kubeflow-checkpoints/arrow_dataset'" in script

    print("test execution complete")


def test_train_only_script_passes_draft_vocab_size():
    """Test that TRAIN_ONLY script passes draft_vocab_size."""
    print("Executing test: TRAIN_ONLY passes draft_vocab_size")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        draft_vocab_size=8192,
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "draft_vocab_size=8192" in script

    print("test execution complete")


def test_train_only_script_draft_vocab_size_none_by_default():
    """Test that draft_vocab_size defaults to None in rendered script."""
    print("Executing test: draft_vocab_size defaults to None")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "draft_vocab_size=None" in script

    print("test execution complete")


def test_train_only_script_contains_vocab_mapping_logic():
    """Test that TRAIN_ONLY script contains vocab mapping imports and logic."""
    print("Executing test: TRAIN_ONLY contains vocab mapping logic")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.TRAIN_ONLY,
        training_resources={"nvidia.com/gpu": 1},
        hidden_states_path="pvc://test-pvc/hidden_states",
        data_path="pvc://test-pvc/arrow_dataset",
        output_dir="pvc://test-pvc/output",
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)
    compile(script, "<test>", "exec")

    assert "build_vocab_mappings_from_distribution" in script
    assert "d2t_path = Path(data_path)" in script
    assert "t2d_path = Path(data_path)" in script
    assert '"d2t": d2t' in script
    assert '"t2d": t2d' in script

    print("test execution complete")


# =============================================================================
# ONLINE mode tests
# =============================================================================


def test_online_mode_valid():
    """Test that ONLINE mode with required fields succeeds."""
    print("Executing test: ONLINE mode valid configuration")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    assert trainer.mode == SpeculatorMode.ONLINE
    assert trainer.dataset_name == "ultrachat"
    assert trainer.output_dir == "pvc://test-pvc/output"

    print("test execution complete")


def test_online_mode_requires_dataset_name():
    """Test that ONLINE mode requires dataset_name."""
    print("Executing test: ONLINE mode requires dataset_name")

    with pytest.raises(ValueError, match="dataset_name is required for ONLINE mode"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.ONLINE,
            output_dir="pvc://test-pvc/output",
            training_resources={"nvidia.com/gpu": 2},
            vllm_resources={"nvidia.com/gpu": 1},
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_online_mode_rejects_vllm_endpoint():
    """Test that ONLINE mode rejects vllm_endpoint (uses sidecar instead)."""
    print("Executing test: ONLINE mode rejects vllm_endpoint")

    with pytest.raises(ValueError, match="vllm_endpoint is only supported in OFFLINE mode"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.ONLINE,
            dataset_name="ultrachat",
            output_dir="pvc://test-pvc/output",
            vllm_endpoint="http://vllm-svc:8000/v1",
            training_resources={"nvidia.com/gpu": 2},
            vllm_resources={"nvidia.com/gpu": 1},
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_online_mode_allows_regenerate_responses():
    """Test that ONLINE mode accepts regenerate_responses."""
    print("Executing test: ONLINE mode allows regenerate_responses")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        regenerate_responses=True,
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    assert trainer.regenerate_responses is True

    print("test execution complete")


def test_online_mode_allows_max_samples():
    """Test that ONLINE mode allows max_samples."""
    print("Executing test: ONLINE mode allows max_samples")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        max_samples=1000,
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    assert trainer.max_samples == 1000

    print("test execution complete")


def test_online_mode_rejects_max_samples_in_train_only():
    """Test that TRAIN_ONLY mode rejects max_samples."""
    print("Executing test: TRAIN_ONLY rejects max_samples")

    with pytest.raises(ValueError, match="max_samples is only supported in"):
        SpeculativeDecodingTrainer(
            verifier_model="Qwen/Qwen3-8B",
            mode=SpeculatorMode.TRAIN_ONLY,
            hidden_states_path="pvc://test-pvc/hidden_states",
            data_path="pvc://test-pvc/arrow_dataset",
            output_dir="pvc://test-pvc/output",
            training_resources={"nvidia.com/gpu": 2},
            max_samples=1000,
            config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
        )

    print("test execution complete")


def test_online_renders_correct_script():
    """Test that ONLINE generates script with _speculator_online call."""
    print("Executing test: ONLINE renders correct script")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://shared/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert "_speculator_online(" in script
    assert "_speculator_data_only(" not in script
    assert "_speculator_train_only(" not in script
    assert "verifier_model='Qwen/Qwen3-8B'" in script
    assert "dataset_name='ultrachat'" in script
    assert "output_dir='/mnt/kubeflow-checkpoints/output'" in script
    assert "vllm_endpoint='http://localhost:8234/v1'" in script

    print("test execution complete")


def test_online_script_uses_on_missing_generate():
    """Test that ONLINE script uses ArrowDataset with on_missing='generate'."""
    print("Executing test: ONLINE script uses on_missing generate")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert 'on_missing="generate"' in script
    assert "DistributedSampler" in script
    assert "data_generation_offline" not in script

    print("test execution complete")


def test_online_script_no_datagen_embedded():
    """Test that ONLINE script does not embed data_generation_offline.py."""
    print("Executing test: ONLINE script has no embedded datagen")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert "_DATAGEN_SCRIPT_B64" not in script
    assert "_REGEN_SCRIPT_B64" in script

    print("test execution complete")


def test_online_script_contains_vllm_health_check():
    """Test that ONLINE script contains vLLM sidecar health check."""
    print("Executing test: ONLINE script contains vLLM health check")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert "health" in script
    assert "vLLM sidecar is ready" in script

    print("test execution complete")


def test_online_script_contains_preprocessing():
    """Test that ONLINE script contains preprocessing step."""
    print("Executing test: ONLINE script contains preprocessing")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert "load_and_preprocess_dataset" in script
    assert "token_freq" in script
    assert "save_to_disk" in script

    print("test execution complete")


def test_online_script_contains_model_setup():
    """Test that ONLINE script contains model setup with vocab mappings."""
    print("Executing test: ONLINE script contains model setup")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert "Eagle3DraftModel" in script
    assert "build_vocab_mappings_from_distribution" in script
    assert "AutoConfig" in script
    assert "Trainer" in script
    assert "TrainerConfig" in script
    assert "run_training" in script

    print("test execution complete")


def test_online_crd_uses_torchrun():
    """Test that ONLINE mode CRD uses TORCH_COMMAND (torchrun)."""
    print("Executing test: ONLINE CRD uses torchrun")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/cuda-ubi9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert runtime.trainer.command == constants.TORCH_COMMAND

    print("test execution complete")


def test_online_crd_has_gpu_resources():
    """Test that ONLINE mode CRD includes GPU resources."""
    print("Executing test: ONLINE CRD has GPU resources")

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="registry.redhat.io/rhaii/cuda-ubi9:3.5",
        ),
    )

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    trainer_crd = get_trainer_cr_from_speculator_trainer(runtime, trainer)

    assert trainer_crd.resources_per_node is not None

    print("test execution complete")


def test_online_sidecar_overrides():
    """Test that ONLINE mode gets sidecar overrides (same as DATA_ONLY)."""
    print("Executing test: ONLINE sidecar overrides")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    result = apply_speculator_sidecar_overrides(trainer, [])

    sidecar = result[0]["spec"]["initContainers"][0]
    env_dict = {e["name"]: e["value"] for e in sidecar["env"]}

    assert env_dict["SPECULATOR_VERIFIER_MODEL"] == "Qwen/Qwen3-8B"
    assert env_dict["SPECULATOR_TARGET_LAYER_IDS"] == "2,16,29,31"
    assert "SPECULATOR_HS_PATH" in env_dict

    print("test execution complete")


def test_online_progression_tracking():
    """Test that ONLINE mode uses training-only progression (scale=100, offset=0)."""
    print("Executing test: ONLINE progression tracking")

    apply_fn, _, handler_class, _, _ = _create_speculator_progression_instrumentation(
        metrics_port=28080,
        mode="online",
        num_epochs=3,
    )

    handler = handler_class.__new__(handler_class)
    progress = handler._get_progress()

    assert progress["progressPercentage"] is not None or progress["progressPercentage"] == 0

    print("test execution complete")


def test_online_script_passes_max_samples():
    """Test that ONLINE script passes max_samples to the function call."""
    print("Executing test: ONLINE script passes max_samples")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        max_samples=500,
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert "max_samples=500" in script

    print("test execution complete")


def test_online_script_passes_target_layer_ids():
    """Test that ONLINE script passes target_layer_ids to the function call."""
    print("Executing test: ONLINE script passes target_layer_ids")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="Qwen/Qwen3-8B",
        mode=SpeculatorMode.ONLINE,
        dataset_name="ultrachat",
        output_dir="pvc://test-pvc/output",
        training_resources={"nvidia.com/gpu": 2},
        vllm_resources={"nvidia.com/gpu": 1},
        config=SpeculatorConfig(target_layer_ids=[2, 16, 29, 31]),
    )

    script = _render_speculator_training_script(trainer)

    assert "target_layer_ids=[2, 16, 29, 31]" in script

    print("test execution complete")


def test_offline_script_with_hidden_states_path():
    """Test that OFFLINE script passes hidden_states_path when set."""
    print("Executing test: OFFLINE script with hidden_states_path")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="pvc://shared/speculator/models/Qwen3-8B",
        mode=SpeculatorMode.OFFLINE,
        dataset_name="ultrachat",
        output_dir="pvc://shared/speculator/output",
        vllm_endpoint="http://vllm-svc:8000/v1",
        hidden_states_path="pvc://shared/offline_output/hidden_states",
        training_resources={"nvidia.com/gpu": 2},
        config=SpeculatorConfig(target_layer_ids=[2, 18, 33, 35]),
    )

    script = _render_speculator_training_script(trainer)

    assert "hidden_states_path='/mnt/kubeflow-checkpoints/offline_output/hidden_states'" in script

    print("test execution complete")


def test_offline_requires_hidden_states_path():
    """Test that OFFLINE mode requires hidden_states_path."""
    print("Executing test: OFFLINE mode requires hidden_states_path")

    with pytest.raises(ValueError, match="hidden_states_path is required for OFFLINE"):
        SpeculativeDecodingTrainer(
            verifier_model="pvc://shared/speculator/models/Qwen3-8B",
            mode=SpeculatorMode.OFFLINE,
            dataset_name="ultrachat",
            output_dir="pvc://shared/speculator/output",
            vllm_endpoint="http://vllm-svc:8000/v1",
            training_resources={"nvidia.com/gpu": 2},
            config=SpeculatorConfig(target_layer_ids=[2, 18, 33, 35]),
        )

    print("test execution complete")


def test_offline_script_skips_model_arg_with_hidden_states_path():
    """Test that OFFLINE script skips --model when hidden_states_path is set."""
    print("Executing test: OFFLINE script skips --model with hidden_states_path")

    trainer = SpeculativeDecodingTrainer(
        verifier_model="pvc://shared/speculator/models/Qwen3-8B",
        mode=SpeculatorMode.OFFLINE,
        dataset_name="ultrachat",
        output_dir="pvc://shared/speculator/output",
        vllm_endpoint="http://vllm-svc:8000/v1",
        hidden_states_path="pvc://shared/offline_output/hidden_states",
        training_resources={"nvidia.com/gpu": 2},
        config=SpeculatorConfig(target_layer_ids=[2, 18, 33, 35]),
    )

    script = _render_speculator_training_script(trainer)

    assert "if not hidden_states_path:" in script
    assert '"--hidden-states-dir"' in script

    print("test execution complete")
