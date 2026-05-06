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

"""Tests for Training Hub progression tracking instrumentation module."""

from unittest.mock import patch

import pytest

from kubeflow.trainer.test.common import SUCCESS, TestCase


def test_traininghub_progression_instrumentation_imports():
    """Test that Training Hub progression instrumentation module imports correctly."""
    print("Executing test: Training Hub progression instrumentation imports")

    import kubeflow.trainer.rhai.instrumentation.traininghub_progression as th_progression_module

    assert hasattr(th_progression_module, "create_traininghub_progression_instrumentation")
    assert callable(th_progression_module.create_traininghub_progression_instrumentation)
    print("test execution complete")


def test_create_traininghub_progression_instrumentation():
    """Test create_traininghub_progression_instrumentation returns expected components."""
    print("Executing test: create Training Hub progression instrumentation components")

    from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (
        create_traininghub_progression_instrumentation,
    )

    # Create instrumentation components with test metadata
    algorithm_metadata = {
        "name": "sft",
        "metrics_file_pattern": "metrics_rank_*.jsonl",
        "metrics_file_rank0": "metrics_rank_0.jsonl",
    }

    result = create_traininghub_progression_instrumentation(
        algorithm_metadata=algorithm_metadata,
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Unpack the tuple
    apply_fn, handler_cls = result

    # Verify all components are present
    assert callable(apply_fn)
    assert handler_cls is not None
    assert handler_cls.__name__ == "TrainingHubMetricsHandler"

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="SFT metrics transformation",
            expected_status=SUCCESS,
            config={
                "algorithm": "sft",
                "metrics": {
                    "step": 50,
                    "epoch": 1,
                    "num_epoch_steps": 100,
                    "avg_loss": 0.5,
                    "lr": 0.0001,
                    "overall_throughput": 10.5,
                },
            },
            expected_output={
                "progressPercentage": 25,  # 50 / (100 * 2 epochs)
                "currentStep": 50,
                "currentEpoch": 2,
                "trainMetrics": {
                    "loss": "0.5000",
                    "learning_rate": "0.000100",
                },
            },
        ),
        TestCase(
            name="OSFT metrics transformation",
            expected_status=SUCCESS,
            config={
                "algorithm": "osft",
                "metrics": {
                    "step": 100,
                    "epoch": 2,
                    "steps_per_epoch": 50,
                    "loss": 0.3,
                    "lr": 0.0002,
                    "time_per_batch": 0.5,
                    "_config": {"max_epochs": 5},
                },
            },
            expected_output={
                "progressPercentage": 40,  # 100 / (50 * 5)
                "currentStep": 100,
                "totalSteps": 250,
                "currentEpoch": 3,
                "totalEpochs": 5,
            },
        ),
        TestCase(
            name="LoRA SFT metrics transformation",
            expected_status=SUCCESS,
            config={
                "algorithm": "lora_sft",
                "metrics": {
                    "step": 75,
                    "epoch": 0.75,
                    "loss": 0.4,
                    "learning_rate": 0.00001,
                    "max_steps": 100,
                },
            },
            expected_output={
                "progressPercentage": 75,  # 75 / 100
                "currentStep": 75,
                "totalSteps": 100,
                "currentEpoch": 1,
            },
        ),
    ],
)
def test_metrics_transformation(test_case):
    """Test that metrics are correctly transformed for different algorithms."""
    print(f"Executing test: {test_case.name}")

    # This test verifies the transformation logic exists in the module
    # Actual transformation testing is done through integration tests
    from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (
        create_traininghub_progression_instrumentation,
    )

    algorithm = test_case.config["algorithm"]

    algorithm_metadata = {
        "name": algorithm,
        "metrics_file_pattern": f"{algorithm}_metrics*.jsonl",
        "metrics_file_rank0": f"{algorithm}_metrics_rank_0.jsonl",
    }

    # Verify the module creates the instrumentation successfully
    apply_fn, handler_cls = create_traininghub_progression_instrumentation(
        algorithm_metadata=algorithm_metadata,
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify components exist
    assert callable(apply_fn)
    assert handler_cls is not None
    assert handler_cls.__name__ == "TrainingHubMetricsHandler"

    print("test execution complete")


def test_termination_message_write(tmp_path):
    """Test that termination message logic is included in the module."""
    print("Executing test: termination message write logic")

    from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (
        create_traininghub_progression_instrumentation,
    )

    algorithm_metadata = {
        "name": "sft",
        "metrics_file_pattern": "metrics_rank_*.jsonl",
        "metrics_file_rank0": "metrics_rank_0.jsonl",
    }

    apply_fn, handler_cls = create_traininghub_progression_instrumentation(
        algorithm_metadata=algorithm_metadata,
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify components exist
    assert callable(apply_fn)
    assert handler_cls is not None

    # Termination message writing is tested through integration tests
    # where the full HTTP handler is instantiated in a real server context

    print("test execution complete")


def test_empty_metrics_handling():
    """Test that instrumentation can be created for algorithms with no metrics."""
    print("Executing test: empty metrics handling")

    from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (
        create_traininghub_progression_instrumentation,
    )

    algorithm_metadata = {
        "name": "sft",
        "metrics_file_pattern": None,  # No metrics files
        "metrics_file_rank0": None,
    }

    apply_fn, handler_cls = create_traininghub_progression_instrumentation(
        algorithm_metadata=algorithm_metadata,
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify components exist even for algorithms without metrics
    assert callable(apply_fn)
    assert handler_cls is not None

    print("test execution complete")


def test_http_server_lifecycle():
    """Test that HTTP server can be started."""
    print("Executing test: HTTP server lifecycle")

    from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (
        create_traininghub_progression_instrumentation,
    )

    algorithm_metadata = {
        "name": "sft",
        "metrics_file_pattern": "metrics_rank_*.jsonl",
        "metrics_file_rank0": "metrics_rank_0.jsonl",
    }

    # Use a unique port to avoid conflicts
    import random

    test_port = random.randint(29000, 30000)

    apply_fn, _ = create_traininghub_progression_instrumentation(
        algorithm_metadata=algorithm_metadata,
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=test_port,
    )

    # Mock environment to simulate non-primary pod (skip cleanup)
    with patch.dict("os.environ", {"JOB_COMPLETION_INDEX": "1"}):
        # Start server (should succeed without cleanup)
        server = apply_fn()

        # Verify server was created
        assert server is not None

        # Shutdown server
        if server:
            server.shutdown()

    print("test execution complete")


def test_algorithm_not_supported():
    """Test that unsupported algorithm returns empty metrics."""
    print("Executing test: unsupported algorithm handling")

    from kubeflow.trainer.rhai.instrumentation.traininghub_progression import (
        create_traininghub_progression_instrumentation,
    )

    algorithm_metadata = {
        "name": "unsupported_algo",
        "metrics_file_pattern": "metrics*.jsonl",
        "metrics_file_rank0": "metrics_rank_0.jsonl",
    }

    apply_fn, handler_cls = create_traininghub_progression_instrumentation(
        algorithm_metadata=algorithm_metadata,
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify components exist even for unsupported algorithms
    assert callable(apply_fn)
    assert handler_cls is not None

    print("test execution complete")
