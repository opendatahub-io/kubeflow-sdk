# Copyright The Kubeflow Authors.
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

"""Unit tests for LocalProcess backend types."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.backends.localprocess.types import (
    LocalBackendJobs,
    LocalBackendStep,
    LocalProcessBackendConfig,
    LocalRuntimeTrainer,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase
from kubeflow.trainer.types import types


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default config has cleanup_venv True",
            expected_status=SUCCESS,
            config={},
            expected_output={"cleanup_venv": True},
        ),
        TestCase(
            name="custom config with cleanup_venv False",
            expected_status=SUCCESS,
            config={"cleanup_venv": False},
            expected_output={"cleanup_venv": False},
        ),
    ],
)
def test_local_process_backend_config(test_case):
    """Test LocalProcessBackendConfig default and custom values."""
    print("Executing test:", test_case.name)

    cfg = LocalProcessBackendConfig(**test_case.config)
    assert cfg.cleanup_venv == test_case.expected_output["cleanup_venv"]

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default runtime trainer has empty packages",
            expected_status=SUCCESS,
            config={
                "trainer_type": types.TrainerType.CUSTOM_TRAINER,
                "framework": "torch",
                "image": "pytorch/pytorch:latest",
            },
            expected_output=[],
        ),
        TestCase(
            name="runtime trainer with packages",
            expected_status=SUCCESS,
            config={
                "trainer_type": types.TrainerType.CUSTOM_TRAINER,
                "framework": "torch",
                "image": "pytorch/pytorch:latest",
                "packages": ["numpy", "scipy"],
            },
            expected_output=["numpy", "scipy"],
        ),
    ],
)
def test_local_runtime_trainer(test_case):
    """Test LocalRuntimeTrainer default and custom packages."""
    print("Executing test:", test_case.name)

    config = test_case.config
    trainer = LocalRuntimeTrainer(
        trainer_type=config["trainer_type"],
        framework=config["framework"],
        image=config["image"],
        packages=config.get("packages", []),
    )
    assert trainer.packages == test_case.expected_output

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="backend step with mock job",
            expected_status=SUCCESS,
            config={"step_name": "training"},
            expected_output="training",
        ),
    ],
)
def test_local_backend_step(test_case):
    """Test LocalBackendStep creation with a mock LocalJob."""
    print("Executing test:", test_case.name)

    mock_job = Mock(spec=LocalJob)
    step = LocalBackendStep.model_construct(
        step_name=test_case.config["step_name"],
        job=mock_job,
    )
    assert step.step_name == test_case.expected_output
    assert step.job is mock_job

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="backend jobs with required fields only",
            expected_status=SUCCESS,
            config={"name": "train-run-1"},
            expected_output={
                "name": "train-run-1",
                "steps": [],
                "runtime": None,
                "created": None,
                "completed": None,
            },
        ),
        TestCase(
            name="backend jobs with timestamps",
            expected_status=SUCCESS,
            config={
                "name": "train-run-2",
                "created": "2025-01-01T00:00:00",
                "completed": "2025-01-01T01:00:00",
            },
            expected_output={
                "name": "train-run-2",
                "created": datetime(2025, 1, 1, 0, 0, 0),
                "completed": datetime(2025, 1, 1, 1, 0, 0),
            },
        ),
    ],
)
def test_local_backend_jobs(test_case):
    """Test LocalBackendJobs creation with various configurations."""
    print("Executing test:", test_case.name)

    config = test_case.config
    jobs = LocalBackendJobs(
        name=config["name"],
        created=config.get("created"),
        completed=config.get("completed"),
    )
    expected = test_case.expected_output
    assert jobs.name == expected["name"]

    if "steps" in expected:
        assert jobs.steps == expected["steps"]
    if "runtime" in expected:
        assert jobs.runtime == expected["runtime"]
    if "created" in expected:
        assert jobs.created == expected["created"]
    if "completed" in expected:
        assert jobs.completed == expected["completed"]

    print("test execution complete")
