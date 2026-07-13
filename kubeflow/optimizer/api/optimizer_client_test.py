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

"""Unit tests for kubeflow.optimizer.api.optimizer_client module."""

from unittest.mock import Mock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.optimizer.api.optimizer_client import OptimizerClient
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
)
from kubeflow.optimizer.types.search_types import Search
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types.types import (
    CustomTrainer,
    TrainJobTemplate,
)


@pytest.fixture
def mock_backend():
    """Provide an OptimizerClient whose backend is fully mocked."""
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch("kubernetes.client.CustomObjectsApi", return_value=Mock()),
        patch("kubernetes.client.CoreV1Api", return_value=Mock()),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.verify_backend",
            return_value=None,
        ),
    ):
        client = OptimizerClient()
        client.backend = Mock()
        yield client


# --- __init__ tests ---


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default config creates KubernetesBackend",
            expected_status=SUCCESS,
        ),
        TestCase(
            name="None config creates KubernetesBackend",
            expected_status=SUCCESS,
            config={"backend_config": None},
        ),
        TestCase(
            name="explicit KubernetesBackendConfig works",
            expected_status=SUCCESS,
            config={"backend_config": KubernetesBackendConfig(namespace="custom")},
        ),
        TestCase(
            name="invalid backend config raises ValueError",
            expected_status=FAILED,
            config={"backend_config": "not-a-config"},
            expected_error=ValueError,
        ),
    ],
)
def test_init(test_case: TestCase):
    """Test OptimizerClient initialization with various backend configs."""
    print("Executing test:", test_case.name)
    try:
        with (
            patch("kubernetes.config.load_kube_config", return_value=None),
            patch("kubernetes.client.CustomObjectsApi", return_value=Mock()),
            patch("kubernetes.client.CoreV1Api", return_value=Mock()),
            patch(
                "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.verify_backend",
                return_value=None,
            ),
        ):
            client = OptimizerClient(**test_case.config) if test_case.config else OptimizerClient()
            assert test_case.expected_status == SUCCESS
            assert client.backend is not None
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert isinstance(e, test_case.expected_error)
    print("test execution complete")


# --- Delegation tests ---


def test_optimize_delegates(mock_backend):
    """Test that optimize() delegates to backend.optimize()."""
    print("Executing test: optimize delegates")
    mock_backend.backend.optimize.return_value = "exp-123"

    template = TrainJobTemplate(
        trainer=CustomTrainer(func=lambda: None, num_nodes=1),
    )
    search_space = {"lr": Search.uniform(min=0.001, max=0.1)}

    result = mock_backend.optimize(
        template,
        search_space=search_space,
        objectives=[Objective()],
        algorithm=RandomSearch(),
    )

    assert result == "exp-123"
    mock_backend.backend.optimize.assert_called_once()
    print("test execution complete")


def test_list_jobs_delegates(mock_backend):
    """Test that list_jobs() delegates to backend.list_jobs()."""
    print("Executing test: list_jobs delegates")
    mock_backend.backend.list_jobs.return_value = []
    result = mock_backend.list_jobs()
    assert result == []
    mock_backend.backend.list_jobs.assert_called_once()
    print("test execution complete")


def test_get_job_delegates(mock_backend):
    """Test that get_job() delegates to backend.get_job()."""
    print("Executing test: get_job delegates")
    sentinel = Mock(spec=OptimizationJob)
    mock_backend.backend.get_job.return_value = sentinel
    result = mock_backend.get_job(name="test-job")
    assert result is sentinel
    mock_backend.backend.get_job.assert_called_once_with(name="test-job")
    print("test execution complete")


def test_get_job_logs_delegates(mock_backend):
    """Test that get_job_logs() delegates to backend.get_job_logs()."""
    print("Executing test: get_job_logs delegates")
    mock_backend.backend.get_job_logs.return_value = iter(["line1", "line2"])
    result = list(mock_backend.get_job_logs(name="job-1", trial_name="t-1", follow=True))
    assert result == ["line1", "line2"]
    mock_backend.backend.get_job_logs.assert_called_once_with(
        name="job-1", trial_name="t-1", follow=True
    )
    print("test execution complete")


def test_get_best_results_delegates(mock_backend):
    """Test that get_best_results() delegates to backend.get_best_results()."""
    print("Executing test: get_best_results delegates")
    mock_backend.backend.get_best_results.return_value = None
    result = mock_backend.get_best_results(name="job-1")
    assert result is None
    mock_backend.backend.get_best_results.assert_called_once_with(name="job-1")
    print("test execution complete")


def test_wait_for_job_status_delegates(mock_backend):
    """Test that wait_for_job_status() delegates to backend.wait_for_job_status()."""
    print("Executing test: wait_for_job_status delegates")
    sentinel = Mock(spec=OptimizationJob)
    mock_backend.backend.wait_for_job_status.return_value = sentinel
    result = mock_backend.wait_for_job_status(name="job-1", timeout=60)
    assert result is sentinel
    mock_backend.backend.wait_for_job_status.assert_called_once()
    print("test execution complete")


def test_delete_job_delegates(mock_backend):
    """Test that delete_job() delegates to backend.delete_job()."""
    print("Executing test: delete_job delegates")
    mock_backend.delete_job(name="job-1")
    mock_backend.backend.delete_job.assert_called_once_with(name="job-1")
    print("test execution complete")


def test_get_job_events_delegates(mock_backend):
    """Test that get_job_events() delegates to backend.get_job_events()."""
    print("Executing test: get_job_events delegates")
    mock_backend.backend.get_job_events.return_value = []
    result = mock_backend.get_job_events(name="job-1")
    assert result == []
    mock_backend.backend.get_job_events.assert_called_once_with(name="job-1")
    print("test execution complete")
