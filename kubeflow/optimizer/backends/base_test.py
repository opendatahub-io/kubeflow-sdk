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

"""Unit tests for kubeflow.optimizer.backends.base module."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from kubeflow.optimizer.backends.base import RuntimeBackend
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Objective,
    OptimizationJob,
    Result,
    TrialConfig,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types.types import CustomTrainer, Event, TrainJobTemplate


class ConcreteBackend(RuntimeBackend):
    """Minimal concrete implementation for testing the ABC contract."""

    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        search_space: dict[str, Any],
        trial_config: TrialConfig | None = None,
        objectives: list[Objective] | None = None,
        algorithm: RandomSearch | None = None,
    ) -> str:
        return "test-experiment"

    def list_jobs(self) -> list[OptimizationJob]:
        return []

    def get_job(self, name: str) -> OptimizationJob:
        raise RuntimeError("not found")

    def get_job_logs(
        self,
        name: str,
        trial_name: str | None,
        follow: bool,
    ) -> Iterator[str]:
        yield "log line 1"

    def get_best_results(self, name: str) -> Result | None:
        return None

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] | None = None,
        timeout: int = 3600,
        polling_interval: int = 2,
        callbacks: list[Callable[[OptimizationJob], None]] | None = None,
    ) -> OptimizationJob:
        raise TimeoutError("timed out")

    def get_job_events(self, name: str) -> list[Event]:
        return []

    def delete_job(self, name: str):
        return None


def test_cannot_instantiate_abstract_class():
    """Test that RuntimeBackend cannot be instantiated directly."""
    print("Executing test: cannot instantiate abstract class")
    with pytest.raises(TypeError):
        RuntimeBackend()
    print("test execution complete")


def test_concrete_subclass_instantiation():
    """Test that a concrete subclass implementing all methods can be instantiated."""
    print("Executing test: concrete subclass instantiation")
    backend = ConcreteBackend()
    assert isinstance(backend, RuntimeBackend)
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="optimize returns experiment name",
            expected_status=SUCCESS,
            config={"method": "optimize"},
            expected_output="test-experiment",
        ),
        TestCase(
            name="list_jobs returns empty list",
            expected_status=SUCCESS,
            config={"method": "list_jobs"},
            expected_output=[],
        ),
        TestCase(
            name="get_best_results returns None",
            expected_status=SUCCESS,
            config={"method": "get_best_results", "name": "job-1"},
            expected_output=None,
        ),
        TestCase(
            name="get_job_events returns empty list",
            expected_status=SUCCESS,
            config={"method": "get_job_events", "name": "job-1"},
            expected_output=[],
        ),
        TestCase(
            name="delete_job returns None",
            expected_status=SUCCESS,
            config={"method": "delete_job", "name": "job-1"},
            expected_output=None,
        ),
        TestCase(
            name="get_job_logs returns log lines",
            expected_status=SUCCESS,
            config={"method": "get_job_logs", "name": "job-1"},
            expected_output=["log line 1"],
        ),
    ],
)
def test_concrete_backend_methods(test_case: TestCase):
    """Test that concrete backend methods fulfill the ABC contract."""
    print("Executing test:", test_case.name)
    backend = ConcreteBackend()
    method_name = test_case.config["method"]
    name_arg = test_case.config.get("name")

    if method_name == "optimize":
        template = TrainJobTemplate(
            trainer=CustomTrainer(func=lambda: None, num_nodes=1),
        )
        result = backend.optimize(trial_template=template, search_space={"lr": None})
        assert result == test_case.expected_output
    elif method_name == "list_jobs":
        result = backend.list_jobs()
        assert result == test_case.expected_output
    elif method_name == "get_best_results":
        result = backend.get_best_results(name_arg)
        assert result is test_case.expected_output
    elif method_name == "get_job_events":
        result = backend.get_job_events(name_arg)
        assert result == test_case.expected_output
    elif method_name == "delete_job":
        result = backend.delete_job(name_arg)
        assert result is test_case.expected_output
    elif method_name == "get_job_logs":
        result = list(backend.get_job_logs(name_arg, trial_name=None, follow=False))
        assert result == test_case.expected_output

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_job raises RuntimeError",
            expected_status=FAILED,
            config={"method": "get_job", "name": "missing"},
            expected_error=RuntimeError,
        ),
        TestCase(
            name="wait_for_job_status raises TimeoutError",
            expected_status=FAILED,
            config={"method": "wait_for_job_status", "name": "job-1"},
            expected_error=TimeoutError,
        ),
    ],
)
def test_concrete_backend_error_cases(test_case: TestCase):
    """Test that concrete backend methods raise expected errors."""
    print("Executing test:", test_case.name)
    backend = ConcreteBackend()
    method_name = test_case.config["method"]
    name_arg = test_case.config.get("name")

    with pytest.raises(test_case.expected_error):
        if method_name == "get_job":
            backend.get_job(name_arg)
        elif method_name == "wait_for_job_status":
            backend.wait_for_job_status(name_arg)

    print("test execution complete")


class PartialBackend(RuntimeBackend):
    """Backend that delegates all methods to super() to verify NotImplementedError."""

    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        search_space: dict[str, Any],
        trial_config: TrialConfig | None = None,
        objectives: list[Objective] | None = None,
        algorithm: RandomSearch | None = None,
    ) -> str:
        return super().optimize(
            trial_template,
            search_space=search_space,
            trial_config=trial_config,
            objectives=objectives,
            algorithm=algorithm,
        )

    def list_jobs(self) -> list[OptimizationJob]:
        return super().list_jobs()

    def get_job(self, name: str) -> OptimizationJob:
        return super().get_job(name)

    def get_job_logs(
        self,
        name: str,
        trial_name: str | None,
        follow: bool,
    ) -> Iterator[str]:
        return super().get_job_logs(name, trial_name, follow)

    def get_best_results(self, name: str) -> Result | None:
        return super().get_best_results(name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] | None = None,
        timeout: int = 3600,
        polling_interval: int = 2,
        callbacks: list[Callable[[OptimizationJob], None]] | None = None,
    ) -> OptimizationJob:
        return super().wait_for_job_status(name, status, timeout, polling_interval, callbacks)

    def get_job_events(self, name: str) -> list[Event]:
        return super().get_job_events(name)

    def delete_job(self, name: str):
        return super().delete_job(name)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="optimize raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "optimize"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="list_jobs raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "list_jobs"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_job raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_job", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_job_logs raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_job_logs", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_best_results raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_best_results", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="wait_for_job_status raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "wait_for_job_status", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_job_events raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_job_events", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="delete_job raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "delete_job", "name": "j"},
            expected_error=NotImplementedError,
        ),
    ],
)
def test_super_raises_not_implemented(test_case: TestCase):
    """Test that calling super() on each abstract method raises NotImplementedError."""
    print("Executing test:", test_case.name)
    backend = PartialBackend()
    method_name = test_case.config["method"]
    name_arg = test_case.config.get("name")

    with pytest.raises(test_case.expected_error):
        if method_name == "optimize":
            template = TrainJobTemplate(
                trainer=CustomTrainer(func=lambda: None, num_nodes=1),
            )
            backend.optimize(template, search_space={"lr": None})
        elif method_name == "list_jobs":
            backend.list_jobs()
        elif method_name == "get_job":
            backend.get_job(name_arg)
        elif method_name == "get_job_logs":
            backend.get_job_logs(name_arg, None, False)
        elif method_name == "get_best_results":
            backend.get_best_results(name_arg)
        elif method_name == "wait_for_job_status":
            backend.wait_for_job_status(name_arg)
        elif method_name == "get_job_events":
            backend.get_job_events(name_arg)
        elif method_name == "delete_job":
            backend.delete_job(name_arg)

    print("test execution complete")
