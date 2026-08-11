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

"""Unit tests for kubeflow.spark.backends.base module."""

from collections.abc import Iterator

from pyspark.sql import SparkSession
import pytest

from kubeflow.spark.backends.base import RuntimeBackend
from kubeflow.spark.test.common import FAILED, SUCCESS, TestCase
from kubeflow.spark.types.types import (
    Driver,
    Executor,
    FileJob,
    FuncJob,
    SparkConnectInfo,
    SparkConnectState,
    SparkJob,
    SparkJobStatus,
)


class ConcreteBackend(RuntimeBackend):
    """Minimal concrete implementation for testing the ABC contract."""

    def create_and_connect(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
        timeout: int = 300,
        connect_timeout: int = 120,
    ) -> SparkSession:
        return None

    def get_session(self, name: str) -> SparkConnectInfo:
        return SparkConnectInfo(name=name, namespace="default", state=SparkConnectState.READY)

    def list_sessions(self) -> list[SparkConnectInfo]:
        return []

    def delete_session(self, name: str) -> None:
        return None

    def get_session_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        yield "log line 1"

    def submit_job(
        self,
        job: FileJob | FuncJob,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
    ) -> SparkJob:
        return SparkJob(name="test-job", namespace="default", status=SparkJobStatus.CREATED)

    def get_job(self, name: str) -> SparkJob:
        return SparkJob(name=name, namespace="default", status=SparkJobStatus.CREATED)

    def list_jobs(self) -> list[SparkJob]:
        return []

    def delete_job(self, name: str) -> None:
        return None

    def wait_for_job_status(
        self,
        name: str,
        status: set[SparkJobStatus] = {SparkJobStatus.COMPLETED},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> SparkJob:
        return SparkJob(name=name, namespace="default", status=SparkJobStatus.COMPLETED)

    def get_job_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        yield "job log line 1"


# --------------------------
# Tests
# --------------------------


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
            name="get_session returns session info",
            expected_status=SUCCESS,
            config={"method": "get_session", "name": "my-session"},
            expected_output="my-session",
        ),
        TestCase(
            name="list_sessions returns empty list",
            expected_status=SUCCESS,
            config={"method": "list_sessions"},
            expected_output=[],
        ),
        TestCase(
            name="delete_session returns None",
            expected_status=SUCCESS,
            config={"method": "delete_session", "name": "my-session"},
            expected_output=None,
        ),
        TestCase(
            name="get_session_logs returns log lines",
            expected_status=SUCCESS,
            config={"method": "get_session_logs", "name": "my-session"},
            expected_output=["log line 1"],
        ),
        TestCase(
            name="submit_job returns SparkJob",
            expected_status=SUCCESS,
            config={"method": "submit_job"},
            expected_output=SparkJobStatus.CREATED,
        ),
        TestCase(
            name="get_job returns SparkJob",
            expected_status=SUCCESS,
            config={"method": "get_job", "name": "test-job"},
            expected_output="test-job",
        ),
        TestCase(
            name="list_jobs returns empty list",
            expected_status=SUCCESS,
            config={"method": "list_jobs"},
            expected_output=[],
        ),
        TestCase(
            name="delete_job returns None",
            expected_status=SUCCESS,
            config={"method": "delete_job", "name": "test-job"},
            expected_output=None,
        ),
        TestCase(
            name="wait_for_job_status returns succeeded",
            expected_status=SUCCESS,
            config={"method": "wait_for_job_status", "name": "test-job"},
            expected_output=SparkJobStatus.COMPLETED,
        ),
        TestCase(
            name="get_job_logs returns log lines",
            expected_status=SUCCESS,
            config={"method": "get_job_logs", "name": "test-job"},
            expected_output=["job log line 1"],
        ),
    ],
)
def test_concrete_backend_methods(test_case: TestCase):
    """Test that concrete backend methods fulfill the ABC contract."""
    print("Executing test:", test_case.name)
    backend = ConcreteBackend()
    method_name = test_case.config["method"]
    name_arg = test_case.config.get("name")

    if method_name == "get_session":
        result = backend.get_session(name_arg)
        assert result.name == test_case.expected_output
    elif method_name == "list_sessions":
        result = backend.list_sessions()
        assert result == test_case.expected_output
    elif method_name == "delete_session":
        result = backend.delete_session(name_arg)
        assert result is test_case.expected_output
    elif method_name == "get_session_logs":
        result = list(backend.get_session_logs(name_arg))
        assert result == test_case.expected_output
    elif method_name == "submit_job":
        job = FileJob(file_source="test.py")
        result = backend.submit_job(job)
        assert result.status == test_case.expected_output
    elif method_name == "get_job":
        result = backend.get_job(name_arg)
        assert result.name == test_case.expected_output
    elif method_name == "list_jobs":
        result = backend.list_jobs()
        assert result == test_case.expected_output
    elif method_name == "delete_job":
        result = backend.delete_job(name_arg)
        assert result is test_case.expected_output
    elif method_name == "wait_for_job_status":
        result = backend.wait_for_job_status(name_arg)
        assert result.status == test_case.expected_output
    elif method_name == "get_job_logs":
        result = list(backend.get_job_logs(name_arg))
        assert result == test_case.expected_output

    print("test execution complete")


class PartialBackend(RuntimeBackend):
    """Backend that delegates all methods to super() to verify NotImplementedError."""

    def create_and_connect(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
        timeout: int = 300,
        connect_timeout: int = 120,
    ) -> SparkSession:
        return super().create_and_connect(
            num_executors=num_executors,
            resources_per_executor=resources_per_executor,
            spark_conf=spark_conf,
            driver=driver,
            executor=executor,
            options=options,
            timeout=timeout,
            connect_timeout=connect_timeout,
        )

    def get_session(self, name: str) -> SparkConnectInfo:
        return super().get_session(name)

    def list_sessions(self) -> list[SparkConnectInfo]:
        return super().list_sessions()

    def delete_session(self, name: str) -> None:
        return super().delete_session(name)

    def get_session_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        return super().get_session_logs(name, follow)

    def submit_job(
        self,
        job: FileJob | FuncJob,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
    ) -> SparkJob:
        return super().submit_job(
            job, num_executors=num_executors, resources_per_executor=resources_per_executor
        )

    def get_job(self, name: str) -> SparkJob:
        return super().get_job(name)

    def list_jobs(self) -> list[SparkJob]:
        return super().list_jobs()

    def delete_job(self, name: str) -> None:
        return super().delete_job(name)

    def wait_for_job_status(
        self,
        name: str,
        status: set[SparkJobStatus] = {SparkJobStatus.COMPLETED},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> SparkJob:
        return super().wait_for_job_status(
            name,
            status=status,
            timeout=timeout,
            polling_interval=polling_interval,
        )

    def get_job_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        return super().get_job_logs(name, follow)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="create_and_connect raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "create_and_connect"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_session raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_session", "name": "s"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="list_sessions raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "list_sessions"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="delete_session raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "delete_session", "name": "s"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_session_logs raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_session_logs", "name": "s"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="submit_job raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "submit_job"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_job raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_job", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="list_jobs raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "list_jobs"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="delete_job raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "delete_job", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="wait_for_job_status raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "wait_for_job_status", "name": "j"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_job_logs raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_job_logs", "name": "j"},
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
        if method_name == "create_and_connect":
            backend.create_and_connect()
        elif method_name == "get_session":
            backend.get_session(name_arg)
        elif method_name == "list_sessions":
            backend.list_sessions()
        elif method_name == "delete_session":
            backend.delete_session(name_arg)
        elif method_name == "get_session_logs":
            backend.get_session_logs(name_arg)
        elif method_name == "submit_job":
            backend.submit_job(FileJob(file_source="test.py"))
        elif method_name == "get_job":
            backend.get_job(name_arg)
        elif method_name == "list_jobs":
            backend.list_jobs()
        elif method_name == "delete_job":
            backend.delete_job(name_arg)
        elif method_name == "wait_for_job_status":
            backend.wait_for_job_status(name_arg)
        elif method_name == "get_job_logs":
            backend.get_job_logs(name_arg)

    print("test execution complete")
