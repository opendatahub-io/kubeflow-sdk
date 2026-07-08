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

import pytest

from kubeflow.spark.backends.base import RuntimeBackend
from kubeflow.spark.test.common import FAILED, SUCCESS, TestCase
from kubeflow.spark.types.types import Driver, Executor, SparkConnectInfo, SparkConnectState


class ConcreteBackend(RuntimeBackend):
    """Minimal concrete implementation for testing the ABC contract."""

    def connect(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
    ) -> SparkConnectInfo:
        return SparkConnectInfo(name="test", namespace="default", state=SparkConnectState.READY)

    def get_session(self, name: str) -> SparkConnectInfo:
        return SparkConnectInfo(name=name, namespace="default", state=SparkConnectState.READY)

    def list_sessions(self) -> list[SparkConnectInfo]:
        return []

    def delete_session(self, name: str) -> None:
        return None

    def _wait_for_session_ready(
        self, name: str, timeout: int = 300, polling_interval: int = 2
    ) -> SparkConnectInfo:
        return SparkConnectInfo(name=name, namespace="default", state=SparkConnectState.READY)

    def get_session_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        yield "log line 1"


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
            name="connect returns SparkConnectInfo",
            expected_status=SUCCESS,
            config={"method": "connect"},
            expected_output=SparkConnectState.READY,
        ),
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
            name="wait_for_session_ready returns ready info",
            expected_status=SUCCESS,
            config={"method": "_wait_for_session_ready", "name": "my-session"},
            expected_output=SparkConnectState.READY,
        ),
        TestCase(
            name="get_session_logs returns log lines",
            expected_status=SUCCESS,
            config={"method": "get_session_logs", "name": "my-session"},
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

    if method_name == "connect":
        result = backend.connect()
        assert result.state == test_case.expected_output
    elif method_name == "get_session":
        result = backend.get_session(name_arg)
        assert result.name == test_case.expected_output
    elif method_name == "list_sessions":
        result = backend.list_sessions()
        assert result == test_case.expected_output
    elif method_name == "delete_session":
        result = backend.delete_session(name_arg)
        assert result is test_case.expected_output
    elif method_name == "_wait_for_session_ready":
        result = backend._wait_for_session_ready(name_arg)
        assert result.state == test_case.expected_output
    elif method_name == "get_session_logs":
        result = list(backend.get_session_logs(name_arg))
        assert result == test_case.expected_output

    print("test execution complete")


class PartialBackend(RuntimeBackend):
    """Backend that delegates all methods to super() to verify NotImplementedError."""

    def connect(
        self,
        num_executors: int | None = None,
        resources_per_executor: dict[str, str] | None = None,
        spark_conf: dict[str, str] | None = None,
        driver: Driver | None = None,
        executor: Executor | None = None,
        options: list | None = None,
    ) -> SparkConnectInfo:
        return super().connect(
            num_executors=num_executors,
            resources_per_executor=resources_per_executor,
            spark_conf=spark_conf,
            driver=driver,
            executor=executor,
            options=options,
        )

    def get_session(self, name: str) -> SparkConnectInfo:
        return super().get_session(name)

    def list_sessions(self) -> list[SparkConnectInfo]:
        return super().list_sessions()

    def delete_session(self, name: str) -> None:
        return super().delete_session(name)

    def _wait_for_session_ready(
        self, name: str, timeout: int = 300, polling_interval: int = 2
    ) -> SparkConnectInfo:
        return super()._wait_for_session_ready(name, timeout, polling_interval)

    def get_session_logs(self, name: str, follow: bool = False) -> Iterator[str]:
        return super().get_session_logs(name, follow)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="connect raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "connect"},
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
            name="wait_for_session_ready raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "_wait_for_session_ready", "name": "s"},
            expected_error=NotImplementedError,
        ),
        TestCase(
            name="get_session_logs raises NotImplementedError via super",
            expected_status=FAILED,
            config={"method": "get_session_logs", "name": "s"},
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

    try:
        if method_name == "connect":
            backend.connect()
        elif method_name == "get_session":
            backend.get_session(name_arg)
        elif method_name == "list_sessions":
            backend.list_sessions()
        elif method_name == "delete_session":
            backend.delete_session(name_arg)
        elif method_name == "_wait_for_session_ready":
            backend._wait_for_session_ready(name_arg)
        elif method_name == "get_session_logs":
            backend.get_session_logs(name_arg)

        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )
    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
