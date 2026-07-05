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

"""Unit tests for the DockerClientAdapter class."""

import sys
from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


def _make_mock_docker_module() -> Mock:
    """Build a fake docker module suitable for sys.modules patching."""
    mock_docker = Mock()
    mock_client = Mock()
    mock_docker.from_env.return_value = mock_client
    mock_docker.DockerClient.return_value = mock_client
    return mock_docker


def _create_adapter(host: str | None = None):
    """Create a DockerClientAdapter with the docker module mocked."""
    mock_docker = _make_mock_docker_module()
    with patch.dict(sys.modules, {"docker": mock_docker}):
        from kubeflow.trainer.backends.container.adapters.docker import (
            DockerClientAdapter,
        )

        adapter = DockerClientAdapter(host=host)
    return adapter, mock_docker


@pytest.fixture
def adapter_and_mock():
    """Provide a DockerClientAdapter with a mocked docker client."""
    adapter, mock_docker = _create_adapter()
    return adapter, mock_docker


# --------------------------
# Init tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="init without host uses from_env",
            expected_status=SUCCESS,
            config={"host": None},
        ),
        TestCase(
            name="init with host uses DockerClient",
            expected_status=SUCCESS,
            config={"host": "tcp://localhost:2375"},
        ),
        TestCase(
            name="init raises ImportError when docker not installed",
            expected_status=FAILED,
            expected_error=ImportError,
        ),
    ],
)
def test_init(test_case):
    """Test DockerClientAdapter initialization."""
    print(f"Executing test: {test_case.name}")

    if test_case.expected_error:
        with patch.dict(sys.modules, {"docker": None}), pytest.raises(ImportError):
            from importlib import reload

            import kubeflow.trainer.backends.container.adapters.docker as mod

            reload(mod)
            mod.DockerClientAdapter()
    else:
        host = test_case.config["host"]
        adapter, mock_docker = _create_adapter(host=host)
        if host:
            mock_docker.DockerClient.assert_called_once_with(base_url=host)
        else:
            mock_docker.from_env.assert_called_once()

    print("test execution complete")


# --------------------------
# Ping test
# --------------------------


def test_ping(adapter_and_mock):
    """Test ping delegates to docker client."""
    print("Executing test: ping delegates to client")
    adapter, _ = adapter_and_mock
    adapter.ping()
    adapter.client.ping.assert_called_once()
    print("test execution complete")


# --------------------------
# Network tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="create_network returns existing network",
            expected_status=SUCCESS,
            config={"existing": True},
            expected_output="test-net",
        ),
        TestCase(
            name="create_network creates new network",
            expected_status=SUCCESS,
            config={"existing": False},
            expected_output="test-net",
        ),
    ],
)
def test_create_network(adapter_and_mock, test_case):
    """Test create_network with existing and new networks."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    if test_case.config["existing"]:
        adapter.client.networks.get.return_value = Mock()
    else:
        adapter.client.networks.get.side_effect = Exception("not found")

    result = adapter.create_network("test-net", {"app": "test"})
    assert result == test_case.expected_output

    if test_case.config["existing"]:
        adapter.client.networks.create.assert_not_called()
    else:
        adapter.client.networks.create.assert_called_once_with(
            name="test-net",
            check_duplicate=True,
            labels={"app": "test"},
        )
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete_network removes successfully",
            expected_status=SUCCESS,
            config={"raises": False},
        ),
        TestCase(
            name="delete_network silences errors",
            expected_status=SUCCESS,
            config={"raises": True},
        ),
    ],
)
def test_delete_network(adapter_and_mock, test_case):
    """Test delete_network handles success and errors."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    if test_case.config["raises"]:
        adapter.client.networks.get.side_effect = Exception("not found")
    else:
        mock_net = Mock()
        adapter.client.networks.get.return_value = mock_net

    adapter.delete_network("net-1")

    if not test_case.config["raises"]:
        mock_net.remove.assert_called_once()
    print("test execution complete")


# --------------------------
# Container lifecycle tests
# --------------------------


def test_create_and_start_container(adapter_and_mock):
    """Test create_and_start_container returns container id."""
    print("Executing test: create_and_start_container returns container id")
    adapter, _ = adapter_and_mock
    mock_container = Mock()
    mock_container.id = "abc123"
    adapter.client.containers.run.return_value = mock_container

    result = adapter.create_and_start_container(
        image="python:3.10",
        command=["python", "train.py"],
        name="worker-0",
        network_id="net-1",
        environment={"KEY": "val"},
        labels={"job": "1"},
        volumes={"/data": {"bind": "/mnt/data", "mode": "rw"}},
        working_dir="/app",
    )

    assert result == "abc123"
    adapter.client.containers.run.assert_called_once_with(
        image="python:3.10",
        command=("python", "train.py"),
        name="worker-0",
        detach=True,
        working_dir="/app",
        network="net-1",
        environment={"KEY": "val"},
        labels={"job": "1"},
        volumes={"/data": {"bind": "/mnt/data", "mode": "rw"}},
        auto_remove=False,
    )
    print("test execution complete")


# --------------------------
# Container logs tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="container_logs follow mode yields decoded chunks",
            expected_status=SUCCESS,
            config={"follow": True},
            expected_output=["hello ", "world"],
        ),
        TestCase(
            name="container_logs non-follow returns full output",
            expected_status=SUCCESS,
            config={"follow": False},
            expected_output=["hello world"],
        ),
    ],
)
def test_container_logs(adapter_and_mock, test_case):
    """Test container_logs streaming and non-streaming modes."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock
    mock_container = Mock()

    if test_case.config["follow"]:
        mock_container.logs.return_value = iter([b"hello ", b"world"])
    else:
        mock_container.logs.return_value = b"hello world"

    adapter.client.containers.get.return_value = mock_container

    result = list(adapter.container_logs("cid", follow=test_case.config["follow"]))
    assert result == test_case.expected_output
    print("test execution complete")


# --------------------------
# Stop / Remove / Pull tests
# --------------------------


def test_stop_container(adapter_and_mock):
    """Test stop_container delegates to the container."""
    print("Executing test: stop_container delegates correctly")
    adapter, _ = adapter_and_mock
    mock_container = Mock()
    adapter.client.containers.get.return_value = mock_container

    adapter.stop_container("cid", timeout=5)
    mock_container.stop.assert_called_once_with(timeout=5)
    print("test execution complete")


def test_remove_container(adapter_and_mock):
    """Test remove_container delegates to the container."""
    print("Executing test: remove_container delegates correctly")
    adapter, _ = adapter_and_mock
    mock_container = Mock()
    adapter.client.containers.get.return_value = mock_container

    adapter.remove_container("cid", force=True)
    mock_container.remove.assert_called_once_with(force=True)
    print("test execution complete")


def test_pull_image(adapter_and_mock):
    """Test pull_image delegates to docker client."""
    print("Executing test: pull_image delegates correctly")
    adapter, _ = adapter_and_mock

    adapter.pull_image("python:3.10")
    adapter.client.images.pull.assert_called_once_with("python:3.10")
    print("test execution complete")


# --------------------------
# Image exists tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="image_exists returns True when found",
            expected_status=SUCCESS,
            expected_output=True,
        ),
        TestCase(
            name="image_exists returns False when not found",
            expected_status=SUCCESS,
            config={"raises": True},
            expected_output=False,
        ),
    ],
)
def test_image_exists(adapter_and_mock, test_case):
    """Test image_exists returns correct boolean."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    if test_case.config.get("raises"):
        adapter.client.images.get.side_effect = Exception("not found")
    else:
        adapter.client.images.get.return_value = Mock()

    result = adapter.image_exists("myimage:latest")
    assert result == test_case.expected_output
    print("test execution complete")


# --------------------------
# Run oneoff container tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="run_oneoff_container success with bytes output",
            expected_status=SUCCESS,
            expected_output="output text",
        ),
        TestCase(
            name="run_oneoff_container failure raises RuntimeError",
            expected_status=FAILED,
            expected_error=RuntimeError,
        ),
    ],
)
def test_run_oneoff_container(adapter_and_mock, test_case):
    """Test run_oneoff_container success and failure paths."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    if test_case.expected_error:
        adapter.client.containers.run.side_effect = Exception("boom")
        with pytest.raises(RuntimeError, match="One-off container failed"):
            adapter.run_oneoff_container("img", ["cmd"])
    else:
        adapter.client.containers.run.return_value = b"output text"
        result = adapter.run_oneoff_container("img", ["cmd"])
        assert result == test_case.expected_output
        adapter.client.containers.run.assert_called_once_with(
            image="img",
            command=("cmd",),
            detach=False,
            remove=True,
        )
    print("test execution complete")


# --------------------------
# Container status tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="container_status running",
            expected_status=SUCCESS,
            config={"status": "running"},
            expected_output=("running", None),
        ),
        TestCase(
            name="container_status exited with exit code",
            expected_status=SUCCESS,
            config={"status": "exited", "exit_code": 1},
            expected_output=("exited", 1),
        ),
        TestCase(
            name="container_status error returns unknown",
            expected_status=SUCCESS,
            config={"raises": True},
            expected_output=("unknown", None),
        ),
    ],
)
def test_container_status(adapter_and_mock, test_case):
    """Test container_status returns correct tuple."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    if test_case.config.get("raises"):
        adapter.client.containers.get.side_effect = Exception("gone")
    else:
        mock_container = Mock()
        mock_container.status = test_case.config["status"]
        if test_case.config["status"] == "exited":
            mock_container.attrs = {"State": {"ExitCode": test_case.config["exit_code"]}}
        adapter.client.containers.get.return_value = mock_container

    result = adapter.container_status("cid")
    assert result == test_case.expected_output
    print("test execution complete")


# --------------------------
# Get container IP tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_container_ip found on exact network",
            expected_status=SUCCESS,
            config={"network_id": "net-1", "ip": "172.17.0.2"},
            expected_output="172.17.0.2",
        ),
        TestCase(
            name="get_container_ip not found returns None",
            expected_status=SUCCESS,
            config={"raises": True},
            expected_output=None,
        ),
    ],
)
def test_get_container_ip(adapter_and_mock, test_case):
    """Test get_container_ip returns IP or None."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    if test_case.config.get("raises"):
        adapter.client.containers.get.side_effect = Exception("gone")
    else:
        mock_container = Mock()
        mock_container.attrs = {
            "NetworkSettings": {
                "Networks": {test_case.config["network_id"]: {"IPAddress": test_case.config["ip"]}}
            }
        }
        adapter.client.containers.get.return_value = mock_container

    result = adapter.get_container_ip("cid", "net-1")
    assert result == test_case.expected_output
    print("test execution complete")


# --------------------------
# List containers tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list_containers without filters",
            expected_status=SUCCESS,
            config={"filters": None},
            expected_output=[
                {
                    "id": "c1",
                    "name": "worker-0",
                    "labels": {"job": "1"},
                    "status": "running",
                    "created": "2025-01-01",
                }
            ],
        ),
        TestCase(
            name="list_containers with filters",
            expected_status=SUCCESS,
            config={"filters": {"label": ["job=1"]}},
            expected_output=[
                {
                    "id": "c1",
                    "name": "worker-0",
                    "labels": {"job": "1"},
                    "status": "running",
                    "created": "2025-01-01",
                }
            ],
        ),
    ],
)
def test_list_containers(adapter_and_mock, test_case):
    """Test list_containers returns formatted list."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    mock_c = Mock()
    mock_c.id = "c1"
    mock_c.name = "worker-0"
    mock_c.labels = {"job": "1"}
    mock_c.status = "running"
    mock_c.attrs = {"Created": "2025-01-01"}
    adapter.client.containers.list.return_value = [mock_c]

    result = adapter.list_containers(filters=test_case.config["filters"])
    assert result == test_case.expected_output
    adapter.client.containers.list.assert_called_once_with(
        all=True, filters=test_case.config["filters"]
    )
    print("test execution complete")


# --------------------------
# Get network tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_network found",
            expected_status=SUCCESS,
            expected_output={
                "id": "net-1",
                "name": "test-net",
                "labels": {"app": "test"},
            },
        ),
        TestCase(
            name="get_network not found returns None",
            expected_status=SUCCESS,
            config={"raises": True},
            expected_output=None,
        ),
    ],
)
def test_get_network(adapter_and_mock, test_case):
    """Test get_network returns dict or None."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock

    if test_case.config.get("raises"):
        adapter.client.networks.get.side_effect = Exception("not found")
    else:
        mock_net = Mock()
        mock_net.id = "net-1"
        mock_net.name = "test-net"
        mock_net.attrs = {"Labels": {"app": "test"}}
        adapter.client.networks.get.return_value = mock_net

    result = adapter.get_network("net-1")
    assert result == test_case.expected_output
    print("test execution complete")


# --------------------------
# Wait for container tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait_for_container success",
            expected_status=SUCCESS,
            expected_output=0,
        ),
        TestCase(
            name="wait_for_container timeout raises TimeoutError",
            expected_status=FAILED,
            expected_error=TimeoutError,
        ),
    ],
)
def test_wait_for_container(adapter_and_mock, test_case):
    """Test wait_for_container returns exit code or raises."""
    print(f"Executing test: {test_case.name}")
    adapter, _ = adapter_and_mock
    mock_container = Mock()
    adapter.client.containers.get.return_value = mock_container

    if test_case.expected_error:
        mock_container.wait.side_effect = Exception("timeout exceeded")
        with pytest.raises(TimeoutError):
            adapter.wait_for_container("cid", timeout=30)
    else:
        mock_container.wait.return_value = {"StatusCode": 0}
        result = adapter.wait_for_container("cid", timeout=60)
        assert result == test_case.expected_output
    print("test execution complete")
