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

"""Unit tests for PodmanClientAdapter."""

from contextlib import nullcontext
import sys
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


def _make_podman_module() -> tuple[ModuleType, MagicMock]:
    """Build a fake ``podman`` module and return (module, MockPodmanClient)."""
    mock_module = ModuleType("podman")
    mock_client_cls = MagicMock(name="PodmanClient")
    mock_module.PodmanClient = mock_client_cls  # type: ignore[attr-defined]
    return mock_module, mock_client_cls


def _create_adapter(
    host: str | None = None,
) -> "PodmanClientAdapter":  # noqa: F821
    """Import and instantiate PodmanClientAdapter with a mocked podman."""
    mock_module, mock_client_cls = _make_podman_module()
    with patch.dict(sys.modules, {"podman": mock_module}):
        from kubeflow.trainer.backends.container.adapters.podman import (
            PodmanClientAdapter,
        )

        adapter = PodmanClientAdapter(host=host)
    adapter._mock_client_cls = mock_client_cls
    return adapter


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="init with host",
            expected_status=SUCCESS,
            config={"host": "unix:///run/podman/podman.sock"},
        ),
        TestCase(
            name="init without host",
            expected_status=SUCCESS,
            config={"host": None},
        ),
        TestCase(
            name="init import error",
            expected_status=FAILED,
            expected_error=ImportError,
        ),
    ],
)
def test_init(test_case: TestCase) -> None:
    """Test PodmanClientAdapter initialization."""
    print(f"Executing test: {test_case.name}")

    if test_case.expected_error is ImportError:
        with (
            patch.dict(sys.modules, {"podman": None}),
            pytest.raises(ImportError, match="podman"),
        ):
            from kubeflow.trainer.backends.container.adapters.podman import (
                PodmanClientAdapter,
            )

            PodmanClientAdapter()
    else:
        host = test_case.config["host"]
        adapter = _create_adapter(host=host)
        assert adapter._runtime_type == "podman"
        assert adapter.client is not None
        if host:
            adapter._mock_client_cls.assert_called_once_with(base_url=host)
        else:
            adapter._mock_client_cls.assert_called_once_with()

    print("test execution complete")


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------
def test_ping() -> None:
    """Test ping delegates to client."""
    print("Executing test: ping")
    adapter = _create_adapter()
    adapter.ping()
    adapter.client.ping.assert_called_once()
    print("test execution complete")


# ---------------------------------------------------------------------------
# create_network
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="network already exists",
            expected_status=SUCCESS,
            config={"name": "existing-net", "labels": {"app": "test"}, "existing": True},
            expected_output="existing-net",
        ),
        TestCase(
            name="new network created",
            expected_status=SUCCESS,
            config={"name": "new-net", "labels": {"app": "train"}, "existing": False},
            expected_output="new-net",
        ),
    ],
)
def test_create_network(test_case: TestCase) -> None:
    """Test create_network returns name and applies correct params."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()
    name = test_case.config["name"]
    labels = test_case.config["labels"]

    if test_case.config["existing"]:
        adapter.client.networks.get.return_value = Mock()
    else:
        adapter.client.networks.get.side_effect = Exception("not found")

    result = adapter.create_network(name, labels)
    assert result == test_case.expected_output

    if not test_case.config["existing"]:
        adapter.client.networks.create.assert_called_once_with(
            name=name,
            driver="bridge",
            dns_enabled=True,
            labels=labels,
        )
    else:
        adapter.client.networks.create.assert_not_called()

    print("test execution complete")


# ---------------------------------------------------------------------------
# delete_network
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete existing network",
            expected_status=SUCCESS,
            config={"network_id": "net-123", "missing": False},
        ),
        TestCase(
            name="delete missing network silently",
            expected_status=SUCCESS,
            config={"network_id": "missing", "missing": True},
        ),
    ],
)
def test_delete_network(test_case: TestCase) -> None:
    """Test delete_network removes or silently ignores errors."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()
    net_id = test_case.config["network_id"]

    if test_case.config["missing"]:
        adapter.client.networks.get.side_effect = Exception("gone")

    adapter.delete_network(net_id)

    if not test_case.config["missing"]:
        mock_net = adapter.client.networks.get.return_value
        mock_net.remove.assert_called_once()

    print("test execution complete")


# ---------------------------------------------------------------------------
# create_and_start_container
# ---------------------------------------------------------------------------
def test_create_and_start_container() -> None:
    """Test create_and_start_container delegates to containers.run."""
    print("Executing test: create_and_start_container")
    adapter = _create_adapter()
    mock_container = Mock(id="ctr-abc")
    adapter.client.containers.run.return_value = mock_container

    cid = adapter.create_and_start_container(
        image="python:3.10",
        command=["python", "train.py"],
        name="worker-0",
        network_id="net-1",
        environment={"LR": "0.01"},
        labels={"job": "train"},
        volumes={"/data": {"bind": "/mnt/data", "mode": "rw"}},
        working_dir="/app",
    )
    assert cid == "ctr-abc"
    adapter.client.containers.run.assert_called_once_with(
        image="python:3.10",
        command=["python", "train.py"],
        name="worker-0",
        network="net-1",
        working_dir="/app",
        environment={"LR": "0.01"},
        labels={"job": "train"},
        volumes={"/data": {"bind": "/mnt/data", "mode": "rw"}},
        detach=True,
        remove=False,
    )
    print("test execution complete")


# ---------------------------------------------------------------------------
# container_logs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="follow logs streaming",
            expected_status=SUCCESS,
            config={"follow": True},
            expected_output=["hello\n", "world\n"],
        ),
        TestCase(
            name="non-follow logs bytes",
            expected_status=SUCCESS,
            config={"follow": False},
            expected_output=["done\n"],
        ),
        TestCase(
            name="non-follow logs string fallback",
            expected_status=SUCCESS,
            config={"follow": False, "logs_type": "str"},
            expected_output=["done-str"],
        ),
    ],
)
def test_container_logs(test_case: TestCase) -> None:
    """Test container_logs yields decoded chunks."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()
    mock_container = Mock()
    adapter.client.containers.get.return_value = mock_container

    follow = test_case.config["follow"]

    if follow:
        mock_container.logs.return_value = iter([b"hello\n", b"world\n"])
    elif test_case.config.get("logs_type") == "str":
        mock_container.logs.return_value = "done-str"
    else:
        mock_container.logs.return_value = b"done\n"

    result = list(adapter.container_logs("ctr-1", follow=follow))
    assert result == test_case.expected_output
    mock_container.logs.assert_called_once_with(stream=bool(follow), follow=bool(follow))
    print("test execution complete")


# ---------------------------------------------------------------------------
# stop / remove / pull
# ---------------------------------------------------------------------------
def test_stop_container() -> None:
    """Test stop_container delegates to container.stop."""
    print("Executing test: stop_container")
    adapter = _create_adapter()
    mock_container = Mock()
    adapter.client.containers.get.return_value = mock_container

    adapter.stop_container("ctr-1", timeout=5)
    mock_container.stop.assert_called_once_with(timeout=5)
    print("test execution complete")


def test_remove_container() -> None:
    """Test remove_container delegates to container.remove."""
    print("Executing test: remove_container")
    adapter = _create_adapter()
    mock_container = Mock()
    adapter.client.containers.get.return_value = mock_container

    adapter.remove_container("ctr-1", force=True)
    mock_container.remove.assert_called_once_with(force=True)
    print("test execution complete")


def test_pull_image() -> None:
    """Test pull_image delegates to client.images.pull."""
    print("Executing test: pull_image")
    adapter = _create_adapter()
    adapter.pull_image("python:3.10")
    adapter.client.images.pull.assert_called_once_with("python:3.10")
    print("test execution complete")


# ---------------------------------------------------------------------------
# image_exists
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="image found",
            expected_status=SUCCESS,
            config={"image": "python:3.10", "exists": True},
            expected_output=True,
        ),
        TestCase(
            name="image not found",
            expected_status=SUCCESS,
            config={"image": "missing:latest", "exists": False},
            expected_output=False,
        ),
    ],
)
def test_image_exists(test_case: TestCase) -> None:
    """Test image_exists returns bool based on images.get."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()

    if test_case.config["exists"]:
        adapter.client.images.get.return_value = Mock()
    else:
        adapter.client.images.get.side_effect = Exception("not found")

    result = adapter.image_exists(test_case.config["image"])
    assert result is test_case.expected_output
    print("test execution complete")


# ---------------------------------------------------------------------------
# run_oneoff_container
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="oneoff returns bytes",
            expected_status=SUCCESS,
            config={"logs": b"output\n"},
            expected_output="output\n",
        ),
        TestCase(
            name="oneoff returns string fallback",
            expected_status=SUCCESS,
            config={"logs": "str-output"},
            expected_output="str-output",
        ),
        TestCase(
            name="oneoff failure",
            expected_status=FAILED,
            expected_error=RuntimeError,
        ),
    ],
)
def test_run_oneoff_container(test_case: TestCase) -> None:
    """Test run_oneoff_container with create+start+wait+logs pattern."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()

    expectation = (
        pytest.raises(RuntimeError, match="One-off container failed")
        if test_case.expected_error
        else nullcontext()
    )

    if test_case.expected_error:
        adapter.client.containers.create.side_effect = Exception("boom")
    else:
        mock_container = Mock()
        mock_container.logs.return_value = test_case.config["logs"]
        adapter.client.containers.create.return_value = mock_container

    with expectation:
        result = adapter.run_oneoff_container("img:1", ["echo", "hi"])
        assert result == test_case.expected_output
        mock_container.start.assert_called_once()
        mock_container.wait.assert_called_once()
        adapter.client.containers.create.assert_called_once_with(
            image="img:1",
            command=["echo", "hi"],
            detach=False,
            remove=True,
        )

    print("test execution complete")


# ---------------------------------------------------------------------------
# container_status
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="running container",
            expected_status=SUCCESS,
            config={"status": "running"},
            expected_output=("running", None),
        ),
        TestCase(
            name="exited container with code",
            expected_status=SUCCESS,
            config={"status": "exited", "exit_code": 1},
            expected_output=("exited", 1),
        ),
        TestCase(
            name="container not found",
            expected_status=SUCCESS,
            config={"missing": True},
            expected_output=("unknown", None),
        ),
    ],
)
def test_container_status(test_case: TestCase) -> None:
    """Test container_status returns (status, exit_code) tuple."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()

    if test_case.config.get("missing"):
        adapter.client.containers.get.side_effect = Exception("gone")
    else:
        mock_container = Mock()
        mock_container.status = test_case.config["status"]
        if test_case.config["status"] == "exited":
            mock_container.attrs = {"State": {"ExitCode": test_case.config["exit_code"]}}
        adapter.client.containers.get.return_value = mock_container

    result = adapter.container_status("ctr-1")
    assert result == test_case.expected_output
    print("test execution complete")


# ---------------------------------------------------------------------------
# get_container_ip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="ip found by exact network",
            expected_status=SUCCESS,
            config={"network_id": "my-net", "ip": "10.0.0.5", "fallback": False},
            expected_output="10.0.0.5",
        ),
        TestCase(
            name="ip fallback to first network",
            expected_status=SUCCESS,
            config={"network_id": "unknown-net", "ip": "10.0.0.9", "fallback": True},
            expected_output="10.0.0.9",
        ),
        TestCase(
            name="container not found returns None",
            expected_status=SUCCESS,
            config={"missing": True},
            expected_output=None,
        ),
    ],
)
def test_get_container_ip(test_case: TestCase) -> None:
    """Test get_container_ip extracts IP from NetworkSettings."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()

    if test_case.config.get("missing"):
        adapter.client.containers.get.side_effect = Exception("gone")
    else:
        net_id = test_case.config["network_id"]
        ip = test_case.config["ip"]
        mock_container = Mock()
        if test_case.config["fallback"]:
            mock_container.attrs = {
                "NetworkSettings": {
                    "Networks": {
                        "other-net": {"IPAddress": ip},
                    }
                }
            }
        else:
            mock_container.attrs = {
                "NetworkSettings": {
                    "Networks": {
                        net_id: {"IPAddress": ip},
                    }
                }
            }
        adapter.client.containers.get.return_value = mock_container

    result = adapter.get_container_ip("ctr-1", test_case.config.get("network_id", "x"))
    assert result == test_case.expected_output
    print("test execution complete")


# ---------------------------------------------------------------------------
# list_containers  (verify filter workaround for single-item lists)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="single-item filter list flattened",
            expected_status=SUCCESS,
            config={
                "filters": {"label": ["app=train"], "name": ["worker"]},
            },
        ),
        TestCase(
            name="multi-item filter list kept",
            expected_status=SUCCESS,
            config={
                "filters": {"label": ["app=train", "job=1"]},
            },
        ),
        TestCase(
            name="list_containers exception returns empty",
            expected_status=SUCCESS,
            config={"filters": {}, "error": True},
            expected_output=[],
        ),
    ],
)
def test_list_containers(test_case: TestCase) -> None:
    """Test list_containers with podman-py single-item filter workaround."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()
    filters = dict(test_case.config["filters"])

    if test_case.config.get("error"):
        adapter.client.containers.list.side_effect = Exception("fail")
        result = adapter.list_containers(filters=filters)
        assert result == []
    else:
        mock_ctr = Mock()
        mock_ctr.id = "ctr-1"
        mock_ctr.name = "worker-0"
        mock_ctr.labels = {"app": "train"}
        mock_ctr.status = "running"
        mock_ctr.attrs = {"Created": "2025-01-01T00:00:00Z"}
        adapter.client.containers.list.return_value = [mock_ctr]

        original_filters = dict(filters)
        result = adapter.list_containers(filters=filters)

        for k, v in original_filters.items():
            if len(v) == 1:
                assert filters[k] == v[0]
            else:
                assert filters[k] == v

        mock_ctr.reload.assert_called_once()
        assert len(result) == 1
        assert result[0]["id"] == "ctr-1"
        assert result[0]["name"] == "worker-0"
        assert result[0]["labels"] == {"app": "train"}
        assert result[0]["status"] == "running"

    print("test execution complete")


# ---------------------------------------------------------------------------
# get_network  (verify lowercase "labels")
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="existing network with lowercase labels",
            expected_status=SUCCESS,
            config={"network_id": "net-1"},
            expected_output={
                "id": "net-id-1",
                "name": "my-net",
                "labels": {"managed-by": "kubeflow"},
            },
        ),
        TestCase(
            name="network not found",
            expected_status=SUCCESS,
            config={"network_id": "missing"},
            expected_output=None,
        ),
    ],
)
def test_get_network(test_case: TestCase) -> None:
    """Test get_network uses lowercase 'labels' key from inspect."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()
    net_id = test_case.config["network_id"]

    if test_case.expected_output is None:
        adapter.client.networks.get.side_effect = Exception("missing")
    else:
        mock_net = Mock()
        mock_net.attrs = {
            "ID": "net-id-1",
            "Name": "my-net",
            "labels": {"managed-by": "kubeflow"},
        }
        adapter.client.networks.get.return_value = mock_net

    result = adapter.get_network(net_id)
    assert result == test_case.expected_output
    print("test execution complete")


# ---------------------------------------------------------------------------
# wait_for_container
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait returns int exit code",
            expected_status=SUCCESS,
            config={"wait_result": 0},
            expected_output=0,
        ),
        TestCase(
            name="wait returns non-zero exit code",
            expected_status=SUCCESS,
            config={"wait_result": 137},
            expected_output=137,
        ),
        TestCase(
            name="wait timeout raises TimeoutError",
            expected_status=FAILED,
            expected_error=TimeoutError,
        ),
    ],
)
def test_wait_for_container(test_case: TestCase) -> None:
    """Test wait_for_container returns int result directly."""
    print(f"Executing test: {test_case.name}")
    adapter = _create_adapter()
    mock_container = Mock()
    adapter.client.containers.get.return_value = mock_container

    expectation = (
        pytest.raises(TimeoutError) if test_case.expected_error is TimeoutError else nullcontext()
    )

    if test_case.expected_error is TimeoutError:
        mock_container.wait.side_effect = Exception("timeout waiting for container")
    else:
        mock_container.wait.return_value = test_case.config["wait_result"]

    with expectation:
        result = adapter.wait_for_container("ctr-1", timeout=30)
        assert result == test_case.expected_output
        assert isinstance(result, int)

    print("test execution complete")
