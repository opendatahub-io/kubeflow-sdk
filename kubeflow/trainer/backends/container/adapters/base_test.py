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

"""Unit tests for the BaseContainerClientAdapter abstract base class."""

from collections.abc import Iterator

import pytest

from kubeflow.trainer.backends.container.adapters.base import (
    BaseContainerClientAdapter,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


class CompleteAdapter(BaseContainerClientAdapter):
    """Concrete implementation of BaseContainerClientAdapter with all methods."""

    def ping(self):
        pass

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        return "network-id"

    def delete_network(self, network_id: str):
        pass

    def create_and_start_container(
        self,
        image: str,
        command: list[str],
        name: str,
        network_id: str,
        environment: dict[str, str],
        labels: dict[str, str],
        volumes: dict[str, dict[str, str]],
        working_dir: str,
    ) -> str:
        return "container-id"

    def get_container(self, container_id: str):
        return None

    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        yield "log"

    def stop_container(self, container_id: str, timeout: int = 10):
        pass

    def remove_container(self, container_id: str, force: bool = True):
        pass

    def pull_image(self, image: str):
        pass

    def image_exists(self, image: str) -> bool:
        return True

    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        return "output"

    def container_status(self, container_id: str) -> tuple[str, int | None]:
        return ("running", None)

    def get_container_ip(self, container_id: str, network_id: str) -> str | None:
        return "172.17.0.2"

    def list_containers(self, filters: dict[str, list[str]] | None = None) -> list[dict]:
        return []

    def get_network(self, network_id: str) -> dict | None:
        return {"id": network_id}

    def wait_for_container(self, container_id: str, timeout: int | None = None) -> int:
        return 0


class PartialAdapter(BaseContainerClientAdapter):
    """Partial implementation with only ping and create_network."""

    def ping(self):
        pass

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        return "network-id"


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="direct instantiation of BaseContainerClientAdapter raises TypeError",
            expected_status=FAILED,
            config={"cls": BaseContainerClientAdapter},
            expected_error=TypeError,
        ),
        TestCase(
            name="complete subclass instantiates successfully",
            expected_status=SUCCESS,
            config={"cls": CompleteAdapter},
            expected_output=CompleteAdapter,
        ),
        TestCase(
            name="partial subclass raises TypeError",
            expected_status=FAILED,
            config={"cls": PartialAdapter},
            expected_error=TypeError,
        ),
    ],
)
def test_container_adapter_instantiation(test_case):
    """Test BaseContainerClientAdapter instantiation rules for abstract base class."""
    print("Executing test:", test_case.name)
    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            test_case.config["cls"]()
    else:
        instance = test_case.config["cls"]()
        assert isinstance(instance, test_case.expected_output)
    print("test execution complete")
