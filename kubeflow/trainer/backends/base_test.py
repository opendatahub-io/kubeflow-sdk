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

"""Unit tests for the RuntimeBackend abstract base class."""

from collections.abc import Iterator

import pytest

from kubeflow.trainer.backends.base import RuntimeBackend
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


class CompleteBackend(RuntimeBackend):
    """Concrete implementation of RuntimeBackend with all abstract methods."""

    def list_runtimes(self) -> list[types.Runtime]:
        return []

    def get_runtime(self, name: str) -> types.Runtime:
        return types.Runtime(name=name)

    def get_runtime_packages(self, runtime: types.Runtime):
        return []

    def train(self, runtime=None, initializer=None, trainer=None, options=None) -> str:
        return "job-id"

    def list_jobs(self, runtime=None) -> list[types.TrainJob]:
        return []

    def get_job(self, name: str) -> types.TrainJob:
        return types.TrainJob(name=name)

    def get_job_logs(self, name="", follow=False, step="node-0") -> Iterator[str]:
        yield "log line"

    def get_job_events(self, name: str) -> list[types.Event]:
        return []

    def wait_for_job_status(
        self, name="", status=None, timeout=600, polling_interval=2, callbacks=None
    ) -> types.TrainJob:
        return types.TrainJob(name=name)

    def delete_job(self, name: str):
        pass


class PartialBackend(RuntimeBackend):
    """Partial implementation missing most abstract methods."""

    def list_runtimes(self) -> list[types.Runtime]:
        return []

    def get_runtime(self, name: str) -> types.Runtime:
        return types.Runtime(name=name)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="direct instantiation of RuntimeBackend raises TypeError",
            expected_status=FAILED,
            config={"cls": RuntimeBackend},
            expected_error=TypeError,
        ),
        TestCase(
            name="complete subclass instantiates successfully",
            expected_status=SUCCESS,
            config={"cls": CompleteBackend},
            expected_output=CompleteBackend,
        ),
        TestCase(
            name="partial subclass raises TypeError",
            expected_status=FAILED,
            config={"cls": PartialBackend},
            expected_error=TypeError,
        ),
    ],
)
def test_runtime_backend_instantiation(test_case):
    """Test RuntimeBackend instantiation rules for abstract base class."""
    print("Executing test:", test_case.name)
    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            test_case.config["cls"]()
    else:
        instance = test_case.config["cls"]()
        assert isinstance(instance, test_case.expected_output)
    print("test execution complete")
