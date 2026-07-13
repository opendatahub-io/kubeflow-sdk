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

"""Unit tests for the LocalJob class in the Kubeflow Trainer SDK."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="init with defaults",
            expected_status=SUCCESS,
            config={
                "name": "test-job",
                "command": ["echo", "hello"],
            },
            expected_output={
                "status": constants.TRAINJOB_CREATED,
                "stdout": "",
                "success": False,
                "returncode": None,
                "dependencies": [],
                "env": {},
                "creation_time": None,
                "completion_time": None,
            },
        ),
        TestCase(
            name="init with custom env and dependencies",
            expected_status=SUCCESS,
            config={
                "name": "custom-job",
                "command": "run.sh",
                "env": {"KEY": "VAL"},
                "dependencies": ["dep1"],
            },
            expected_output={
                "env": {"KEY": "VAL"},
                "dependencies": ["dep1"],
            },
        ),
    ],
)
def test_local_job_init(test_case):
    """Test LocalJob initialization with various configurations."""
    print("Executing test:", test_case.name)

    config = test_case.config
    job = LocalJob(
        name=config["name"],
        command=config["command"],
        env=config.get("env"),
        dependencies=config.get("dependencies"),
    )

    expected = test_case.expected_output
    assert job.status == expected.get("status", constants.TRAINJOB_CREATED)
    assert job.env == expected["env"]
    assert job.dependencies == expected["dependencies"]

    if "stdout" in expected:
        assert job.stdout == expected["stdout"]
    if "success" in expected:
        assert job.success == expected["success"]
    if "returncode" in expected:
        assert job.returncode == expected["returncode"]
    if "creation_time" in expected:
        assert job.creation_time == expected["creation_time"]
    if "completion_time" in expected:
        assert job.completion_time == expected["completion_time"]

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="cancel sets the cancel flag",
            expected_status=SUCCESS,
            config={"name": "cancel-job", "command": ["sleep", "10"]},
        ),
    ],
)
def test_local_job_cancel(test_case):
    """Test that cancel sets the internal cancel-requested event."""
    print("Executing test:", test_case.name)

    job = LocalJob(name=test_case.config["name"], command=test_case.config["command"])
    assert not job._cancel_requested.is_set()
    job.cancel()
    assert job._cancel_requested.is_set()

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="logs without follow returns splitlines",
            expected_status=SUCCESS,
            config={
                "name": "logs-job",
                "command": ["echo"],
                "stdout_content": "line1\nline2\nline3",
            },
            expected_output=["line1", "line2", "line3"],
        ),
        TestCase(
            name="logs without follow on empty stdout",
            expected_status=SUCCESS,
            config={
                "name": "empty-logs-job",
                "command": ["echo"],
                "stdout_content": "",
            },
            expected_output=[],
        ),
    ],
)
def test_local_job_logs(test_case):
    """Test logs method returns stdout lines when follow=False."""
    print("Executing test:", test_case.name)

    job = LocalJob(
        name=test_case.config["name"],
        command=test_case.config["command"],
    )
    job._stdout = test_case.config["stdout_content"]

    result = job.logs(follow=False)
    assert result == test_case.expected_output

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="dependency failure skips execution",
            expected_status=FAILED,
            config={
                "name": "dep-fail-job",
                "command": ["echo", "should-not-run"],
            },
            expected_output="Dependency failing-dep failed. Skipping",
        ),
    ],
)
def test_local_job_dependency_failure(test_case):
    """Test that a failed dependency prevents job execution."""
    print("Executing test:", test_case.name)

    failing_dep = Mock()
    failing_dep.name = "failing-dep"
    failing_dep.success = False
    failing_dep.join = Mock()

    job = LocalJob(
        name=test_case.config["name"],
        command=test_case.config["command"],
        dependencies=[failing_dep],
    )
    job.run()

    assert job.stdout == test_case.expected_output
    failing_dep.join.assert_called_once()

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="successful subprocess sets Complete status",
            expected_status=SUCCESS,
            config={
                "name": "success-job",
                "command": ["echo", "hello"],
                "returncode": 0,
                "output_lines": ["hello\n", ""],
            },
            expected_output={
                "status": constants.TRAINJOB_COMPLETE,
                "success": True,
            },
        ),
        TestCase(
            name="failed subprocess sets Failed status",
            expected_status=FAILED,
            config={
                "name": "fail-job",
                "command": ["false"],
                "returncode": 1,
                "output_lines": ["error\n", ""],
            },
            expected_output={
                "status": constants.TRAINJOB_FAILED,
                "success": False,
            },
        ),
    ],
)
def test_local_job_run_subprocess(test_case):
    """Test job run with mocked subprocess for success and failure."""
    print("Executing test:", test_case.name)

    config = test_case.config
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = config["output_lines"]
    mock_process.poll.return_value = config["returncode"]
    mock_process.wait.return_value = config["returncode"]
    mock_process.returncode = config["returncode"]
    mock_process.stdout.close = Mock()

    with patch("subprocess.Popen", return_value=mock_process), patch("os.chdir"):
        job = LocalJob(name=config["name"], command=config["command"])
        job.run()

    expected = test_case.expected_output
    assert job.status == expected["status"]
    assert job.success == expected["success"]
    assert job.creation_time is not None
    assert job.completion_time is not None

    print("test execution complete")
