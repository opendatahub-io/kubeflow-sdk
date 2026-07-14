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

"""Unit tests for kubeflow.common.utils module."""

from unittest.mock import mock_open, patch

import pytest

from kubeflow.common import constants
from kubeflow.common.test.common import SUCCESS, TestCase
from kubeflow.common.utils import get_default_target_namespace, is_running_in_k8s

# --------------------------
# is_running_in_k8s tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="running inside kubernetes",
            expected_status=SUCCESS,
            config={"isdir_return": True},
            expected_output=True,
        ),
        TestCase(
            name="running outside kubernetes",
            expected_status=SUCCESS,
            config={"isdir_return": False},
            expected_output=False,
        ),
    ],
)
def test_is_running_in_k8s(test_case: TestCase):
    """Test is_running_in_k8s detects in-cluster environment via SA directory."""
    print("Executing test:", test_case.name)
    with patch(
        "kubeflow.common.utils.os.path.isdir",
        return_value=test_case.config["isdir_return"],
    ):
        result = is_running_in_k8s()
    assert test_case.expected_status == SUCCESS
    assert result is test_case.expected_output
    print("test execution complete")


# --------------------------
# get_default_target_namespace tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="not in k8s with specific context found",
            expected_status=SUCCESS,
            config={
                "in_k8s": False,
                "context": "staging",
                "all_contexts": [
                    {"name": "staging", "context": {"namespace": "staging-ns"}},
                    {"name": "prod", "context": {"namespace": "prod-ns"}},
                ],
                "current_context": {"context": {"namespace": "current-ns"}},
            },
            expected_output="staging-ns",
        ),
        TestCase(
            name="not in k8s with specific context not found falls back to current",
            expected_status=SUCCESS,
            config={
                "in_k8s": False,
                "context": "nonexistent",
                "all_contexts": [
                    {"name": "staging", "context": {"namespace": "staging-ns"}},
                ],
                "current_context": {"context": {"namespace": "current-ns"}},
            },
            expected_output="current-ns",
        ),
        TestCase(
            name="not in k8s without context uses current context",
            expected_status=SUCCESS,
            config={
                "in_k8s": False,
                "context": None,
                "all_contexts": [],
                "current_context": {"context": {"namespace": "default-ns"}},
            },
            expected_output="default-ns",
        ),
        TestCase(
            name="not in k8s with kubeconfig exception falls back to default",
            expected_status=SUCCESS,
            config={
                "in_k8s": False,
                "context": None,
                "raise_exception": True,
            },
            expected_output=constants.DEFAULT_NAMESPACE,
        ),
        TestCase(
            name="in k8s reads namespace from service account",
            expected_status=SUCCESS,
            config={
                "in_k8s": True,
                "sa_namespace": "kube-system\n",
            },
            expected_output="kube-system",
        ),
    ],
)
def test_get_default_target_namespace(test_case: TestCase):
    """Test get_default_target_namespace resolves namespace from various sources."""
    print("Executing test:", test_case.name)

    with patch(
        "kubeflow.common.utils.is_running_in_k8s",
        return_value=test_case.config["in_k8s"],
    ):
        if test_case.config["in_k8s"]:
            sa_ns = test_case.config["sa_namespace"]
            with patch("builtins.open", mock_open(read_data=sa_ns)):
                result = get_default_target_namespace(test_case.config.get("context"))
        elif test_case.config.get("raise_exception"):
            with patch(
                "kubeflow.common.utils.config.list_kube_config_contexts",
                side_effect=Exception("kubeconfig not found"),
            ):
                result = get_default_target_namespace(test_case.config["context"])
        else:
            with patch(
                "kubeflow.common.utils.config.list_kube_config_contexts",
                return_value=(
                    test_case.config["all_contexts"],
                    test_case.config["current_context"],
                ),
            ):
                result = get_default_target_namespace(test_case.config["context"])

    assert test_case.expected_status == SUCCESS
    assert result == test_case.expected_output
    print("test execution complete")
