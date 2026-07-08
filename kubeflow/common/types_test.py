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

"""Unit tests for kubeflow.common.types module."""

from kubernetes import client
from pydantic import ValidationError
import pytest

from kubeflow.common.test.common import FAILED, SUCCESS, TestCase
from kubeflow.common.types import KubernetesBackendConfig


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default instantiation with all fields none",
            expected_status=SUCCESS,
            config={},
            expected_output={
                "namespace": None,
                "config_file": None,
                "context": None,
                "client_configuration": None,
            },
        ),
        TestCase(
            name="custom namespace",
            expected_status=SUCCESS,
            config={"namespace": "my-namespace"},
            expected_output={"namespace": "my-namespace"},
        ),
        TestCase(
            name="custom config file",
            expected_status=SUCCESS,
            config={"config_file": "/home/user/.kube/config"},
            expected_output={"config_file": "/home/user/.kube/config"},
        ),
        TestCase(
            name="custom context",
            expected_status=SUCCESS,
            config={"context": "minikube"},
            expected_output={"context": "minikube"},
        ),
        TestCase(
            name="all string fields set",
            expected_status=SUCCESS,
            config={
                "namespace": "prod",
                "config_file": "/etc/kube/config",
                "context": "prod-cluster",
            },
            expected_output={
                "namespace": "prod",
                "config_file": "/etc/kube/config",
                "context": "prod-cluster",
            },
        ),
    ],
)
def test_kubernetes_backend_config(test_case: TestCase):
    """Test KubernetesBackendConfig instantiation and field values."""
    print("Executing test:", test_case.name)
    cfg = KubernetesBackendConfig(**test_case.config)

    assert test_case.expected_status == SUCCESS
    for key, expected_val in test_case.expected_output.items():
        assert getattr(cfg, key) == expected_val
    print("test execution complete")


def test_kubernetes_backend_config_client_configuration():
    """Test KubernetesBackendConfig accepts arbitrary client.Configuration type."""
    print("Executing test: client configuration with arbitrary type")
    k8s_config = client.Configuration()
    cfg = KubernetesBackendConfig(client_configuration=k8s_config)

    assert isinstance(cfg.client_configuration, client.Configuration)
    assert cfg.client_configuration is k8s_config
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="invalid client configuration type raises validation error",
            expected_status=FAILED,
            config={"client_configuration": "not-a-config-object"},
            expected_error=ValidationError,
        ),
    ],
)
def test_kubernetes_backend_config_failure(test_case: TestCase):
    """Test KubernetesBackendConfig rejects invalid input."""
    print("Executing test:", test_case.name)
    with pytest.raises(test_case.expected_error):
        KubernetesBackendConfig(**test_case.config)
    print("test execution complete")
