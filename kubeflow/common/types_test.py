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

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


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
        TestCase(
            name="client configuration with arbitrary type",
            expected_status=SUCCESS,
            config={"use_client_config": True},
            expected_output={"has_client_config": True},
        ),
        TestCase(
            name="invalid client configuration type raises validation error",
            expected_status=FAILED,
            config={"client_configuration": "not-a-config-object"},
            expected_error=ValidationError,
        ),
    ],
)
def test_kubernetes_backend_config(test_case: TestCase):
    """Test KubernetesBackendConfig instantiation and field values."""
    print("Executing test:", test_case.name)
    try:
        config_kwargs = {k: v for k, v in test_case.config.items() if k != "use_client_config"}
        if test_case.config.get("use_client_config"):
            config_kwargs["client_configuration"] = client.Configuration()

        cfg = KubernetesBackendConfig(**config_kwargs)

        assert test_case.expected_status == SUCCESS, (
            f"Expected exception but none was raised for {test_case.name}"
        )

        for key, expected_val in test_case.expected_output.items():
            if key == "has_client_config":
                assert isinstance(cfg.client_configuration, client.Configuration)
            else:
                assert getattr(cfg, key) == expected_val

    except Exception as e:
        assert test_case.expected_status == FAILED, f"Unexpected exception in {test_case.name}: {e}"
        if test_case.expected_error:
            assert type(e) is test_case.expected_error
    print("test execution complete")
