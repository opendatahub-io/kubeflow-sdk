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

"""Unit tests for Container backend Pydantic models."""

from pydantic import ValidationError
import pytest

from kubeflow.trainer.backends.container.types import (
    ContainerBackendConfig,
    TrainingRuntimeSource,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default ContainerBackendConfig values",
            expected_status=SUCCESS,
            config={},
            expected_output={
                "pull_policy": "IfNotPresent",
                "auto_remove": True,
                "container_host": None,
                "container_runtime": None,
                "dataset_initializer_image": (
                    "ghcr.io/kubeflow/trainer/dataset-initializer:latest"
                ),
                "model_initializer_image": ("ghcr.io/kubeflow/trainer/model-initializer:latest"),
                "initializer_timeout": 600,
            },
        ),
        TestCase(
            name="custom ContainerBackendConfig values",
            expected_status=SUCCESS,
            config={
                "pull_policy": "Always",
                "auto_remove": False,
                "container_host": "tcp://192.168.1.100:2375",
                "container_runtime": "docker",
                "initializer_timeout": 300,
            },
            expected_output={
                "pull_policy": "Always",
                "auto_remove": False,
                "container_host": "tcp://192.168.1.100:2375",
                "container_runtime": "docker",
                "initializer_timeout": 300,
            },
        ),
        TestCase(
            name="ContainerBackendConfig with docker runtime",
            expected_status=SUCCESS,
            config={"container_runtime": "docker"},
            expected_output={"container_runtime": "docker"},
        ),
        TestCase(
            name="ContainerBackendConfig with podman runtime",
            expected_status=SUCCESS,
            config={"container_runtime": "podman"},
            expected_output={"container_runtime": "podman"},
        ),
        TestCase(
            name="invalid container_runtime raises ValidationError",
            expected_status=FAILED,
            config={"container_runtime": "containerd"},
            expected_error=ValidationError,
        ),
    ],
)
def test_container_backend_config(test_case: TestCase):
    """Test ContainerBackendConfig model creation and validation."""
    print(f"Executing test: {test_case.name}")

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            ContainerBackendConfig(**test_case.config)
    else:
        cfg = ContainerBackendConfig(**test_case.config)
        for key, expected_val in test_case.expected_output.items():
            assert getattr(cfg, key) == expected_val, (
                f"{key}: expected {expected_val}, got {getattr(cfg, key)}"
            )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="TrainingRuntimeSource defaults",
            expected_status=SUCCESS,
            config={},
            expected_output=["github://kubeflow/trainer"],
        ),
        TestCase(
            name="TrainingRuntimeSource custom sources",
            expected_status=SUCCESS,
            config={
                "sources": [
                    "file:///opt/runtimes",
                    "https://example.com/runtimes",
                ],
            },
            expected_output=[
                "file:///opt/runtimes",
                "https://example.com/runtimes",
            ],
        ),
    ],
)
def test_training_runtime_source(test_case: TestCase):
    """Test TrainingRuntimeSource model creation."""
    print(f"Executing test: {test_case.name}")

    src = TrainingRuntimeSource(**test_case.config)
    assert src.sources == test_case.expected_output

    print("test execution complete")
