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

"""Tests for RHAI utils functions."""

import pytest

from kubeflow.trainer.rhai.constants import CHECKPOINT_MOUNT_PATH, CHECKPOINT_VOLUME_NAME
from kubeflow.trainer.rhai.utils import parse_output_dir_uri
from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="output_dir is None - returns None, None",
            expected_status=SUCCESS,
            config={"output_dir": None},
            expected_output={"resolved_path": None, "volume_specs": None},
        ),
        TestCase(
            name="local path - returns path unchanged with no volume specs",
            expected_status=SUCCESS,
            config={"output_dir": "/tmp/checkpoints"},
            expected_output={"resolved_path": "/tmp/checkpoints", "volume_specs": None},
        ),
        TestCase(
            name="absolute path - returns path unchanged with no volume specs",
            expected_status=SUCCESS,
            config={"output_dir": "/mnt/storage/checkpoints"},
            expected_output={"resolved_path": "/mnt/storage/checkpoints", "volume_specs": None},
        ),
        TestCase(
            name="relative path - returns path unchanged with no volume specs",
            expected_status=SUCCESS,
            config={"output_dir": "checkpoints/model-v1"},
            expected_output={"resolved_path": "checkpoints/model-v1", "volume_specs": None},
        ),
    ],
)
def test_parse_output_dir_uri_happy_path(test_case):
    """Test parse_output_dir_uri happy paths for None and regular filesystem paths."""
    print(f"Executing test: {test_case.name}")

    resolved_path, volume_specs = parse_output_dir_uri(test_case.config["output_dir"])

    assert test_case.expected_status == SUCCESS
    assert resolved_path == test_case.expected_output["resolved_path"]
    assert volume_specs == test_case.expected_output["volume_specs"]

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="pvc uri with subpath - resolves path and creates volume specs",
            expected_status=SUCCESS,
            config={
                "output_dir": "pvc://my-pvc/checkpoints",
                "pvc_name": "my-pvc",
                "checkpoint_subpath": "checkpoints",
            },
        ),
        TestCase(
            name="pvc uri without subpath - uses mount path directly",
            expected_status=SUCCESS,
            config={
                "output_dir": "pvc://my-pvc",
                "pvc_name": "my-pvc",
                "checkpoint_subpath": "",
            },
        ),
        TestCase(
            name="pvc uri with deep subpath - resolves correctly",
            expected_status=SUCCESS,
            config={
                "output_dir": "pvc://storage-pvc/models/llama/checkpoints",
                "pvc_name": "storage-pvc",
                "checkpoint_subpath": "models/llama/checkpoints",
            },
        ),
    ],
)
def test_parse_output_dir_uri_with_pvc(test_case):
    """Test parse_output_dir_uri correctly parses PVC URIs and returns volume specs."""
    print(f"Executing test: {test_case.name}")

    output_dir = test_case.config["output_dir"]
    pvc_name = test_case.config["pvc_name"]
    checkpoint_subpath = test_case.config["checkpoint_subpath"]

    resolved_path, volume_specs = parse_output_dir_uri(output_dir)

    assert test_case.expected_status == SUCCESS

    # Verify resolved path
    if checkpoint_subpath:
        expected_path = f"{CHECKPOINT_MOUNT_PATH}/{checkpoint_subpath}"
    else:
        expected_path = CHECKPOINT_MOUNT_PATH
    assert resolved_path == expected_path

    # Verify volume specs structure
    assert volume_specs is not None
    assert "volume" in volume_specs
    assert "volumeMount" in volume_specs

    # Verify volume spec
    volume_spec = volume_specs["volume"]
    assert volume_spec["name"] == CHECKPOINT_VOLUME_NAME
    assert volume_spec["persistentVolumeClaim"]["claimName"] == pvc_name

    # Verify volumeMount spec
    volume_mount_spec = volume_specs["volumeMount"]
    assert volume_mount_spec["name"] == CHECKPOINT_VOLUME_NAME
    assert volume_mount_spec["mountPath"] == CHECKPOINT_MOUNT_PATH
    assert volume_mount_spec["readOnly"] is False

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
