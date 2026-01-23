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

from unittest.mock import MagicMock

import pytest

from kubeflow.trainer.rhai.constants import (
    CHECKPOINT_MOUNT_PATH,
    CHECKPOINT_VOLUME_NAME,
    S3_CREDENTIAL_KEYS,
)
from kubeflow.trainer.rhai.utils import (
    get_s3_credential_env_vars,
    parse_output_dir_uri,
    validate_secret_exists,
)
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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="s3 uri - returns staging path and ephemeral volume specs",
            expected_status=SUCCESS,
            config={"output_dir": "s3://my-bucket/checkpoints"},
        ),
        TestCase(
            name="s3 uri with nested path - returns staging path",
            expected_status=SUCCESS,
            config={"output_dir": "s3://bucket/models/llm/v1"},
        ),
    ],
)
def test_parse_output_dir_uri_with_s3(test_case):
    """Test parse_output_dir_uri correctly parses S3 URIs and returns ephemeral volume specs."""
    print(f"Executing test: {test_case.name}")

    resolved_path, volume_specs = parse_output_dir_uri(test_case.config["output_dir"])

    assert test_case.expected_status == SUCCESS

    # S3 URIs return local staging path (training writes here before uploading to S3)
    assert resolved_path == CHECKPOINT_MOUNT_PATH

    # Verify volume specs structure
    assert volume_specs is not None
    assert "volume" in volume_specs
    assert "volumeMount" in volume_specs

    # Verify ephemeral volume spec
    volume_spec = volume_specs["volume"]
    assert volume_spec["name"] == CHECKPOINT_VOLUME_NAME
    assert "ephemeral" in volume_spec
    assert "volumeClaimTemplate" in volume_spec["ephemeral"]

    # Verify volumeClaimTemplate spec
    vct_spec = volume_spec["ephemeral"]["volumeClaimTemplate"]["spec"]
    assert vct_spec["accessModes"] == ["ReadWriteOnce"]
    assert "resources" in vct_spec
    # Note: storageClassName is not set - uses cluster default

    # Verify volumeMount spec
    volume_mount_spec = volume_specs["volumeMount"]
    assert volume_mount_spec["name"] == CHECKPOINT_VOLUME_NAME
    assert volume_mount_spec["mountPath"] == CHECKPOINT_MOUNT_PATH

    print("test execution complete")


def test_get_s3_credential_env_vars():
    """Test get_s3_credential_env_vars returns EnvVar objects for all S3 credential keys."""
    print("Executing test: get_s3_credential_env_vars")

    secret_name = "my-s3-secret"
    env_vars = get_s3_credential_env_vars(secret_name)

    assert len(env_vars) == len(S3_CREDENTIAL_KEYS)

    for env_var, expected_key in zip(env_vars, S3_CREDENTIAL_KEYS):
        assert env_var.name == expected_key
        assert env_var.value_from is not None
        assert env_var.value_from.secret_key_ref is not None
        assert env_var.value_from.secret_key_ref.name == secret_name
        assert env_var.value_from.secret_key_ref.key == expected_key
        assert env_var.value_from.secret_key_ref.optional is True

    print("test execution complete")


def test_validate_secret_exists_success():
    """Test validate_secret_exists passes when secret exists."""
    print("Executing test: validate_secret_exists_success")

    mock_core_api = MagicMock()
    mock_core_api.read_namespaced_secret.return_value = MagicMock()

    # Should not raise
    validate_secret_exists(mock_core_api, "my-secret", "default")

    mock_core_api.read_namespaced_secret.assert_called_once_with(
        name="my-secret", namespace="default"
    )

    print("test execution complete")


def test_validate_secret_exists_not_found():
    """Test validate_secret_exists raises ValueError when secret not found."""
    print("Executing test: validate_secret_exists_not_found")

    from kubernetes.client.rest import ApiException

    mock_core_api = MagicMock()
    mock_core_api.read_namespaced_secret.side_effect = ApiException(status=404)

    with pytest.raises(ValueError) as exc_info:
        validate_secret_exists(mock_core_api, "missing-secret", "test-ns")

    assert "missing-secret" in str(exc_info.value)
    assert "test-ns" in str(exc_info.value)
    assert "not found" in str(exc_info.value)

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
