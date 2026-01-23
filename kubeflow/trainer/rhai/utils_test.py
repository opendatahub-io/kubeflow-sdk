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
)
from kubeflow.trainer.rhai.utils import (
    get_s3_credential_env_vars,
    inject_s3_credentials,
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

    # Expected volume specs structure (storageClassName not set - uses cluster default)
    expected_volume_specs = {
        "volume": {
            "name": CHECKPOINT_VOLUME_NAME,
            "ephemeral": {
                "volumeClaimTemplate": {
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "resources": {"requests": {"storage": "50Gi"}},
                    }
                }
            },
        },
        "volumeMount": {
            "name": CHECKPOINT_VOLUME_NAME,
            "mountPath": CHECKPOINT_MOUNT_PATH,
            "readOnly": False,
        },
    }

    # Convert Quantity object to string for comparison (access actual_instance)
    actual_storage = volume_specs["volume"]["ephemeral"]["volumeClaimTemplate"]["spec"][
        "resources"
    ]["requests"]["storage"]
    volume_specs["volume"]["ephemeral"]["volumeClaimTemplate"]["spec"]["resources"]["requests"][
        "storage"
    ] = actual_storage.actual_instance

    assert volume_specs == expected_volume_specs

    print("test execution complete")


def test_get_s3_credential_env_vars():
    """Test get_s3_credential_env_vars returns EnvVar objects for all keys in secret."""
    print("Executing test: get_s3_credential_env_vars")

    # Mock the CoreV1Api and secret
    mock_core_api = MagicMock()
    mock_secret = MagicMock()
    mock_secret.data = {
        "AWS_ACCESS_KEY_ID": "base64encodedvalue1",
        "AWS_SECRET_ACCESS_KEY": "base64encodedvalue2",
        "AWS_DEFAULT_REGION": "base64encodedvalue3",
        "AWS_S3_ENDPOINT": "base64encodedvalue4",
        "AWS_SESSION_TOKEN": "base64encodedvalue5",  # Extra key to test dynamic fetching
    }
    mock_core_api.read_namespaced_secret.return_value = mock_secret

    secret_name = "my-s3-secret"
    namespace = "default"
    env_vars = get_s3_credential_env_vars(mock_core_api, secret_name, namespace)

    # Verify the secret was read
    mock_core_api.read_namespaced_secret.assert_called_once_with(
        name=secret_name, namespace=namespace
    )

    # Convert to list of dicts for easier comparison
    actual = [
        {
            "name": env.name,
            "secretName": env.value_from.secret_key_ref.name,
            "secretKey": env.value_from.secret_key_ref.key,
        }
        for env in env_vars
    ]

    expected = [
        {"name": key, "secretName": secret_name, "secretKey": key} for key in mock_secret.data
    ]

    assert actual == expected
    assert len(env_vars) == 5  # All 5 keys from the mock secret

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


def test_inject_s3_credentials_with_s3_output_dir():
    """Test inject_s3_credentials injects credentials when using S3 output_dir."""
    print("Executing test: inject_s3_credentials_with_s3_output_dir")

    mock_core_api = MagicMock()
    mock_secret = MagicMock()
    mock_secret.data = {
        "AWS_ACCESS_KEY_ID": "key1",
        "AWS_SECRET_ACCESS_KEY": "key2",
        "AWS_DEFAULT_REGION": "key3",
        "AWS_S3_ENDPOINT": "key4",
    }
    mock_core_api.read_namespaced_secret.return_value = mock_secret

    # Create a mock trainer with S3 output_dir
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "s3://my-bucket/checkpoints"
    mock_trainer.data_connection_name = "my-s3-secret"

    # Create a mock trainer_cr with no existing env vars
    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = None

    result = inject_s3_credentials(mock_trainer, mock_trainer_cr, mock_core_api, "default")

    # Should read the secret twice (once for validation, once for getting keys)
    assert mock_core_api.read_namespaced_secret.call_count == 2
    mock_core_api.read_namespaced_secret.assert_called_with(
        name="my-s3-secret", namespace="default"
    )

    # Should add all keys from the secret as env vars
    assert result.env is not None
    assert len(result.env) == len(mock_secret.data)

    print("test execution complete")


def test_inject_s3_credentials_without_s3_output_dir():
    """Test inject_s3_credentials does nothing when not using S3 output_dir."""
    print("Executing test: inject_s3_credentials_without_s3_output_dir")

    mock_core_api = MagicMock()

    # Create a mock trainer with PVC output_dir
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "pvc://my-pvc/checkpoints"
    mock_trainer.data_connection_name = None

    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = []

    result = inject_s3_credentials(mock_trainer, mock_trainer_cr, mock_core_api, "default")

    # Should NOT call validate_secret_exists
    mock_core_api.read_namespaced_secret.assert_not_called()

    # Should return trainer_cr unchanged
    assert result.env == []

    print("test execution complete")


def test_inject_s3_credentials_without_data_connection_name():
    """Test inject_s3_credentials does nothing when data_connection_name is missing."""
    print("Executing test: inject_s3_credentials_without_data_connection_name")

    mock_core_api = MagicMock()

    # Create a mock trainer with S3 output_dir but no data_connection_name
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "s3://my-bucket/checkpoints"
    mock_trainer.data_connection_name = None

    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = []

    result = inject_s3_credentials(mock_trainer, mock_trainer_cr, mock_core_api, "default")

    # Should NOT call validate_secret_exists
    mock_core_api.read_namespaced_secret.assert_not_called()

    # Should return trainer_cr unchanged
    assert result.env == []

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
