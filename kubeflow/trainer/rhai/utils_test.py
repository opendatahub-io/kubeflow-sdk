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

import os
from unittest.mock import MagicMock, patch

from kubernetes.client.rest import ApiException
import pytest

from kubeflow.trainer.rhai.constants import (
    CHECKPOINT_MOUNT_PATH,
    CHECKPOINT_VOLUME_NAME,
)
from kubeflow.trainer.rhai.utils import (
    get_cloud_storage_credential_env_vars,
    inject_cloud_storage_credentials,
    is_primary_pod,
    parse_output_dir_uri,
    setup_rhai_trainer_storage,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


def test_is_primary_pod_with_job_completion_index_0():
    """Test is_primary_pod returns True when JOB_COMPLETION_INDEX is 0."""
    print("Executing test: is_primary_pod_with_job_completion_index_0")

    with patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "0"}, clear=True):
        assert is_primary_pod() is True

    print("test execution complete")


def test_is_primary_pod_with_job_completion_index_1():
    """Test is_primary_pod returns False when JOB_COMPLETION_INDEX is 1."""
    print("Executing test: is_primary_pod_with_job_completion_index_1")

    with patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "1"}, clear=True):
        assert is_primary_pod() is False

    print("test execution complete")


def test_is_primary_pod_with_pet_node_rank_0():
    """Test is_primary_pod returns True when PET_NODE_RANK is 0."""
    print("Executing test: is_primary_pod_with_pet_node_rank_0")

    with patch.dict(os.environ, {"PET_NODE_RANK": "0"}, clear=True):
        assert is_primary_pod() is True

    print("test execution complete")


def test_is_primary_pod_with_pet_node_rank_1():
    """Test is_primary_pod returns False when PET_NODE_RANK is 1."""
    print("Executing test: is_primary_pod_with_pet_node_rank_1")

    with patch.dict(os.environ, {"PET_NODE_RANK": "1"}, clear=True):
        assert is_primary_pod() is False

    print("test execution complete")


def test_is_primary_pod_job_completion_index_takes_precedence():
    """Test JOB_COMPLETION_INDEX takes precedence over PET_NODE_RANK."""
    print("Executing test: is_primary_pod_job_completion_index_takes_precedence")

    # JOB_COMPLETION_INDEX=0, PET_NODE_RANK=1 -> should return True (JOB_COMPLETION_INDEX wins)
    with patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "0", "PET_NODE_RANK": "1"}, clear=True):
        assert is_primary_pod() is True

    # JOB_COMPLETION_INDEX=1, PET_NODE_RANK=0 -> should return False (JOB_COMPLETION_INDEX wins)
    with patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "1", "PET_NODE_RANK": "0"}, clear=True):
        assert is_primary_pod() is False

    print("test execution complete")


def test_is_primary_pod_no_environment_variables():
    """Test is_primary_pod returns False when neither env var is set (conservative)."""
    print("Executing test: is_primary_pod_no_environment_variables")

    with patch.dict(os.environ, {}, clear=True):
        assert is_primary_pod() is False

    print("test execution complete")


def test_is_primary_pod_with_non_numeric_job_completion_index():
    """Test is_primary_pod handles non-0 string values correctly."""
    print("Executing test: is_primary_pod_with_non_numeric_job_completion_index")

    # String "0" should match (primary)
    with patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "0"}, clear=True):
        assert is_primary_pod() is True

    # String "00" should NOT match (not exactly "0")
    with patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "00"}, clear=True):
        assert is_primary_pod() is False

    # String "2" should NOT match
    with patch.dict(os.environ, {"JOB_COMPLETION_INDEX": "2"}, clear=True):
        assert is_primary_pod() is False

    print("test execution complete")


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
            name="s3 uri - returns staging path and emptyDir volume specs",
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
    """Test parse_output_dir_uri correctly parses S3 URIs and returns emptyDir volume specs."""
    print(f"Executing test: {test_case.name}")

    resolved_path, volume_specs = parse_output_dir_uri(test_case.config["output_dir"])

    assert test_case.expected_status == SUCCESS

    # S3 URIs return local staging path (training writes here before uploading to S3)
    assert resolved_path == CHECKPOINT_MOUNT_PATH

    # Expected volume specs structure (using emptyDir for temporary staging)
    expected_volume_specs = {
        "volume": {
            "name": CHECKPOINT_VOLUME_NAME,
            "emptyDir": {},
        },
        "volumeMount": {
            "name": CHECKPOINT_VOLUME_NAME,
            "mountPath": CHECKPOINT_MOUNT_PATH,
            "readOnly": False,
        },
    }

    assert volume_specs == expected_volume_specs

    print("test execution complete")


def test_get_cloud_storage_credential_env_vars():
    """Test get_cloud_storage_credential_env_vars returns EnvVar objects for all keys in secret."""
    print("Executing test: get_cloud_storage_credential_env_vars")

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

    data_connection_name = "my-s3-secret"
    namespace = "default"
    env_vars = get_cloud_storage_credential_env_vars(mock_core_api, data_connection_name, namespace)

    # Verify the secret was read
    mock_core_api.read_namespaced_secret.assert_called_once_with(
        name=data_connection_name, namespace=namespace
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
        {"name": key, "secretName": data_connection_name, "secretKey": key}
        for key in mock_secret.data
    ]

    assert actual == expected
    assert len(env_vars) == 5  # All 5 keys from the mock secret

    print("test execution complete")


def test_get_cloud_storage_credential_env_vars_secret_not_found():
    """Test get_cloud_storage_credential_env_vars raises ValueError when secret not found."""
    print("Executing test: get_cloud_storage_credential_env_vars_secret_not_found")

    mock_core_api = MagicMock()
    mock_core_api.read_namespaced_secret.side_effect = ApiException(status=404)

    with pytest.raises(ValueError) as exc_info:
        get_cloud_storage_credential_env_vars(mock_core_api, "missing-secret", "test-ns")

    assert "missing-secret" in str(exc_info.value)
    assert "test-ns" in str(exc_info.value)
    assert "not found" in str(exc_info.value)
    assert "Data Connection" in str(exc_info.value)

    print("test execution complete")


def test_get_cloud_storage_credential_env_vars_permission_denied():
    """Test get_cloud_storage_credential_env_vars raises ValueError on RBAC error."""
    print("Executing test: get_cloud_storage_credential_env_vars_permission_denied")

    mock_core_api = MagicMock()
    mock_core_api.read_namespaced_secret.side_effect = ApiException(status=403)

    with pytest.raises(ValueError) as exc_info:
        get_cloud_storage_credential_env_vars(mock_core_api, "my-secret", "test-ns")

    assert "my-secret" in str(exc_info.value)
    assert "test-ns" in str(exc_info.value)
    assert "permission denied" in str(exc_info.value)
    assert "Data Connection" in str(exc_info.value)

    print("test execution complete")


def test_inject_cloud_storage_credentials_with_s3_output_dir():
    """Test inject_cloud_storage_credentials injects credentials when using S3 output_dir."""
    print("Executing test: inject_cloud_storage_credentials_with_s3_output_dir")

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

    result = inject_cloud_storage_credentials(
        mock_trainer, mock_trainer_cr, mock_core_api, "default"
    )

    # Verify API calls
    mock_core_api.read_namespaced_secret.assert_called_with(
        name="my-s3-secret", namespace="default"
    )

    # Verify all secret keys are added as env vars
    actual_env = [
        {"name": env.name, "secretKey": env.value_from.secret_key_ref.key} for env in result.env
    ]
    expected_env = [{"name": key, "secretKey": key} for key in mock_secret.data]
    assert actual_env == expected_env

    print("test execution complete")


def test_inject_cloud_storage_credentials_without_s3_output_dir():
    """Test inject_cloud_storage_credentials does nothing when not using S3 output_dir."""
    print("Executing test: inject_cloud_storage_credentials_without_s3_output_dir")

    mock_core_api = MagicMock()

    # Create a mock trainer with PVC output_dir
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "pvc://my-pvc/checkpoints"
    mock_trainer.data_connection_name = None

    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = []

    result = inject_cloud_storage_credentials(
        mock_trainer, mock_trainer_cr, mock_core_api, "default"
    )

    # Should NOT read any secrets
    mock_core_api.read_namespaced_secret.assert_not_called()

    # Should return trainer_cr unchanged
    assert result.env == []

    print("test execution complete")


def test_inject_cloud_storage_credentials_without_data_connection_name():
    """Test inject_cloud_storage_credentials does nothing when data_connection_name is missing."""
    print("Executing test: inject_cloud_storage_credentials_without_data_connection_name")

    mock_core_api = MagicMock()

    # Create a mock trainer with S3 output_dir but no data_connection_name
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "s3://my-bucket/checkpoints"
    mock_trainer.data_connection_name = None

    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = []

    result = inject_cloud_storage_credentials(
        mock_trainer, mock_trainer_cr, mock_core_api, "default"
    )

    # Should NOT read any secrets
    mock_core_api.read_namespaced_secret.assert_not_called()

    # Should return trainer_cr unchanged
    assert result.env == []

    print("test execution complete")


def test_inject_cloud_storage_credentials_duplicate_env_var():
    """Test inject_cloud_storage_credentials raises error on duplicate env var names."""
    print("Executing test: inject_cloud_storage_credentials_duplicate_env_var")

    mock_core_api = MagicMock()
    mock_secret = MagicMock()
    mock_secret.data = {
        "AWS_ACCESS_KEY_ID": "key1",
        "AWS_SECRET_ACCESS_KEY": "key2",
    }
    mock_core_api.read_namespaced_secret.return_value = mock_secret

    # Create a mock trainer with S3 output_dir
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "s3://my-bucket/checkpoints"
    mock_trainer.data_connection_name = "my-s3-secret"

    # Create a mock trainer_cr with existing env var that conflicts
    existing_env = MagicMock()
    existing_env.name = "AWS_ACCESS_KEY_ID"  # Same name as in secret
    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = [existing_env]

    with pytest.raises(ValueError, match=r"AWS_ACCESS_KEY_ID.*conflicts"):
        inject_cloud_storage_credentials(mock_trainer, mock_trainer_cr, mock_core_api, "default")

    print("test execution complete")


def test_setup_rhai_trainer_storage_with_s3():
    """Test setup_rhai_trainer_storage handles S3 output_dir with volume mounts and credentials."""
    print("Executing test: setup_rhai_trainer_storage_with_s3")

    mock_core_api = MagicMock()
    mock_secret = MagicMock()
    mock_secret.data = {
        "AWS_ACCESS_KEY_ID": "key1",
        "AWS_SECRET_ACCESS_KEY": "key2",
    }
    mock_core_api.read_namespaced_secret.return_value = mock_secret

    # Create a mock trainer with S3 output_dir
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "s3://my-bucket/checkpoints"
    mock_trainer.data_connection_name = "my-s3-secret"

    # Create a mock trainer_cr with no existing env vars
    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = None

    resolved_dir, result_cr, result_overrides = setup_rhai_trainer_storage(
        mock_trainer, mock_trainer_cr, None, mock_core_api, "default"
    )

    # Verify expected results
    expected = {
        "resolved_dir": CHECKPOINT_MOUNT_PATH,
        "has_overrides": True,
        "env_count": 2,  # 2 keys in mock secret
    }
    actual = {
        "resolved_dir": resolved_dir,
        "has_overrides": result_overrides is not None and len(result_overrides) > 0,
        "env_count": len(result_cr.env) if result_cr.env else 0,
    }
    assert actual == expected

    print("test execution complete")


def test_setup_rhai_trainer_storage_with_pvc():
    """Test setup_rhai_trainer_storage handles PVC output_dir without S3 credentials."""
    print("Executing test: setup_rhai_trainer_storage_with_pvc")

    mock_core_api = MagicMock()

    # Create a mock trainer with PVC output_dir
    mock_trainer = MagicMock()
    mock_trainer.output_dir = "pvc://my-pvc/checkpoints"
    mock_trainer.data_connection_name = None

    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = []

    resolved_dir, result_cr, result_overrides = setup_rhai_trainer_storage(
        mock_trainer, mock_trainer_cr, None, mock_core_api, "default"
    )

    # Verify expected results
    expected = {
        "resolved_dir": f"{CHECKPOINT_MOUNT_PATH}/checkpoints",
        "has_overrides": True,
        "env": [],
        "secret_api_called": False,
    }
    actual = {
        "resolved_dir": resolved_dir,
        "has_overrides": result_overrides is not None and len(result_overrides) > 0,
        "env": result_cr.env,
        "secret_api_called": mock_core_api.read_namespaced_secret.called,
    }
    assert actual == expected

    print("test execution complete")


def test_setup_rhai_trainer_storage_no_output_dir():
    """Test setup_rhai_trainer_storage handles trainer without output_dir."""
    print("Executing test: setup_rhai_trainer_storage_no_output_dir")

    mock_core_api = MagicMock()

    # Create a mock trainer without output_dir
    mock_trainer = MagicMock(spec=[])  # No output_dir attribute

    mock_trainer_cr = MagicMock()
    mock_trainer_cr.env = []

    resolved_dir, result_cr, result_overrides = setup_rhai_trainer_storage(
        mock_trainer, mock_trainer_cr, None, mock_core_api, "default"
    )

    # Verify expected results
    expected = {
        "resolved_dir": None,
        "overrides": [],
        "secret_api_called": False,
    }
    actual = {
        "resolved_dir": resolved_dir,
        "overrides": result_overrides,
        "secret_api_called": mock_core_api.read_namespaced_secret.called,
    }
    assert actual == expected

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
