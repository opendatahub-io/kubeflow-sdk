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

"""Tests for RHAI constants."""

import pytest

from kubeflow.trainer.rhai.constants import (
    ANNOTATION_METRICS_POLL_INTERVAL,
    ANNOTATION_METRICS_PORT,
    ANNOTATION_PROGRESSION_TRACKING,
    ANNOTATION_TRAINER_STATUS,
    CHECKPOINT_EPHEMERAL_VOLUME_SIZE,
    CHECKPOINT_INCOMPLETE_MARKER,
    CHECKPOINT_MOUNT_PATH,
    CHECKPOINT_STAGING_DIR,
    CHECKPOINT_VOLUME_NAME,
    PVC_URI_SCHEME,
    S3_URI_SCHEME,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="annotation_progression_tracking uses opendatahub.io domain",
            expected_status=SUCCESS,
            config={"value": ANNOTATION_PROGRESSION_TRACKING},
            expected_output="trainer.opendatahub.io/",
        ),
        TestCase(
            name="annotation_metrics_port uses opendatahub.io domain",
            expected_status=SUCCESS,
            config={"value": ANNOTATION_METRICS_PORT},
            expected_output="trainer.opendatahub.io/",
        ),
        TestCase(
            name="annotation_metrics_poll_interval uses opendatahub.io domain",
            expected_status=SUCCESS,
            config={"value": ANNOTATION_METRICS_POLL_INTERVAL},
            expected_output="trainer.opendatahub.io/",
        ),
        TestCase(
            name="annotation_trainer_status uses opendatahub.io domain",
            expected_status=SUCCESS,
            config={"value": ANNOTATION_TRAINER_STATUS},
            expected_output="trainer.opendatahub.io/",
        ),
    ],
)
def test_annotations_use_opendatahub_domain(test_case: TestCase) -> None:
    """All RHAI annotations must use the trainer.opendatahub.io domain."""
    print(f"Executing test: {test_case.name}")
    assert test_case.config["value"].startswith(test_case.expected_output)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="s3 URI scheme is s3://",
            expected_status=SUCCESS,
            config={"value": S3_URI_SCHEME},
            expected_output="s3://",
        ),
        TestCase(
            name="pvc URI scheme is pvc://",
            expected_status=SUCCESS,
            config={"value": PVC_URI_SCHEME},
            expected_output="pvc://",
        ),
    ],
)
def test_uri_schemes_end_with_double_slash(test_case: TestCase) -> None:
    """URI schemes must end with :// to be valid prefixes for path construction."""
    print(f"Executing test: {test_case.name}")
    assert test_case.config["value"] == test_case.expected_output
    assert test_case.config["value"].endswith("://")


def test_checkpoint_mount_path_is_absolute() -> None:
    """Checkpoint mount path must be absolute for Kubernetes volume mounts."""
    print("Executing test: checkpoint_mount_path_is_absolute")
    assert CHECKPOINT_MOUNT_PATH.startswith("/")


def test_checkpoint_volume_name_is_dns_safe() -> None:
    """Volume names must be valid DNS labels (lowercase, alphanumeric, hyphens)."""
    print("Executing test: checkpoint_volume_name_is_dns_safe")
    assert CHECKPOINT_VOLUME_NAME.lower() == CHECKPOINT_VOLUME_NAME
    assert all(c.isalnum() or c == "-" for c in CHECKPOINT_VOLUME_NAME)
    assert not CHECKPOINT_VOLUME_NAME.startswith("-")
    assert not CHECKPOINT_VOLUME_NAME.endswith("-")


def test_checkpoint_incomplete_marker_has_extension() -> None:
    """Marker file should have a file extension for easy identification."""
    print("Executing test: checkpoint_incomplete_marker_has_extension")
    assert "." in CHECKPOINT_INCOMPLETE_MARKER


def test_checkpoint_staging_dir_has_no_slashes() -> None:
    """Staging dir is a relative directory name, not a path."""
    print("Executing test: checkpoint_staging_dir_has_no_slashes")
    assert "/" not in CHECKPOINT_STAGING_DIR


def test_ephemeral_volume_size_is_valid_k8s_quantity() -> None:
    """Ephemeral volume size must be a positive integer with a binary/SI suffix."""
    print("Executing test: ephemeral_volume_size_is_valid_k8s_quantity")
    assert CHECKPOINT_EPHEMERAL_VOLUME_SIZE.endswith(("Gi", "Mi", "Ki", "G", "M", "K"))
    numeric_part = CHECKPOINT_EPHEMERAL_VOLUME_SIZE
    for suffix in ("Gi", "Mi", "Ki", "G", "M", "K"):
        if numeric_part.endswith(suffix):
            numeric_part = numeric_part[: -len(suffix)]
            break
    assert numeric_part.isdigit()
    assert int(numeric_part) > 0


def test_all_constants_are_non_empty_strings() -> None:
    """No constant should be an empty string."""
    print("Executing test: all_constants_are_non_empty_strings")
    constants = [
        ANNOTATION_PROGRESSION_TRACKING,
        ANNOTATION_METRICS_PORT,
        ANNOTATION_METRICS_POLL_INTERVAL,
        ANNOTATION_TRAINER_STATUS,
        PVC_URI_SCHEME,
        CHECKPOINT_MOUNT_PATH,
        CHECKPOINT_VOLUME_NAME,
        CHECKPOINT_INCOMPLETE_MARKER,
        CHECKPOINT_STAGING_DIR,
        CHECKPOINT_EPHEMERAL_VOLUME_SIZE,
        S3_URI_SCHEME,
    ]
    for const in constants:
        assert isinstance(const, str)
        assert len(const) > 0
