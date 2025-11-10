# Copyright 2024 The Kubeflow Authors.
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

"""Unit tests for progress monitoring utilities."""

import json
from unittest.mock import MagicMock

from kubernetes import client

from kubeflow.trainer.constants.constants import ANNOTATION_TRAINER_STATUS
from kubeflow.trainer.experimental.utils.progress import _parse_progress_from_job


def test_parse_progress_from_job_with_v1_object_meta():
    """Test parsing progress from TrainJob with V1ObjectMeta."""
    progress_data = {
        "status": "training",
        "status_message": "45% complete",
        "progress": {"step_current": 450, "step_total": 1000, "percent": 45.0},
        "metrics": {"loss": 0.234, "learning_rate": 0.001},
    }

    job = MagicMock()
    job.metadata = client.V1ObjectMeta(
        annotations={ANNOTATION_TRAINER_STATUS: json.dumps(progress_data)}
    )

    result = _parse_progress_from_job(job)

    assert result is not None
    assert result["status"] == "training"
    assert result["progress"]["step_current"] == 450
    assert result["metrics"]["loss"] == 0.234


def test_parse_progress_from_job_with_dict():
    """Test parsing progress from TrainJob dict format."""
    progress_data = {
        "status": "completed",
        "status_message": "Training completed",
        "progress": {"percent": 100.0},
    }

    job = {"metadata": {"annotations": {ANNOTATION_TRAINER_STATUS: json.dumps(progress_data)}}}

    result = _parse_progress_from_job(job)

    assert result is not None
    assert result["status"] == "completed"
    assert result["progress"]["percent"] == 100.0


def test_parse_progress_no_annotation():
    """Test parsing when progression tracking annotation is missing."""
    job = MagicMock()
    job.metadata = client.V1ObjectMeta(annotations={})

    result = _parse_progress_from_job(job)

    assert result is None


def test_parse_progress_none_annotations():
    """Test parsing when annotations is None."""
    job = MagicMock()
    job.metadata = client.V1ObjectMeta(annotations=None)

    result = _parse_progress_from_job(job)

    assert result is None


def test_parse_progress_invalid_json():
    """Test parsing with malformed JSON in annotation."""
    job = MagicMock()
    job.metadata = client.V1ObjectMeta(annotations={ANNOTATION_TRAINER_STATUS: "not valid json{"})

    result = _parse_progress_from_job(job)

    assert result is None


def test_parse_progress_empty_annotation():
    """Test parsing with empty string annotation."""
    job = MagicMock()
    job.metadata = client.V1ObjectMeta(annotations={ANNOTATION_TRAINER_STATUS: ""})

    result = _parse_progress_from_job(job)

    assert result is None


def test_parse_progress_invalid_job_object():
    """Test parsing with invalid job object."""
    result = _parse_progress_from_job(None)
    assert result is None

    result = _parse_progress_from_job("invalid")
    assert result is None


def test_parse_progress_with_custom_metrics():
    """Test parsing progress with custom metrics."""
    progress_data = {
        "status": "training",
        "metrics": {
            "loss": 0.5,
            "accuracy": 0.95,
            "f1_score": 0.88,
            "custom_metric": 123.45,
        },
    }

    job = MagicMock()
    job.metadata = client.V1ObjectMeta(
        annotations={ANNOTATION_TRAINER_STATUS: json.dumps(progress_data)}
    )

    result = _parse_progress_from_job(job)

    assert result is not None
    assert result["metrics"]["accuracy"] == 0.95
    assert result["metrics"]["f1_score"] == 0.88
    assert result["metrics"]["custom_metric"] == 123.45


def test_parse_progress_full_structure():
    """Test parsing complete progress data structure."""
    progress_data = {
        "status": "training",
        "status_message": "Training in progress: 45.0% complete, 2h 44m remaining",
        "status_details": {
            "last_event": "step_completed",
            "last_event_time": "2024-01-15 14:30:00",
        },
        "progress": {"step_current": 450, "step_total": 1000, "percent": 45.0, "epoch": 2},
        "time": {
            "started_sec": 1705329000.0,
            "started_at": "2024-01-15 12:00:00",
            "elapsed_sec": 9000.0,
            "elapsed": "2h 30m",
            "remaining_sec": 11000.0,
            "remaining": "3h 3m",
            "updated_sec": 1705338000.0,
            "updated_at": "2024-01-15 14:30:00",
        },
        "metrics": {
            "loss": 0.234,
            "learning_rate": 0.0001,
            "throughput_samples_sec": 125.5,
        },
        "checkpoint": {"last_step": 400, "last_path": "/workspace/checkpoints/step-400"},
    }

    job = MagicMock()
    job.metadata = client.V1ObjectMeta(
        annotations={ANNOTATION_TRAINER_STATUS: json.dumps(progress_data)}
    )

    result = _parse_progress_from_job(job)

    assert result is not None
    assert result["status"] == "training"
    assert result["status_message"] == "Training in progress: 45.0% complete, 2h 44m remaining"
    assert result["progress"]["step_current"] == 450
    assert result["progress"]["epoch"] == 2
    assert result["time"]["elapsed"] == "2h 30m"
    assert result["time"]["remaining"] == "3h 3m"
    assert result["metrics"]["loss"] == 0.234
    assert result["checkpoint"]["last_step"] == 400


def test_annotation_constant_matches_expected_format():
    """Test that the annotation constant has the correct format."""
    # This ensures we're using the right annotation key format
    assert ANNOTATION_TRAINER_STATUS == "trainer.opendatahub.io/trainerStatus"
    assert "/" in ANNOTATION_TRAINER_STATUS
    assert ANNOTATION_TRAINER_STATUS.startswith("trainer.opendatahub.io/")
