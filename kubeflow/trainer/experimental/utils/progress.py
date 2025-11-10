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

"""Progress monitoring for experimental trainers with progression tracking."""

import json
from typing import Optional

from kubeflow.trainer.constants.constants import ANNOTATION_TRAINER_STATUS


def get_job_progress(
    name: str,
    namespace: str = "default",
    backend_config=None,
) -> Optional[dict]:
    """Get current training progress for a TrainJob.

    This function retrieves progression tracking metrics from the TrainJob's
    annotations. Only works for jobs with progression tracking enabled
    (e.g., TransformersTrainer).

    Args:
        name: Name of the TrainJob
        namespace: Kubernetes namespace (only for KubernetesBackend)
        backend_config: Backend configuration (defaults to KubernetesBackendConfig)

    Returns:
        Dictionary with training progress data, or None if progression tracking
        is not enabled or metrics are not yet available.

        Dictionary structure:
        {
            "status": str,              # "training", "completed", etc.
            "status_message": str,      # "45% complete • 2h 44m left"
            "progress": {
                "step_current": int,
                "step_total": int,
                "percent": float,
                "epoch": int
            },
            "time": {
                "elapsed": str,         # "2h 15m"
                "remaining": str,       # "2h 44m" (or None)
                "started_at": str,      # "12:00:00"
                "last_update": str      # "14:15:45"
            },
            "metrics": {
                "loss": float,
                "learning_rate": float,
                "throughput_samples_sec": float,
                # ... any custom metrics
            },
            "checkpoint": {
                "last_step": int,
                "last_path": str
            }
        }

    Raises:
        TimeoutError: Timeout to get TrainJob
        RuntimeError: Failed to get TrainJob
    """
    # Import here to avoid circular import
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.trainer.api import TrainerClient

    if backend_config is None:
        backend_config = KubernetesBackendConfig(namespace=namespace)

    client = TrainerClient(backend_config=backend_config)
    job = client.get_job(name=name)
    return _parse_progress_from_job(job)


def _parse_progress_from_job(job) -> Optional[dict]:
    """Parse progression tracking metrics from TrainJob annotations.

    Args:
        job: TrainJob object

    Returns:
        Dictionary with progress data or None if progression tracking not available
    """
    # Import here to avoid circular dependency
    from kubernetes import client

    # Handle both dict and V1ObjectMeta types
    if hasattr(job, "metadata") and isinstance(job.metadata, client.V1ObjectMeta):
        annotations = job.metadata.annotations or {}
    elif isinstance(job, dict):
        annotations = job.get("metadata", {}).get("annotations", {})
    else:
        return None

    status_annotation = annotations.get(ANNOTATION_TRAINER_STATUS)
    if not status_annotation:
        return None

    try:
        return json.loads(status_annotation)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse progress annotation: {e}")
        return None
