import logging
from typing import Any, NoReturn, Optional, Union

from kubeflow_trainer_api import models

from kubeflow.trainer.rhai import traininghub, transformers
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


def get_trainer_cr_from_rhai_trainer(
    runtime: types.Runtime,
    trainer: Union[transformers.TransformersTrainer, traininghub.TrainingHubTrainer],
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    if isinstance(trainer, traininghub.TrainingHubTrainer):
        return traininghub.get_trainer_cr_from_training_hub_trainer(
            runtime,
            trainer,
            initializer,
        )

    elif isinstance(trainer, transformers.TransformersTrainer):
        return transformers.get_trainer_cr_from_transformers_trainer(
            runtime,
            trainer,
            initializer,
        )

    else:
        _raise_unknown_rhai_trainer(trainer)


def _raise_unknown_rhai_trainer(trainer: object) -> NoReturn:
    raise ValueError(f"Unknown trainer {trainer}.")


def get_job_progress(
    name: str,
    namespace: str = "default",
    backend_config: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Get training progress metrics from TrainJob annotations.

    Only works for jobs with progression tracking enabled (e.g., TransformersTrainer
    with enable_progression_tracking=True).

    Args:
        name: Name of the TrainJob
        namespace: Kubernetes namespace (only for KubernetesBackend)
        backend_config: Backend configuration (defaults to KubernetesBackendConfig)

    Returns:
        Dictionary with progress data including progressPercentage, estimatedRemainingSeconds,
        currentStep, totalSteps, currentEpoch, totalEpochs, trainMetrics, and evalMetrics.
        Returns None if progression tracking not enabled or metrics unavailable.

    Raises:
        TimeoutError: Timeout to get TrainJob
        RuntimeError: Failed to get TrainJob
    """
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.trainer.api import TrainerClient

    if backend_config is None:
        backend_config = KubernetesBackendConfig(namespace=namespace)

    client = TrainerClient(backend_config=backend_config)
    job = client.get_job(name=name)
    return _parse_progress_from_job(job)


def _parse_progress_from_job(job: Any) -> Optional[dict[str, Any]]:
    """Parse progression tracking metrics from TrainJob annotations.

    Args:
        job: TrainJob object (supports dict or V1ObjectMeta)

    Returns:
        Progress data dictionary or None if unavailable
    """
    import json

    from kubernetes import client

    from kubeflow.trainer.rhai.constants import ANNOTATION_TRAINER_STATUS

    if hasattr(job, "metadata") and isinstance(job.metadata, client.V1ObjectMeta):
        annotations = job.metadata.annotations or {}
    elif isinstance(job, dict):
        annotations = job.get("metadata", {}).get("annotations", {})
    else:
        return None

    status_annotation = annotations.get(ANNOTATION_TRAINER_STATUS)
    if not status_annotation:
        logger.info(
            "Progression tracking is not enabled for this TrainJob. "
            "Use TransformersTrainer with enable_progression_tracking=True "
            "to enable real-time progress."
        )
        return None

    try:
        return json.loads(status_annotation)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse progression tracking data: {e}")
        return None
