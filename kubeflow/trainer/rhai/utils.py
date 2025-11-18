import logging
from typing import Optional, Union

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
        raise ValueError(f"Unknown trainer {trainer}.")


def merge_progression_annotations(
    trainer: Union[transformers.TransformersTrainer, traininghub.TrainingHubTrainer],
    spec_annotations: Optional[dict[str, str]] = None,
) -> Optional[dict[str, str]]:
    """Merge progression tracking annotations for RHAI trainers with existing annotations.

    Args:
        trainer: RHAI trainer instance (TransformersTrainer or TrainingHubTrainer).
        spec_annotations: Existing annotations dict to merge with, if any.

    Returns:
        Merged annotations dict with progression tracking added (if enabled),
        or original spec_annotations if progression tracking is disabled.
    """
    if (
        not hasattr(trainer, "enable_progression_tracking")
        or not trainer.enable_progression_tracking
        or not hasattr(trainer, "metrics_port")
        or not hasattr(trainer, "metrics_poll_interval_seconds")
    ):
        return spec_annotations

    from kubeflow.trainer.rhai.constants import (
        ANNOTATION_FRAMEWORK,
        ANNOTATION_METRICS_POLL_INTERVAL,
        ANNOTATION_METRICS_PORT,
        ANNOTATION_PROGRESSION_TRACKING,
    )

    if isinstance(trainer, transformers.TransformersTrainer):
        framework = "transformers"
    elif isinstance(trainer, traininghub.TrainingHubTrainer):
        framework = "traininghub"
    else:
        framework = "unknown"

    progression_annotations = {
        ANNOTATION_PROGRESSION_TRACKING: "true",
        ANNOTATION_METRICS_PORT: str(trainer.metrics_port),
        ANNOTATION_METRICS_POLL_INTERVAL: str(trainer.metrics_poll_interval_seconds),
        ANNOTATION_FRAMEWORK: framework,
    }

    if spec_annotations is None:
        return progression_annotations
    return {**spec_annotations, **progression_annotations}
