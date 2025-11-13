from typing import NoReturn, Optional

from kubeflow_trainer_api import models

from kubeflow.trainer.rhai import (
    RHAITrainer,
    traininghub,
    transformers,
)
from kubeflow.trainer.types import types


def get_trainer_cr_from_rhai_trainer(
    runtime: types.Runtime,
    trainer: RHAITrainer,
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
