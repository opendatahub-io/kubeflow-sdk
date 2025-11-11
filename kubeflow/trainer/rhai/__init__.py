from typing import Union

from kubeflow.trainer.rhai.traininghub import TrainingHubAlgorithms, TrainingHubTrainer
from kubeflow.trainer.rhai.transformers import TransformersTrainer

__all__ = (
    "RHAITrainer",
    "TrainingHubAlgorithms",
    "TrainingHubTrainer",
    "TransformersTrainer",
)

RHAITrainer = Union[TransformersTrainer, TrainingHubTrainer]
