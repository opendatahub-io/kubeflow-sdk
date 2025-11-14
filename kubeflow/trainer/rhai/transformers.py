from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


@dataclass
class TransformersTrainer:
    """Minimal RHAI Transformers trainer (skeleton)."""

    func: Callable
    func_args: Optional[dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    env: Optional[dict[str, str]] = None

    enable_jit_checkpointing: bool = True


def get_trainer_cr_from_transformers_trainer(
    runtime: types.Runtime,
    trainer: TransformersTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    """Build a minimal Trainer CRD for Transformers trainer."""
    trainer_crd = models.TrainerV1alpha1Trainer()

    return trainer_crd
