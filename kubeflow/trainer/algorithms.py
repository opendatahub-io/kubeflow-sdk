"""Centralized registry for training algorithms and their properties.

This module defines a single authoritative place for supported training algorithms,
their metrics file patterns, and validation logic.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmSpec:
    """Specification for a training algorithm.

    This is a frozen (immutable) dataclass that defines the properties
    and behavior of a supported training algorithm.

    Attributes:
        name: The algorithm identifier (e.g., "sft", "osft").
        metrics_file_patterns: Glob patterns for metrics files written by this algorithm.
            Used for metrics reading and cleanup operations.
        validate: Validation function that takes a training config and raises
            ValueError if the config is invalid for this algorithm.
    """

    name: str
    metrics_file_patterns: Iterable[str]
    validate: Callable[[Any], None]


def _no_op_validate(config: Any) -> None:
    """Default no-op validation function.

    This is used for algorithms that don't require algorithm-specific validation.
    Trainer-level validation (ports, intervals, etc.) is handled separately in
    TrainingHubTrainer.__post_init__.

    Args:
        config: The training configuration to validate.

    Raises:
        ValueError: Never raises - this is a no-op.
    """
    pass


# Registry of all supported training algorithms
# Each entry maps algorithm name to its specification
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "sft": AlgorithmSpec(
        name="sft",
        metrics_file_patterns=("training_params_and_metrics_global*.jsonl",),
        validate=_no_op_validate,
    ),
    "osft": AlgorithmSpec(
        name="osft",
        metrics_file_patterns=("training_metrics_*.jsonl",),
        validate=_no_op_validate,
    ),
}


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    """Retrieve the specification for a training algorithm by name.

    Args:
        name: The algorithm identifier (e.g., "sft", "osft").

    Returns:
        The AlgorithmSpec for the requested algorithm.

    Raises:
        ValueError: If the algorithm name is not supported.

    Examples:
        >>> spec = get_algorithm_spec("sft")
        >>> spec.name
        'sft'
        >>> spec.metrics_file_patterns
        ('training_params_and_metrics_global*.jsonl',)

        >>> get_algorithm_spec("unknown")
        Traceback (most recent call last):
            ...
        ValueError: Unsupported training algorithm: 'unknown'. Supported algorithms: osft, sft
    """
    if name not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS.keys()))
        raise ValueError(
            f"Unsupported training algorithm: '{name}'. Supported algorithms: {supported}"
        )

    return ALGORITHMS[name]
