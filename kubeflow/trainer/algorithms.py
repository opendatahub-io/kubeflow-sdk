"""Centralized registry for training algorithms and their properties.

This module defines a single authoritative place for supported training algorithms,
their metrics file patterns, and validation logic.

Adding a New Algorithm:
    To add support for a new training algorithm, simply add an entry to the
    ALGORITHMS registry:

    Example with metrics files:
        "my_algorithm": AlgorithmSpec(
            name="my_algorithm",
            metrics_file_patterns=("my_metrics_*.jsonl",),
            validate=_no_op_validate,
        ),

    Example without metrics files (no progress tracking):
        "my_algorithm": AlgorithmSpec(
            name="my_algorithm",
            metrics_file_patterns=(),  # Empty tuple for no metrics
            validate=_no_op_validate,
        ),
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

    def __post_init__(self):
        """Validate the AlgorithmSpec at creation time.

        This fails fast when adding a new algorithm to the registry, catching
        configuration errors during development rather than at runtime.

        Raises:
            ValueError: If any required field is invalid.
        """
        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"Algorithm name must be a non-empty string, got: {self.name!r}")

        if self.name != self.name.lower():
            raise ValueError(
                f"Algorithm name must be lowercase, got: {self.name!r}. "
                f"Use {self.name.lower()!r} instead."
            )

        # Validate metrics_file_patterns (can be empty for algorithms without metrics)
        if self.metrics_file_patterns is None:
            raise ValueError(
                f"Algorithm '{self.name}' metrics_file_patterns cannot be None "
                "(use empty tuple for no metrics)"
            )

        # Convert to list to check contents
        patterns_list = list(self.metrics_file_patterns)

        # Empty patterns are allowed (for algorithms that don't produce metrics files)
        # but if patterns are provided, validate them
        for i, pattern in enumerate(patterns_list):
            if not isinstance(pattern, str):
                raise ValueError(
                    f"Algorithm '{self.name}' pattern {i} must be a string, "
                    f"got {type(pattern).__name__}: {pattern!r}"
                )
            if not pattern:
                raise ValueError(
                    f"Algorithm '{self.name}' has empty string in patterns at index {i}"
                )

        # Validate validate function
        if not callable(self.validate):
            raise ValueError(
                f"Algorithm '{self.name}' validate must be callable, "
                f"got {type(self.validate).__name__}"
            )


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
    "lora_sft": AlgorithmSpec(
        name="lora_sft",
        metrics_file_patterns=(),  # LoRA uses HF Trainer logging, not JSONL metrics files
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
        ValueError: If the algorithm name is not supported or invalid.

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
    # Input validation
    if not name or not isinstance(name, str):
        raise ValueError(
            f"Algorithm name must be a non-empty string, got: {name!r} ({type(name).__name__})"
        )

    if name not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS.keys()))
        raise ValueError(
            f"Unsupported training algorithm: '{name}'. Supported algorithms: {supported}"
        )

    return ALGORITHMS[name]


def get_algorithm_pod_metadata(name: str) -> dict:
    """Get algorithm metadata dict for pod injection.

    This builds a complete metadata dict containing all information needed by
    pod-injected code, derived from the centralized AlgorithmSpec.

    The metadata is automatically derived from the AlgorithmSpec:
    - metrics_file_pattern: First pattern from spec.metrics_file_patterns (None if no patterns)
    - metrics_file_rank0: Derived by replacing '*' with '0' in the pattern (None if no patterns)

    Args:
        name: The algorithm identifier (e.g., "sft", "osft").

    Returns:
        Dict containing:
            - name: Algorithm name
            - metrics_file_pattern: Glob pattern for metrics files
              (None if algorithm produces no metrics)
            - metrics_file_rank0: Specific filename for rank 0 metrics
              (None if algorithm produces no metrics)

    Raises:
        ValueError: If the algorithm name is not supported.

    Examples:
        >>> metadata = get_algorithm_pod_metadata("sft")
        >>> metadata["name"]
        'sft'
        >>> metadata["metrics_file_rank0"]
        'training_params_and_metrics_global0.jsonl'

        >>> metadata = get_algorithm_pod_metadata("lora_sft")  # No metrics
        >>> metadata["metrics_file_pattern"]
        None
    """
    # get_algorithm_spec() validates the name parameter
    spec = get_algorithm_spec(name)

    # Derive metadata from the spec - no algorithm branching needed!
    patterns = list(spec.metrics_file_patterns)

    # Handle algorithms without metrics files
    if not patterns:
        return {
            "name": name,
            "metrics_file_pattern": None,
            "metrics_file_rank0": None,
        }

    # Use first pattern for algorithms with metrics
    pattern = patterns[0]

    # Derive rank0 file by replacing wildcard with 0
    rank0_file = pattern.replace("*", "0")

    return {
        "name": name,
        "metrics_file_pattern": pattern,
        "metrics_file_rank0": rank0_file,
    }
