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

"""Tests for training algorithms registry."""

import ast

import pytest

from kubeflow.trainer.algorithms import (
    ALGORITHMS,
    AlgorithmSpec,
    get_algorithm_pod_metadata,
    get_algorithm_spec,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


def test_algorithm_spec_is_frozen():
    """Test that AlgorithmSpec is frozen and cannot be modified after creation."""
    print("Executing test: algorithm_spec_is_frozen")

    spec = AlgorithmSpec(
        name="test",
        metrics_file_patterns=("pattern_*.jsonl",),
        validate=lambda x: None,
    )

    # Attempt to modify frozen dataclass should raise FrozenInstanceError or AttributeError
    with pytest.raises((AttributeError, Exception)):
        spec.name = "modified"

    print("test execution complete")


def test_algorithm_spec_validation_empty_name():
    """Test that AlgorithmSpec validates name is not empty."""
    print("Executing test: algorithm_spec_validation_empty_name")

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        AlgorithmSpec(
            name="",
            metrics_file_patterns=("pattern_*.jsonl",),
            validate=lambda x: None,
        )

    print("test execution complete")


def test_algorithm_spec_validation_uppercase_name():
    """Test that AlgorithmSpec validates name is lowercase."""
    print("Executing test: algorithm_spec_validation_uppercase_name")

    with pytest.raises(ValueError, match="must be lowercase"):
        AlgorithmSpec(
            name="SFT",
            metrics_file_patterns=("pattern_*.jsonl",),
            validate=lambda x: None,
        )

    print("test execution complete")


def test_algorithm_spec_allows_empty_patterns():
    """Test that AlgorithmSpec allows empty patterns for algorithms without metrics."""
    print("Executing test: algorithm_spec_allows_empty_patterns")

    # Empty patterns should be allowed (for algorithms that don't produce metrics files)
    spec = AlgorithmSpec(
        name="test",
        metrics_file_patterns=(),
        validate=lambda x: None,
    )

    assert spec.name == "test"
    assert list(spec.metrics_file_patterns) == []

    print("test execution complete")


def test_algorithm_spec_validation_non_callable_validate():
    """Test that AlgorithmSpec validates validate is callable."""
    print("Executing test: algorithm_spec_validation_non_callable_validate")

    with pytest.raises(ValueError, match="validate must be callable"):
        AlgorithmSpec(
            name="test",
            metrics_file_patterns=("pattern_*.jsonl",),
            validate="not a function",  # type: ignore[arg-type]
        )

    print("test execution complete")


def test_get_algorithm_pod_metadata_rank0_derived_correctly():
    """Test that rank0 file is correctly derived from pattern."""
    print("Executing test: get_algorithm_pod_metadata_rank0_derived_correctly")

    for algorithm_name in ALGORITHMS:
        metadata = get_algorithm_pod_metadata(algorithm_name)
        pattern = metadata["metrics_file_pattern"]
        rank0 = metadata["metrics_file_rank0"]

        # Skip algorithms without metrics
        if pattern is None:
            assert rank0 is None, (
                f"Algorithm '{algorithm_name}' has no pattern but rank0 is not None"
            )
            continue

        # Verify derivation: pattern with * replaced by 0
        assert rank0 == pattern.replace("*", "0"), (
            f"Algorithm '{algorithm_name}' rank0 file should be pattern with * replaced by 0. "
            f"Expected '{pattern.replace('*', '0')}', got '{rank0}'"
        )

    print("test execution complete")


def test_all_algorithms_have_valid_metrics_patterns():
    """Test that all registered algorithms define valid metrics_file_patterns."""
    print("Executing test: all_algorithms_have_valid_metrics_patterns")

    for algorithm_name, spec in ALGORITHMS.items():
        print("  Checking algorithm:", algorithm_name)

        # Verify metrics_file_patterns is not None
        assert spec.metrics_file_patterns is not None, (
            f"Algorithm '{algorithm_name}' has None metrics_file_patterns"
        )

        # Verify metrics_file_patterns is iterable
        assert hasattr(spec.metrics_file_patterns, "__iter__"), (
            f"Algorithm '{algorithm_name}' metrics_file_patterns is not iterable"
        )

        # Convert to list to check contents
        patterns_list = list(spec.metrics_file_patterns)

        # Empty patterns are allowed (for algorithms without metrics files)
        # but if patterns are provided, validate them
        for pattern in patterns_list:
            assert isinstance(pattern, str), (
                f"Algorithm '{algorithm_name}' has non-string pattern: {pattern}"
            )
            assert len(pattern) > 0, f"Algorithm '{algorithm_name}' has empty string pattern"

    print("test execution complete")


def test_all_algorithms_have_callable_validate():
    """Test that all registered algorithms define a callable validate function."""
    print("Executing test: all_algorithms_have_callable_validate")

    for algorithm_name, spec in ALGORITHMS.items():
        print("  Checking algorithm:", algorithm_name)

        # Verify validate is not None
        assert spec.validate is not None, f"Algorithm '{algorithm_name}' has None validate function"

        # Verify validate is callable
        assert callable(spec.validate), f"Algorithm '{algorithm_name}' validate is not callable"

        # Verify validate can be called without raising (with any argument)
        try:
            spec.validate(None)
        except Exception as e:
            # If it raises, it should be ValueError (validation failed)
            # Any other exception is a bug in the validate function
            if not isinstance(e, ValueError):
                pytest.fail(
                    f"Algorithm '{algorithm_name}' validate raised unexpected "
                    f"exception: {type(e).__name__}: {e}"
                )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="unknown algorithm raises ValueError",
            expected_status=FAILED,
            config={"algorithm": "unknown_algorithm"},
            expected_error=ValueError,
        ),
        TestCase(
            name="empty string algorithm raises ValueError",
            expected_status=FAILED,
            config={"algorithm": ""},
            expected_error=ValueError,
        ),
        TestCase(
            name="valid sft algorithm returns spec",
            expected_status=SUCCESS,
            config={"algorithm": "sft"},
            expected_output="sft",
        ),
        TestCase(
            name="valid osft algorithm returns spec",
            expected_status=SUCCESS,
            config={"algorithm": "osft"},
            expected_output="osft",
        ),
    ],
)
def test_get_algorithm_spec(test_case):
    """Test get_algorithm_spec for various inputs."""
    print("Executing test:", test_case.name)

    try:
        spec = get_algorithm_spec(test_case.config["algorithm"])

        assert test_case.expected_status == SUCCESS
        assert spec.name == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

        # For error cases, verify error message contains expected information
        error_message = str(e)

        # Empty string gets caught by input validation, not unsupported algorithm check
        if test_case.config["algorithm"] == "":
            assert "must be a non-empty string" in error_message
        else:
            # Non-empty invalid algorithm names
            assert "Unsupported training algorithm" in error_message
            assert test_case.config["algorithm"] in error_message

    print("test execution complete")


def test_get_algorithm_spec_validates_input():
    """Test that get_algorithm_spec validates input parameter."""
    print("Executing test: get_algorithm_spec_validates_input")

    # Empty string
    with pytest.raises(ValueError, match="must be a non-empty string"):
        get_algorithm_spec("")

    # None
    with pytest.raises(ValueError, match="must be a non-empty string"):
        get_algorithm_spec(None)  # type: ignore[arg-type]

    # Non-string type
    with pytest.raises(ValueError, match="must be a non-empty string"):
        get_algorithm_spec(123)  # type: ignore[arg-type]

    print("test execution complete")


def test_algorithms_registry_not_empty():
    """Test that ALGORITHMS registry is not empty."""
    print("Executing test: algorithms_registry_not_empty")

    assert len(ALGORITHMS) > 0, "ALGORITHMS registry should not be empty"

    # Verify known algorithms exist (baseline check)
    # This ensures we don't accidentally break existing algorithms
    known_algorithms = {"sft", "osft", "lora_sft"}
    for algo in known_algorithms:
        assert algo in ALGORITHMS, f"Known algorithm '{algo}' missing from registry"

    print("test execution complete")


def test_algorithm_spec_name_matches_registry_key():
    """Test that each AlgorithmSpec.name matches its registry key."""
    print("Executing test: algorithm_spec_name_matches_registry_key")

    for registry_key, spec in ALGORITHMS.items():
        assert spec.name == registry_key, (
            f"Registry key '{registry_key}' does not match spec.name '{spec.name}'"
        )

    print("test execution complete")


def test_sft_metrics_patterns_match_constants():
    """Test that SFT metrics patterns match the patterns used in traininghub.py."""
    print("Executing test: sft_metrics_patterns_match_constants")

    spec = get_algorithm_spec("sft")
    patterns = list(spec.metrics_file_patterns)

    # Should match TRAININGHUB_SFT_METRICS_FILE_PATTERN from traininghub.py
    assert "training_params_and_metrics_global*.jsonl" in patterns

    print("test execution complete")


def test_osft_metrics_patterns_match_constants():
    """Test that OSFT metrics patterns match the patterns used in traininghub.py."""
    print("Executing test: osft_metrics_patterns_match_constants")

    spec = get_algorithm_spec("osft")
    patterns = list(spec.metrics_file_patterns)

    # Should match TRAININGHUB_OSFT_METRICS_FILE_PATTERN from traininghub.py
    assert "training_metrics_*.jsonl" in patterns

    print("test execution complete")


def test_algorithm_spec_metrics_patterns_are_tuples():
    """Test that metrics_file_patterns are stored as tuples (immutable)."""
    print("Executing test: algorithm_spec_metrics_patterns_are_tuples")

    for algorithm_name, spec in ALGORITHMS.items():
        print("  Checking algorithm:", algorithm_name)

        # Verify patterns are tuples (immutable)
        assert isinstance(spec.metrics_file_patterns, tuple), (
            f"Algorithm '{algorithm_name}' metrics_file_patterns should be tuple, "
            f"got {type(spec.metrics_file_patterns).__name__}"
        )

    print("test execution complete")


def test_get_algorithm_pod_metadata_sft():
    """Test get_algorithm_pod_metadata returns correct metadata for SFT."""
    print("Executing test: get_algorithm_pod_metadata_sft")

    metadata = get_algorithm_pod_metadata("sft")

    assert metadata["name"] == "sft"
    assert metadata["metrics_file_pattern"] == "training_params_and_metrics_global*.jsonl"
    assert metadata["metrics_file_rank0"] == "training_params_and_metrics_global0.jsonl"

    print("test execution complete")


def test_get_algorithm_pod_metadata_osft():
    """Test get_algorithm_pod_metadata returns correct metadata for OSFT."""
    print("Executing test: get_algorithm_pod_metadata_osft")

    metadata = get_algorithm_pod_metadata("osft")

    assert metadata["name"] == "osft"
    assert metadata["metrics_file_pattern"] == "training_metrics_*.jsonl"
    assert metadata["metrics_file_rank0"] == "training_metrics_0.jsonl"

    print("test execution complete")


def test_get_algorithm_pod_metadata_unknown_raises():
    """Test get_algorithm_pod_metadata raises for unknown algorithm."""
    print("Executing test: get_algorithm_pod_metadata_unknown_raises")

    with pytest.raises(ValueError) as exc_info:
        get_algorithm_pod_metadata("unknown")

    assert "Unsupported training algorithm" in str(exc_info.value)

    print("test execution complete")


def test_get_algorithm_pod_metadata_no_metrics():
    """Test get_algorithm_pod_metadata returns None for algorithms without metrics."""
    print("Executing test: get_algorithm_pod_metadata_no_metrics")

    # Temporarily add an algorithm with no metrics to test
    from kubeflow.trainer.algorithms import ALGORITHMS, AlgorithmSpec, _no_op_validate

    # Save original registry
    original_algorithms = ALGORITHMS.copy()

    try:
        # Add test algorithm with no metrics
        ALGORITHMS["test_no_metrics"] = AlgorithmSpec(
            name="test_no_metrics",
            metrics_file_patterns=(),
            validate=_no_op_validate,
        )

        metadata = get_algorithm_pod_metadata("test_no_metrics")

        assert metadata["name"] == "test_no_metrics"
        assert metadata["metrics_file_pattern"] is None
        assert metadata["metrics_file_rank0"] is None

    finally:
        # Restore original registry
        ALGORITHMS.clear()
        ALGORITHMS.update(original_algorithms)

    print("test execution complete")


def test_pod_metadata_has_all_required_keys():
    """Test that pod metadata contains all required keys for all algorithms."""
    print("Executing test: pod_metadata_has_all_required_keys")

    required_keys = {"name", "metrics_file_pattern", "metrics_file_rank0"}

    for algorithm_name in ALGORITHMS:
        print("  Checking algorithm:", algorithm_name)
        metadata = get_algorithm_pod_metadata(algorithm_name)

        # Verify all required keys are present
        assert set(metadata.keys()) == required_keys, (
            f"Algorithm '{algorithm_name}' metadata missing required keys. "
            f"Expected {required_keys}, got {set(metadata.keys())}"
        )

        # Verify name matches
        assert metadata["name"] == algorithm_name

        # Patterns can be None for algorithms without metrics
        # If pattern is not None, verify rank0 is also not None
        if metadata["metrics_file_pattern"] is not None:
            assert metadata["metrics_file_rank0"] is not None
        else:
            assert metadata["metrics_file_rank0"] is None

    print("test execution complete")


def test_pod_metadata_rank0_derived_from_pattern():
    """Test that metrics_file_rank0 is correctly derived from pattern."""
    print("Executing test: pod_metadata_rank0_derived_from_pattern")

    for algorithm_name in ALGORITHMS:
        print("  Checking algorithm:", algorithm_name)
        metadata = get_algorithm_pod_metadata(algorithm_name)

        pattern = metadata["metrics_file_pattern"]

        # Skip algorithms without metrics
        if pattern is None:
            assert metadata["metrics_file_rank0"] is None, (
                f"Algorithm '{algorithm_name}' has no pattern but rank0 is not None"
            )
            continue

        # Verify rank0 file is derived by replacing * with 0
        expected_rank0 = pattern.replace("*", "0")
        assert metadata["metrics_file_rank0"] == expected_rank0, (
            f"Algorithm '{algorithm_name}': rank0 file should be pattern with * replaced by 0. "
            f"Expected '{expected_rank0}', got '{metadata['metrics_file_rank0']}'"
        )

    print("test execution complete")


def test_pod_metadata_self_contained():
    """Test that pod metadata is self-contained and usable without imports."""
    print("Executing test: pod_metadata_self_contained")

    for algorithm_name in ALGORITHMS:
        print("  Checking algorithm:", algorithm_name)
        metadata = get_algorithm_pod_metadata(algorithm_name)

        # Verify metadata is a plain dict (serializable to pod code)
        assert isinstance(metadata, dict)

        # Verify all values are basic Python types (str, int, bool, None, etc.)
        # Pod code should not need to import anything to use this metadata
        for key, value in metadata.items():
            assert isinstance(key, str), f"Metadata key should be string, got {type(key)}"
            # None is allowed for algorithms without metrics
            assert isinstance(value, (str, int, bool, type(None))), (
                f"Metadata value for key '{key}' should be basic type, got {type(value).__name__}"
            )

    print("test execution complete")


def test_pod_metadata_patterns_are_strings_or_none():
    """Test that all file patterns in pod metadata are strings or None."""
    print("Executing test: pod_metadata_patterns_are_strings_or_none")

    for algorithm_name in ALGORITHMS:
        print("  Checking algorithm:", algorithm_name)
        metadata = get_algorithm_pod_metadata(algorithm_name)

        # Verify pattern fields are strings or None
        assert isinstance(metadata["metrics_file_pattern"], (str, type(None)))
        assert isinstance(metadata["metrics_file_rank0"], (str, type(None)))

        # If patterns exist, verify they are not empty
        if metadata["metrics_file_pattern"] is not None:
            assert len(metadata["metrics_file_pattern"]) > 0
            assert len(metadata["metrics_file_rank0"]) > 0

    print("test execution complete")


def test_pod_metadata_serializable():
    """Test that pod metadata can be serialized and embedded in code."""
    print("Executing test: pod_metadata_serializable")

    for algorithm_name in ALGORITHMS:
        print("  Checking algorithm:", algorithm_name)
        metadata = get_algorithm_pod_metadata(algorithm_name)

        # Verify metadata can be repr'd (for embedding in generated code)
        metadata_repr = repr(metadata)
        assert isinstance(metadata_repr, str)
        assert len(metadata_repr) > 0

        # Verify metadata can be eval'd back (sanity check)
        metadata_evaled = ast.literal_eval(metadata_repr)
        assert metadata_evaled == metadata

    print("test execution complete")


def test_registry_all_specs_are_frozen():
    """Test that all AlgorithmSpecs in registry are frozen (immutable)."""
    print("Executing test: registry_all_specs_are_frozen")

    for algorithm_name, spec in ALGORITHMS.items():
        print("  Checking algorithm:", algorithm_name)

        # Verify spec is an AlgorithmSpec instance
        assert isinstance(spec, AlgorithmSpec)

        # Verify spec is frozen (attempt to modify should fail)
        with pytest.raises((AttributeError, Exception)):
            spec.name = "modified"

    print("test execution complete")


def test_registry_keys_are_lowercase():
    """Test that all algorithm registry keys are lowercase."""
    print("Executing test: registry_keys_are_lowercase")

    for algorithm_name in ALGORITHMS:
        assert algorithm_name == algorithm_name.lower(), (
            f"Algorithm name '{algorithm_name}' should be lowercase"
        )

    print("test execution complete")


def test_registry_no_duplicate_patterns():
    """Test that algorithms don't share identical metrics patterns."""
    print("Executing test: registry_no_duplicate_patterns")

    patterns_seen = {}

    for algorithm_name, spec in ALGORITHMS.items():
        print("  Checking algorithm:", algorithm_name)
        patterns = tuple(spec.metrics_file_patterns)

        if patterns in patterns_seen:
            other_algo = patterns_seen[patterns]
            pytest.fail(
                f"Algorithm '{algorithm_name}' has same patterns as '{other_algo}': {patterns}"
            )

        patterns_seen[patterns] = algorithm_name

    print("test execution complete")


def test_registry_validate_functions_accept_any():
    """Test that all validate functions accept Any parameter."""
    print("Executing test: registry_validate_functions_accept_any")

    for algorithm_name, spec in ALGORITHMS.items():
        print("  Checking algorithm:", algorithm_name)

        # Verify validate function doesn't crash on None
        try:
            spec.validate(None)
        except ValueError:
            # ValueError is acceptable (validation failed)
            pass
        except Exception as e:
            # Any other exception is a bug
            pytest.fail(
                f"Algorithm '{algorithm_name}' validate raised unexpected "
                f"exception for None: {type(e).__name__}: {e}"
            )

        # Verify validate function doesn't crash on empty dict
        try:
            spec.validate({})
        except ValueError:
            # ValueError is acceptable (validation failed)
            pass
        except Exception as e:
            # Any other exception is a bug
            pytest.fail(
                f"Algorithm '{algorithm_name}' validate raised unexpected "
                f"exception for {{}}: {type(e).__name__}: {e}"
            )

    print("test execution complete")


def test_no_algorithm_branching_outside_pod_code():
    """Test that algorithm branching only exists in allowed places.

    This guardrail test prevents regression to scattered algorithm logic.
    Algorithm branching (if algorithm == "sft") is only allowed in:
    1. Pod-injected code (_create_training_hub_progression_instrumentation)
       - For method dispatch to algorithm-specific readers/transformers
    2. Pod wrapper code (_render_algorithm_wrapper)
       - For termination message writing

    All other code should use the centralized registry.
    """
    print("Executing test: no_algorithm_branching_outside_pod_code")

    import os
    import re

    # Read the traininghub.py file
    traininghub_path = os.path.join(os.path.dirname(__file__), "rhai", "traininghub.py")

    if not os.path.exists(traininghub_path):
        # File might be in different location during testing
        pytest.skip(f"Could not find traininghub.py at {traininghub_path}")

    with open(traininghub_path) as f:
        content = f.read()

    # Pattern to find algorithm branching
    # Matches: if algorithm == "...", elif algorithm == "...", etc.
    branching_pattern = re.compile(r'(if|elif)\s+algorithm\s*==\s*["\'](\w+)["\']')

    # Find all matches with their line numbers
    matches = []
    for i, line in enumerate(content.split("\n"), start=1):
        for match in branching_pattern.finditer(line):
            algo_name = match.group(2)
            matches.append((i, line.strip(), algo_name))

    # Identify which function each match is in
    # Extract function definitions to map line numbers to functions
    func_pattern = re.compile(r"^def\s+(\w+)\(")
    current_func = None
    func_ranges = []

    for i, line in enumerate(content.split("\n"), start=1):
        func_match = func_pattern.match(line)
        if func_match:
            current_func = func_match.group(1)
            func_ranges.append((current_func, i))

    # Allowed functions where algorithm branching is permitted
    allowed_functions = {
        "_create_training_hub_progression_instrumentation",
        "_read_latest_metrics",  # method inside pod-injected class
        "_transform_schema",  # method inside pod-injected class
        "_write_termination_message",  # inside _render_algorithm_wrapper
    }

    # Check each match
    forbidden_matches = []
    for line_num, line_content, _algo_name in matches:
        # Find which function this line is in
        func_name = None
        for fname, fline in reversed(func_ranges):
            if fline <= line_num:
                func_name = fname
                break

        # Check if this is in an allowed function or part of pod-injected code
        # Also allow if it's inside a nested method (e.g., _read_* or _transform_*)
        if (
            func_name not in allowed_functions
            and "_read_" not in line_content
            and "_transform_" not in line_content
        ):
            forbidden_matches.append((line_num, line_content, func_name))

    if forbidden_matches:
        error_msg = [
            "Algorithm branching found outside allowed pod-injected code:",
            "",
        ]
        for line_num, line_content, func_name in forbidden_matches:
            error_msg.append(f"  Line {line_num} in {func_name}: {line_content}")
        error_msg.append("")
        error_msg.append("Algorithm branching should only exist in pod-injected functions.")
        error_msg.append("Use get_algorithm_spec() or get_algorithm_pod_metadata() instead.")
        pytest.fail("\n".join(error_msg))

    print("  ✓ No forbidden algorithm branching found")
    print("test execution complete")


def test_algorithm_without_metrics_integration():
    """Test that algorithms without metrics work end-to-end."""
    print("Executing test: algorithm_without_metrics_integration")

    from kubeflow.trainer.algorithms import ALGORITHMS, AlgorithmSpec, _no_op_validate

    # Save original registry
    original_algorithms = ALGORITHMS.copy()

    try:
        # Add test algorithm with no metrics
        ALGORITHMS["test_no_metrics"] = AlgorithmSpec(
            name="test_no_metrics",
            metrics_file_patterns=(),
            validate=_no_op_validate,
        )

        # Test get_algorithm_spec
        spec = get_algorithm_spec("test_no_metrics")
        assert spec.name == "test_no_metrics"
        assert list(spec.metrics_file_patterns) == []

        # Test get_algorithm_pod_metadata
        metadata = get_algorithm_pod_metadata("test_no_metrics")
        assert metadata["name"] == "test_no_metrics"
        assert metadata["metrics_file_pattern"] is None
        assert metadata["metrics_file_rank0"] is None

        # Test that metadata is serializable (for pod injection)
        import json

        metadata_json = json.dumps(metadata)
        assert metadata_json
        rehydrated = json.loads(metadata_json)
        assert rehydrated == metadata

    finally:
        # Restore original registry
        ALGORITHMS.clear()
        ALGORITHMS.update(original_algorithms)

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
