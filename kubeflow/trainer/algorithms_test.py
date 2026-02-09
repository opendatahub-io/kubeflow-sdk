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

import pytest

from kubeflow.trainer.algorithms import (
    ALGORITHMS,
    AlgorithmSpec,
    get_algorithm_spec,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


def test_algorithm_spec_is_frozen():
    """Test that AlgorithmSpec is frozen and cannot be modified after creation."""
    print("Executing test: algorithm_spec_is_frozen")

    spec = AlgorithmSpec(
        name="test",
        metrics_file_patterns=("pattern.jsonl",),
        validate=lambda x: None,
    )

    # Attempt to modify frozen dataclass should raise FrozenInstanceError or AttributeError
    with pytest.raises((AttributeError, Exception)):
        spec.name = "modified"

    print("test execution complete")


def test_all_algorithms_have_non_empty_metrics_patterns():
    """Test that all registered algorithms define non-empty metrics_file_patterns."""
    print("Executing test: all_algorithms_have_non_empty_metrics_patterns")

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

        # Convert to list to check length
        patterns_list = list(spec.metrics_file_patterns)

        # Verify metrics_file_patterns is not empty
        assert len(patterns_list) > 0, (
            f"Algorithm '{algorithm_name}' has empty metrics_file_patterns"
        )

        # Verify all patterns are non-empty strings
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
        assert "Unsupported training algorithm" in error_message

        # For non-empty algorithm names, verify the name is in the error
        if test_case.config["algorithm"]:
            assert test_case.config["algorithm"] in error_message

    print("test execution complete")


def test_algorithms_registry_has_expected_algorithms():
    """Test that ALGORITHMS registry contains exactly the expected algorithms."""
    print("Executing test: algorithms_registry_has_expected_algorithms")

    # As of now, only sft and osft are supported (lora_sft is TODO)
    expected_algorithms = {"sft", "osft"}
    actual_algorithms = set(ALGORITHMS.keys())

    assert actual_algorithms == expected_algorithms, (
        f"Expected algorithms {expected_algorithms}, but found {actual_algorithms}"
    )

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
