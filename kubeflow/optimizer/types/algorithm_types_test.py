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

"""Unit tests for kubeflow.optimizer.types.algorithm_types module."""

import pytest

from kubeflow.optimizer.types.algorithm_types import (
    ALGORITHM_REGISTRY,
    BaseAlgorithm,
    GridSearch,
    RandomSearch,
    algorithm_to_katib_spec,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="grid search algorithm name",
            expected_status=SUCCESS,
            expected_output="grid",
        ),
        TestCase(
            name="random search algorithm name",
            expected_status=SUCCESS,
            expected_output="random",
        ),
    ],
)
def test_algorithm_name(test_case: TestCase):
    """Test that algorithm classes return correct algorithm_name."""
    print("Executing test:", test_case.name)

    algo = GridSearch() if test_case.expected_output == "grid" else RandomSearch()

    assert algo.algorithm_name == test_case.expected_output
    print("test execution complete")


def test_base_algorithm_cannot_be_instantiated():
    """Test that BaseAlgorithm ABC cannot be instantiated directly."""
    print("Executing test: base algorithm cannot be instantiated")
    with pytest.raises(TypeError):
        BaseAlgorithm()
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="grid search has no settings",
            expected_status=SUCCESS,
            config={"algo": GridSearch()},
            expected_output={"algorithm_name": "grid", "algorithm_settings": None},
        ),
        TestCase(
            name="random search with no random_state",
            expected_status=SUCCESS,
            config={"algo": RandomSearch()},
            expected_output={"algorithm_name": "random", "algorithm_settings": None},
        ),
        TestCase(
            name="random search with random_state set",
            expected_status=SUCCESS,
            config={"algo": RandomSearch(random_state=42)},
            expected_output={
                "algorithm_name": "random",
                "has_settings": True,
                "setting_name": "random_state",
                "setting_value": "42",
            },
        ),
    ],
)
def test_algorithm_to_katib_spec(test_case: TestCase):
    """Test that algorithm_to_katib_spec converts algorithms to Katib AlgorithmSpec."""
    print("Executing test:", test_case.name)

    spec = algorithm_to_katib_spec(test_case.config["algo"])

    assert spec.algorithm_name == test_case.expected_output["algorithm_name"]

    if test_case.expected_output.get("has_settings"):
        assert spec.algorithm_settings is not None
        assert len(spec.algorithm_settings) == 1
        assert spec.algorithm_settings[0].name == test_case.expected_output["setting_name"]
        assert spec.algorithm_settings[0].value == test_case.expected_output["setting_value"]
    else:
        assert spec.algorithm_settings == test_case.expected_output["algorithm_settings"]

    print("test execution complete")


def test_to_katib_spec_delegates_to_helper():
    """Test that _to_katib_spec on concrete classes produces matching output."""
    print("Executing test: _to_katib_spec delegates to helper")
    algo = RandomSearch(random_state=7)
    spec_via_method = algo._to_katib_spec()
    spec_via_helper = algorithm_to_katib_spec(algo)

    assert spec_via_method.algorithm_name == spec_via_helper.algorithm_name
    assert len(spec_via_method.algorithm_settings) == len(spec_via_helper.algorithm_settings)
    assert spec_via_method.algorithm_settings[0].name == spec_via_helper.algorithm_settings[0].name
    assert (
        spec_via_method.algorithm_settings[0].value == spec_via_helper.algorithm_settings[0].value
    )
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="registry maps grid to GridSearch",
            expected_status=SUCCESS,
            config={"key": "grid"},
            expected_output=GridSearch,
        ),
        TestCase(
            name="registry maps random to RandomSearch",
            expected_status=SUCCESS,
            config={"key": "random"},
            expected_output=RandomSearch,
        ),
    ],
)
def test_algorithm_registry(test_case: TestCase):
    """Test that ALGORITHM_REGISTRY maps algorithm names to correct classes."""
    print("Executing test:", test_case.name)
    assert ALGORITHM_REGISTRY[test_case.config["key"]] is test_case.expected_output
    print("test execution complete")


def test_algorithm_registry_completeness():
    """Test that ALGORITHM_REGISTRY contains exactly the expected algorithms."""
    print("Executing test: algorithm registry completeness")
    assert set(ALGORITHM_REGISTRY.keys()) == {"grid", "random"}
    print("test execution complete")


def test_random_search_default_random_state():
    """Test that RandomSearch defaults random_state to None."""
    print("Executing test: random search default random_state")
    algo = RandomSearch()
    assert algo.random_state is None
    print("test execution complete")
