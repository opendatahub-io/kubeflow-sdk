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

"""Unit tests for kubeflow.optimizer.types.search_types module."""

import pytest

import kubeflow.optimizer.constants.constants as constants
from kubeflow.optimizer.types.search_types import (
    CategoricalSearchSpace,
    ContinuousSearchSpace,
    Distribution,
    Search,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="uniform creates double parameter with uniform distribution",
            expected_status=SUCCESS,
            config={"min": 0.001, "max": 0.1},
            expected_output={
                "parameterType": constants.DOUBLE_PARAMETER,
                "min": "0.001",
                "max": "0.1",
                "distribution": "uniform",
            },
        ),
        TestCase(
            name="uniform with integer boundaries",
            expected_status=SUCCESS,
            config={"min": 1, "max": 100},
            expected_output={
                "parameterType": constants.DOUBLE_PARAMETER,
                "min": "1",
                "max": "100",
                "distribution": "uniform",
            },
        ),
    ],
)
def test_search_uniform(test_case: TestCase):
    """Test that Search.uniform() produces correct ParameterSpec."""
    print("Executing test:", test_case.name)
    result = Search.uniform(min=test_case.config["min"], max=test_case.config["max"])
    assert result.parameter_type == test_case.expected_output["parameterType"]
    assert result.feasible_space.min == test_case.expected_output["min"]
    assert result.feasible_space.max == test_case.expected_output["max"]
    assert result.feasible_space.distribution == test_case.expected_output["distribution"]
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="loguniform creates double parameter with logUniform distribution",
            expected_status=SUCCESS,
            config={"min": 1e-5, "max": 1e-1},
            expected_output={
                "parameterType": constants.DOUBLE_PARAMETER,
                "min": str(1e-5),
                "max": str(1e-1),
                "distribution": "logUniform",
            },
        ),
    ],
)
def test_search_loguniform(test_case: TestCase):
    """Test that Search.loguniform() produces correct ParameterSpec."""
    print("Executing test:", test_case.name)
    result = Search.loguniform(min=test_case.config["min"], max=test_case.config["max"])
    assert result.parameter_type == test_case.expected_output["parameterType"]
    assert result.feasible_space.min == test_case.expected_output["min"]
    assert result.feasible_space.max == test_case.expected_output["max"]
    assert result.feasible_space.distribution == test_case.expected_output["distribution"]
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="choice with string values",
            expected_status=SUCCESS,
            config={"values": ["adam", "sgd", "rmsprop"]},
            expected_output={
                "parameterType": constants.CATEGORICAL_PARAMETERS,
                "list": ["adam", "sgd", "rmsprop"],
            },
        ),
        TestCase(
            name="choice with integer values stringifies them",
            expected_status=SUCCESS,
            config={"values": [10, 20, 30]},
            expected_output={
                "parameterType": constants.CATEGORICAL_PARAMETERS,
                "list": ["10", "20", "30"],
            },
        ),
        TestCase(
            name="choice with mixed types",
            expected_status=SUCCESS,
            config={"values": [0.01, "auto", 5]},
            expected_output={
                "parameterType": constants.CATEGORICAL_PARAMETERS,
                "list": ["0.01", "auto", "5"],
            },
        ),
    ],
)
def test_search_choice(test_case: TestCase):
    """Test that Search.choice() produces correct ParameterSpec with stringified values."""
    print("Executing test:", test_case.name)
    result = Search.choice(values=test_case.config["values"])
    assert result.parameter_type == test_case.expected_output["parameterType"]
    assert result.feasible_space.list == test_case.expected_output["list"]
    print("test execution complete")


def test_distribution_enum_values():
    """Test that Distribution enum has correct string values."""
    print("Executing test: distribution enum values")
    assert Distribution.UNIFORM.value == "uniform"
    assert Distribution.LOG_UNIFORM.value == "logUniform"
    print("test execution complete")


def test_continuous_search_space():
    """Test ContinuousSearchSpace dataclass construction."""
    print("Executing test: continuous search space")
    space = ContinuousSearchSpace(min=0.0, max=1.0, distribution=Distribution.UNIFORM)
    assert space.min == 0.0
    assert space.max == 1.0
    assert space.distribution == Distribution.UNIFORM
    print("test execution complete")


def test_categorical_search_space():
    """Test CategoricalSearchSpace dataclass construction."""
    print("Executing test: categorical search space")
    space = CategoricalSearchSpace(choices=["a", "b", "c"])
    assert space.choices == ["a", "b", "c"]
    print("test execution complete")
