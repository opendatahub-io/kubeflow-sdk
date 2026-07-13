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

"""Unit tests for kubeflow.optimizer.types.optimization_types module."""

from datetime import datetime

import pytest

import kubeflow.common.constants as common_constants
from kubeflow.optimizer.types.algorithm_types import GridSearch, RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Direction,
    Metric,
    Objective,
    OptimizationJob,
    Result,
    Trial,
    TrialConfig,
)
from kubeflow.optimizer.types.search_types import (
    CategoricalSearchSpace,
    ContinuousSearchSpace,
    Distribution as SearchDistribution,
)
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types.types import Runtime, RuntimeTrainer, TrainerType, TrainJob

# --- Direction enum ---


def test_direction_enum_values():
    """Test that Direction enum has correct string values."""
    print("Executing test: direction enum values")
    assert Direction.MAXIMIZE.value == "maximize"
    assert Direction.MINIMIZE.value == "minimize"
    print("test execution complete")


def test_direction_enum_from_value():
    """Test that Direction enum can be constructed from string values."""
    print("Executing test: direction enum from value")
    assert Direction("maximize") is Direction.MAXIMIZE
    assert Direction("minimize") is Direction.MINIMIZE
    print("test execution complete")


def test_direction_enum_invalid_value():
    """Test that Direction enum raises ValueError for unknown strings."""
    print("Executing test: direction enum invalid value")
    with pytest.raises(ValueError):
        Direction("invalid")
    print("test execution complete")


# --- Objective ---


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default objective",
            expected_status=SUCCESS,
            expected_output={"metric": "loss", "direction": Direction.MINIMIZE},
        ),
        TestCase(
            name="custom objective with enum direction",
            expected_status=SUCCESS,
            config={"metric": "accuracy", "direction": Direction.MAXIMIZE},
            expected_output={"metric": "accuracy", "direction": Direction.MAXIMIZE},
        ),
        TestCase(
            name="objective with string direction coercion",
            expected_status=SUCCESS,
            config={"metric": "f1", "direction": "maximize"},
            expected_output={"metric": "f1", "direction": Direction.MAXIMIZE},
        ),
        TestCase(
            name="objective with invalid string direction",
            expected_status=FAILED,
            config={"metric": "loss", "direction": "unknown"},
            expected_error=ValueError,
        ),
    ],
)
def test_objective(test_case: TestCase):
    """Test Objective dataclass construction and direction coercion."""
    print("Executing test:", test_case.name)
    try:
        obj = Objective(**test_case.config) if test_case.config else Objective()
        assert test_case.expected_status == SUCCESS
        assert obj.metric == test_case.expected_output["metric"]
        assert obj.direction == test_case.expected_output["direction"]
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert isinstance(e, test_case.expected_error)
    print("test execution complete")


# --- TrialConfig ---


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default trial config",
            expected_status=SUCCESS,
            expected_output={"num_trials": 10, "parallel_trials": 1, "max_failed_trials": None},
        ),
        TestCase(
            name="custom trial config",
            expected_status=SUCCESS,
            config={"num_trials": 50, "parallel_trials": 5, "max_failed_trials": 3},
            expected_output={"num_trials": 50, "parallel_trials": 5, "max_failed_trials": 3},
        ),
    ],
)
def test_trial_config(test_case: TestCase):
    """Test TrialConfig dataclass defaults and custom values."""
    print("Executing test:", test_case.name)
    tc = TrialConfig(**test_case.config) if test_case.config else TrialConfig()
    assert tc.num_trials == test_case.expected_output["num_trials"]
    assert tc.parallel_trials == test_case.expected_output["parallel_trials"]
    assert tc.max_failed_trials == test_case.expected_output["max_failed_trials"]
    print("test execution complete")


# --- Metric ---


def test_metric_construction():
    """Test Metric dataclass stores all fields."""
    print("Executing test: metric construction")
    m = Metric(name="loss", min="0.01", max="0.99", latest="0.15")
    assert m.name == "loss"
    assert m.min == "0.01"
    assert m.max == "0.99"
    assert m.latest == "0.15"
    print("test execution complete")


# --- Result ---


def test_result_construction():
    """Test Result dataclass stores parameters and metrics."""
    print("Executing test: result construction")
    metrics = [Metric(name="loss", min="0.01", max="0.5", latest="0.1")]
    r = Result(parameters={"lr": "0.01", "epochs": "10"}, metrics=metrics)
    assert r.parameters == {"lr": "0.01", "epochs": "10"}
    assert len(r.metrics) == 1
    assert r.metrics[0].name == "loss"
    print("test execution complete")


# --- Trial ---


def _make_trainjob(name: str = "test-job", status: str = common_constants.UNKNOWN) -> TrainJob:
    """Build a minimal valid TrainJob for testing."""
    runtime = Runtime(
        name="test-runtime",
        trainer=RuntimeTrainer(
            trainer_type=TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="test:latest",
        ),
    )
    return TrainJob(
        name=name,
        runtime=runtime,
        steps=[],
        num_nodes=1,
        creation_timestamp=datetime(2026, 1, 1),
        status=status,
    )


def test_trial_default_metrics():
    """Test Trial defaults to empty metrics list."""
    print("Executing test: trial default metrics")
    trainjob = _make_trainjob()
    t = Trial(name="trial-1", parameters={"lr": "0.01"}, trainjob=trainjob)
    assert t.name == "trial-1"
    assert t.parameters == {"lr": "0.01"}
    assert t.metrics == []
    print("test execution complete")


def test_trial_with_metrics():
    """Test Trial with explicit metrics."""
    print("Executing test: trial with metrics")
    trainjob = _make_trainjob(status="Running")
    metrics = [Metric(name="accuracy", min="0.5", max="0.95", latest="0.9")]
    t = Trial(name="trial-2", parameters={"lr": "0.1"}, trainjob=trainjob, metrics=metrics)
    assert len(t.metrics) == 1
    assert t.metrics[0].latest == "0.9"
    print("test execution complete")


# --- OptimizationJob ---


def test_optimization_job_default_status():
    """Test OptimizationJob defaults status to UNKNOWN."""
    print("Executing test: optimization job default status")
    job = OptimizationJob(
        name="job-1",
        search_space={
            "lr": ContinuousSearchSpace(
                min=0.001, max=0.1, distribution=SearchDistribution.UNIFORM
            ),
        },
        objectives=[Objective()],
        algorithm=RandomSearch(),
        trial_config=TrialConfig(),
        trials=[],
        creation_timestamp=datetime(2026, 1, 1),
    )
    assert job.status == common_constants.UNKNOWN
    print("test execution complete")


def test_optimization_job_custom_status():
    """Test OptimizationJob with explicit status."""
    print("Executing test: optimization job custom status")
    job = OptimizationJob(
        name="job-2",
        search_space={"opt": CategoricalSearchSpace(choices=["adam", "sgd"])},
        objectives=[Objective(metric="accuracy", direction=Direction.MAXIMIZE)],
        algorithm=GridSearch(),
        trial_config=TrialConfig(num_trials=5),
        trials=[],
        creation_timestamp=datetime(2026, 7, 1),
        status="Running",
    )
    assert job.name == "job-2"
    assert job.status == "Running"
    assert job.algorithm.algorithm_name == "grid"
    assert job.trial_config.num_trials == 5
    print("test execution complete")
