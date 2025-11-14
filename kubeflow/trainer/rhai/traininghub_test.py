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

"""Unit tests for RHAI TrainingHub trainer builder."""

from typing import Optional

from kubeflow_trainer_api import models
import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai import TrainingHubAlgorithms, TrainingHubTrainer
from kubeflow.trainer.rhai.traininghub import get_trainer_cr_from_training_hub_trainer
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


def create_runtime_type(name: str) -> types.Runtime:
    """Create a minimal Runtime with Torch command to simulate distributed launch."""
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework=name,
        num_nodes=2,
        device="gpu",
        device_count="4",
        image="example.com/test-runtime",
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return types.Runtime(name=name, trainer=trainer)


def get_expected_resources_per_node(gpu_count: int) -> models.IoK8sApiCoreV1ResourceRequirements:
    return models.IoK8sApiCoreV1ResourceRequirements(
        requests={"nvidia.com/gpu": models.IoK8sApimachineryPkgApiResourceQuantity(str(gpu_count))},
        limits={"nvidia.com/gpu": models.IoK8sApimachineryPkgApiResourceQuantity(str(gpu_count))},
    )


def _simple_training_fn():
    # Simple function to allow inspect.getsource to work
    return "ok"


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="requires algorithm when func is None",
            expected_status=FAILED,
            config={
                "runtime": create_runtime_type(name="torch"),
                "trainer": TrainingHubTrainer(
                    func=None,
                    func_args={"nnodes": 1, "nproc_per_node": 1},
                    packages_to_install=None,
                    algorithm=None,
                ),
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="builds CRD for SFT algorithm wrapper",
            expected_status=SUCCESS,
            config={
                "runtime": create_runtime_type(name="torch"),
                "trainer": TrainingHubTrainer(
                    func=None,
                    func_args={"nnodes": 2, "nproc_per_node": 2, "data_path": "/data/file.json"},
                    packages_to_install=["training_hub"],
                    algorithm=TrainingHubAlgorithms.SFT,
                ),
            },
            expected_output=models.TrainerV1alpha1Trainer(
                numNodes=2,
                resourcesPerNode=get_expected_resources_per_node(2),
                command=["bash", "-c"],
            ),
        ),
        TestCase(
            name="embeds user function and maps env",
            expected_status=SUCCESS,
            config={
                "runtime": create_runtime_type(name="torch"),
                "trainer": TrainingHubTrainer(
                    func=_simple_training_fn,
                    func_args={"param": 1},
                    packages_to_install=["some_pkg"],
                    algorithm=None,
                    env={"A": "1", "B": "two"},
                ),
            },
            expected_output=models.TrainerV1alpha1Trainer(
                numNodes=1,
                resourcesPerNode=get_expected_resources_per_node(1),
                command=["bash", "-c"],
                env=[
                    models.IoK8sApiCoreV1EnvVar(name="A", value="1"),
                    models.IoK8sApiCoreV1EnvVar(name="B", value="two"),
                ],
            ),
        ),
    ],
)
def test_traininghub_builder(test_case):
    """Test TrainingHub CRD builder for both algorithm and user function flows."""
    print("Executing test:", test_case.name)

    try:
        runtime: types.Runtime = test_case.config["runtime"]
        trainer: TrainingHubTrainer = test_case.config["trainer"]

        crd = get_trainer_cr_from_training_hub_trainer(runtime=runtime, trainer=trainer)

        assert test_case.expected_status == SUCCESS
        assert isinstance(crd, models.TrainerV1alpha1Trainer)

        # Validate baseline fields if provided in expected_output
        exp: Optional[models.TrainerV1alpha1Trainer] = test_case.expected_output
        if exp is not None:
            if exp.num_nodes is not None:
                assert crd.num_nodes == exp.num_nodes
            if exp.resources_per_node is not None:
                # Compare gpu quantities
                assert crd.resources_per_node is not None
                assert crd.resources_per_node.limits is not None
                assert (
                    crd.resources_per_node.limits.get("nvidia.com/gpu").actual_instance
                    == exp.resources_per_node.limits.get("nvidia.com/gpu").actual_instance  # type: ignore
                )
            if exp.command is not None:
                assert crd.command == exp.command

        # Specific content checks
        if trainer.func is None:
            # Algorithm wrapper path; ensure algorithm import appears
            assert crd.args is not None and len(crd.args) == 1
            script = crd.args[0]
            algo_name = trainer.algorithm.value if trainer.algorithm else ""
            assert f"from training_hub import {algo_name}" in script
            assert "training_script.py" in script
        else:
            # User function path; ensure function name appears in generated script
            assert crd.args is not None and len(crd.args) == 1
            script = crd.args[0]
            assert "_simple_training_fn" in script

        # Env mapping check (if env was provided)
        if trainer.env:
            # Compare (name, value) pairs for stability
            actual_env = sorted([(e.name, e.value) for e in crd.env])  # type: ignore
            expected_env = sorted([(e.name, e.value) for e in test_case.expected_output.env])  # type: ignore
            assert actual_env == expected_env

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def _gpu_quantity(resource_reqs: models.IoK8sApiCoreV1ResourceRequirements) -> Optional[int]:
    if not resource_reqs or not resource_reqs.requests:
        return None
    qty = resource_reqs.requests.get(constants.GPU_LABEL)
    if not qty:
        return None
    return qty.actual_instance  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="algorithm wrapper builds CRD with nnodes and nproc correctly",
            expected_status=SUCCESS,
            config={
                "trainer": TrainingHubTrainer(
                    func=None,
                    func_args={"nnodes": 2, "nproc_per_node": 2, "data_path": "/data/file.json"},
                    packages_to_install=["training_hub"],
                    pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
                    algorithm=TrainingHubAlgorithms.SFT,
                    env={"A": "1", "B": "2"},
                ),
            },
        ),
        TestCase(
            name="missing algorithm when func is None raises ValueError",
            expected_status=FAILED,
            config={
                "trainer": TrainingHubTrainer(
                    func=None,
                    func_args={"nnodes": 1, "nproc_per_node": 1},
                    packages_to_install=None,
                    algorithm=None,
                ),
            },
            expected_error=ValueError,
        ),
    ],
)
def test_traininghub_algorithm_crd(test_case):
    print("Executing test:", test_case.name)
    runtime = create_runtime_type("torch")
    trainer_cfg: TrainingHubTrainer = test_case.config["trainer"]

    try:
        crd = get_trainer_cr_from_training_hub_trainer(runtime, trainer_cfg)

        assert test_case.expected_status == SUCCESS
        # Validate topology mapping.
        nnodes_expected = trainer_cfg.func_args.get("nnodes", 1) if trainer_cfg.func_args else 1
        nproc_expected = (
            trainer_cfg.func_args.get("nproc_per_node", 1) if trainer_cfg.func_args else 1
        )
        assert crd.num_nodes == nnodes_expected
        assert _gpu_quantity(crd.resources_per_node) == str(nproc_expected)

        # Validate command/args structure.
        assert crd.command == ["bash", "-c"]
        assert isinstance(crd.args, list) and len(crd.args) == 1 and isinstance(crd.args[0], str)
        args_str = crd.args[0]

        # Algorithm wrapper should import training_hub.<algo>
        if trainer_cfg.algorithm:
            assert f"from training_hub import {trainer_cfg.algorithm.value}" in args_str
            # PIP install header if packages requested.
            if trainer_cfg.packages_to_install:
                assert "pip install" in args_str

        # Validate env mapping.
        if trainer_cfg.env:
            assert crd.env is not None
            env_dict = {e.name: e.value for e in crd.env}
            assert env_dict == trainer_cfg.env

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def _sample_train_func():
    print("Hello from user func")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="user function embedding builds CRD and includes function and args",
            expected_status=SUCCESS,
            config={
                "trainer": TrainingHubTrainer(
                    func=_sample_train_func,
                    func_args={"epochs": 3, "lr": 0.1},
                    packages_to_install=["training_hub"],
                    pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
                ),
            },
        ),
    ],
)
def test_traininghub_user_func_crd(test_case):
    print("Executing test:", test_case.name)
    runtime = create_runtime_type("torch")
    trainer_cfg: TrainingHubTrainer = test_case.config["trainer"]

    try:
        crd = get_trainer_cr_from_training_hub_trainer(runtime, trainer_cfg)

        assert test_case.expected_status == SUCCESS
        # Topology defaulted to 1/1 if not specified.
        assert crd.num_nodes == 1
        assert _gpu_quantity(crd.resources_per_node) == "1"

        # Validate command/args
        assert crd.command == ["bash", "-c"]
        assert isinstance(crd.args, list) and len(crd.args) == 1 and isinstance(crd.args[0], str)
        args_str = crd.args[0]

        # Must contain the user function source identifier and provided kwargs
        assert "_sample_train_func" in args_str
        assert "epochs" in args_str and "3" in args_str
        assert "lr" in args_str and "0.1" in args_str

        # PIP install header if packages requested.
        if trainer_cfg.packages_to_install:
            assert "pip install" in args_str

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
