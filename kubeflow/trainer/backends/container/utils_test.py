# Copyright The Kubeflow Authors.
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

"""Unit tests for Container backend utility functions."""

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.common.constants import UNKNOWN
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.backends.container.utils import (
    aggregate_status_from_containers,
    build_environment,
    build_pip_install_cmd,
    container_status_to_trainjob_status,
    create_workdir,
    get_container_status,
    get_dataset_initializer,
    get_model_initializer,
    get_optional_initializer_envs,
    get_training_script_code,
    maybe_pull_image,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


# -------------------------------------------------------
# create_workdir
# -------------------------------------------------------
def test_create_workdir(tmp_path):
    """Test that create_workdir creates the expected directory."""
    print("Executing test: create_workdir creates directory")

    with patch("kubeflow.trainer.backends.container.utils.Path.home", return_value=tmp_path):
        result = create_workdir("my-job")

    expected = str((tmp_path / ".kubeflow" / "trainer" / "containers" / "my-job").resolve())
    assert result == expected
    assert (tmp_path / ".kubeflow" / "trainer" / "containers" / "my-job").is_dir()

    print("test execution complete")


# -------------------------------------------------------
# get_training_script_code
# -------------------------------------------------------
def _sample_train_func():
    x = 1 + 1
    return x


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="training script without func_args",
            expected_status=SUCCESS,
            config={"func_args": None},
            expected_output="_sample_train_func()",
        ),
        TestCase(
            name="training script with func_args",
            expected_status=SUCCESS,
            config={"func_args": {"lr": 0.01}},
            expected_output="_sample_train_func(**{'lr': 0.01})",
        ),
    ],
)
def test_get_training_script_code(test_case: TestCase):
    """Test training script code generation."""
    print(f"Executing test: {test_case.name}")

    trainer = types.CustomTrainer(
        func=_sample_train_func,
        func_args=test_case.config["func_args"],
    )
    code = get_training_script_code(trainer)

    assert "def _sample_train_func():" in code
    assert test_case.expected_output in code

    print("test execution complete")


# -------------------------------------------------------
# build_environment
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="build environment with env vars",
            expected_status=SUCCESS,
            config={"env": {"MY_VAR": "val1", "OTHER": "val2"}},
            expected_output={"MY_VAR": "val1", "OTHER": "val2"},
        ),
        TestCase(
            name="build environment without env",
            expected_status=SUCCESS,
            config={"env": None},
            expected_output={},
        ),
    ],
)
def test_build_environment(test_case: TestCase):
    """Test environment variable construction from trainer config."""
    print(f"Executing test: {test_case.name}")

    trainer = types.CustomTrainer(
        func=_sample_train_func,
        env=test_case.config["env"],
    )
    result = build_environment(trainer)
    assert result == test_case.expected_output

    print("test execution complete")


# -------------------------------------------------------
# build_pip_install_cmd
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="pip install with packages",
            expected_status=SUCCESS,
            config={"packages": ["torch", "numpy"]},
            expected_output=(
                "PIP_DISABLE_PIP_VERSION_CHECK=1 pip install "
                "--no-warn-script-location "
                "--index-url https://pypi.org/simple  "
                '"torch" "numpy" && '
            ),
        ),
        TestCase(
            name="pip install without packages",
            expected_status=SUCCESS,
            config={"packages": None},
            expected_output="",
        ),
        TestCase(
            name="pip install with index urls",
            expected_status=SUCCESS,
            config={
                "packages": ["transformers"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://download.pytorch.org/whl/cpu",
                ],
            },
            expected_output=(
                "PIP_DISABLE_PIP_VERSION_CHECK=1 pip install "
                "--no-warn-script-location "
                "--index-url https://pypi.org/simple "
                "--extra-index-url https://download.pytorch.org/whl/cpu "
                '"transformers" && '
            ),
        ),
    ],
)
def test_build_pip_install_cmd(test_case: TestCase):
    """Test pip install command generation."""
    print(f"Executing test: {test_case.name}")

    kwargs = {"func": _sample_train_func, "packages_to_install": test_case.config["packages"]}
    if "pip_index_urls" in test_case.config:
        kwargs["pip_index_urls"] = test_case.config["pip_index_urls"]
    trainer = types.CustomTrainer(**kwargs)

    result = build_pip_install_cmd(trainer)
    assert result == test_case.expected_output

    print("test execution complete")


# -------------------------------------------------------
# container_status_to_trainjob_status
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="running container",
            expected_status=SUCCESS,
            config={"status": "running", "exit_code": 0},
            expected_output=constants.TRAINJOB_RUNNING,
        ),
        TestCase(
            name="created container",
            expected_status=SUCCESS,
            config={"status": "created", "exit_code": 0},
            expected_output=constants.TRAINJOB_CREATED,
        ),
        TestCase(
            name="exited container with exit code 0",
            expected_status=SUCCESS,
            config={"status": "exited", "exit_code": 0},
            expected_output=constants.TRAINJOB_COMPLETE,
        ),
        TestCase(
            name="exited container with exit code 1",
            expected_status=SUCCESS,
            config={"status": "exited", "exit_code": 1},
            expected_output=constants.TRAINJOB_FAILED,
        ),
        TestCase(
            name="unknown container status",
            expected_status=SUCCESS,
            config={"status": "paused", "exit_code": 0},
            expected_output=UNKNOWN,
        ),
    ],
)
def test_container_status_to_trainjob_status(test_case: TestCase):
    """Test mapping of container status to TrainJob status."""
    print(f"Executing test: {test_case.name}")

    result = container_status_to_trainjob_status(
        test_case.config["status"],
        test_case.config["exit_code"],
    )
    assert result == test_case.expected_output

    print("test execution complete")


# -------------------------------------------------------
# aggregate_status_from_containers
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="failed takes priority",
            expected_status=SUCCESS,
            config={
                "statuses": [
                    constants.TRAINJOB_RUNNING,
                    constants.TRAINJOB_FAILED,
                    constants.TRAINJOB_COMPLETE,
                ],
            },
            expected_output=constants.TRAINJOB_FAILED,
        ),
        TestCase(
            name="running takes priority over complete",
            expected_status=SUCCESS,
            config={
                "statuses": [
                    constants.TRAINJOB_RUNNING,
                    constants.TRAINJOB_COMPLETE,
                ],
            },
            expected_output=constants.TRAINJOB_RUNNING,
        ),
        TestCase(
            name="all complete",
            expected_status=SUCCESS,
            config={
                "statuses": [
                    constants.TRAINJOB_COMPLETE,
                    constants.TRAINJOB_COMPLETE,
                ],
            },
            expected_output=constants.TRAINJOB_COMPLETE,
        ),
        TestCase(
            name="created present",
            expected_status=SUCCESS,
            config={
                "statuses": [
                    constants.TRAINJOB_CREATED,
                    constants.TRAINJOB_COMPLETE,
                ],
            },
            expected_output=constants.TRAINJOB_CREATED,
        ),
    ],
)
def test_aggregate_status_from_containers(test_case: TestCase):
    """Test aggregation of multiple container statuses."""
    print(f"Executing test: {test_case.name}")

    result = aggregate_status_from_containers(test_case.config["statuses"])
    assert result == test_case.expected_output

    print("test execution complete")


# -------------------------------------------------------
# maybe_pull_image
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="IfNotPresent when image exists",
            expected_status=SUCCESS,
            config={
                "policy": "IfNotPresent",
                "image_exists": True,
            },
            expected_output="no_pull",
        ),
        TestCase(
            name="IfNotPresent when image missing",
            expected_status=SUCCESS,
            config={
                "policy": "IfNotPresent",
                "image_exists": False,
            },
            expected_output="pull",
        ),
        TestCase(
            name="Always pulls regardless",
            expected_status=SUCCESS,
            config={
                "policy": "Always",
                "image_exists": True,
            },
            expected_output="pull",
        ),
        TestCase(
            name="Never when image exists",
            expected_status=SUCCESS,
            config={
                "policy": "Never",
                "image_exists": True,
            },
            expected_output="no_pull",
        ),
        TestCase(
            name="Never when image missing raises",
            expected_status=FAILED,
            config={
                "policy": "Never",
                "image_exists": False,
            },
            expected_error=RuntimeError,
        ),
    ],
)
def test_maybe_pull_image(test_case: TestCase):
    """Test image pulling behavior for different pull policies."""
    print(f"Executing test: {test_case.name}")

    adapter = MagicMock()
    adapter.image_exists.return_value = test_case.config["image_exists"]
    image = "test-image:latest"

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            maybe_pull_image(adapter, image, test_case.config["policy"])
    else:
        maybe_pull_image(adapter, image, test_case.config["policy"])
        if test_case.expected_output == "pull":
            adapter.pull_image.assert_called_once_with(image)
        else:
            adapter.pull_image.assert_not_called()

    print("test execution complete")


# -------------------------------------------------------
# get_container_status
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_container_status success",
            expected_status=SUCCESS,
            config={"status": "running", "exit_code": 0},
            expected_output=constants.TRAINJOB_RUNNING,
        ),
        TestCase(
            name="get_container_status error returns UNKNOWN",
            expected_status=SUCCESS,
            config={"raise_error": True},
            expected_output=UNKNOWN,
        ),
    ],
)
def test_get_container_status(test_case: TestCase):
    """Test container status retrieval with success and error paths."""
    print(f"Executing test: {test_case.name}")

    adapter = MagicMock()
    if test_case.config.get("raise_error"):
        adapter.container_status.side_effect = Exception("connection lost")
    else:
        adapter.container_status.return_value = (
            test_case.config["status"],
            test_case.config["exit_code"],
        )

    result = get_container_status(adapter, "container-abc")
    assert result == test_case.expected_output

    print("test execution complete")


# -------------------------------------------------------
# get_optional_initializer_envs
# -------------------------------------------------------
def test_get_optional_initializer_envs():
    """Test extraction of optional fields as environment variables."""
    print("Executing test: get_optional_initializer_envs")

    init = types.HuggingFaceDatasetInitializer(
        storage_uri="hf://user/dataset",
        access_token="tok123",
    )
    env = get_optional_initializer_envs(init, required_fields={"storage_uri"})
    assert "ACCESS_TOKEN" in env
    assert env["ACCESS_TOKEN"] == "tok123"
    assert "STORAGE_URI" not in env

    print("test execution complete")


# -------------------------------------------------------
# get_dataset_initializer
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="dataset initializer with HuggingFace",
            expected_status=SUCCESS,
            config={
                "dataset": types.HuggingFaceDatasetInitializer(
                    storage_uri="hf://user/my-dataset",
                ),
            },
            expected_output={
                "name": "dataset-initializer",
                "env_storage_uri": "hf://user/my-dataset",
                "env_output_path": constants.DATASET_PATH,
            },
        ),
        TestCase(
            name="dataset initializer with S3",
            expected_status=SUCCESS,
            config={
                "dataset": types.S3DatasetInitializer(
                    storage_uri="s3://bucket/data",
                    endpoint="https://s3.example.com",
                ),
            },
            expected_output={
                "name": "dataset-initializer",
                "env_storage_uri": "s3://bucket/data",
                "env_output_path": constants.DATASET_PATH,
            },
        ),
        TestCase(
            name="unsupported dataset initializer type",
            expected_status=FAILED,
            config={"dataset": "not-an-initializer"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_dataset_initializer(test_case: TestCase):
    """Test dataset initializer container config generation."""
    print(f"Executing test: {test_case.name}")

    cfg = ContainerBackendConfig()

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            get_dataset_initializer(test_case.config["dataset"], cfg)
    else:
        result = get_dataset_initializer(test_case.config["dataset"], cfg)
        assert result.name == test_case.expected_output["name"]
        assert result.env["STORAGE_URI"] == test_case.expected_output["env_storage_uri"]
        assert result.env["OUTPUT_PATH"] == test_case.expected_output["env_output_path"]
        assert result.image == cfg.dataset_initializer_image

    print("test execution complete")


# -------------------------------------------------------
# get_model_initializer
# -------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="model initializer with HuggingFace",
            expected_status=SUCCESS,
            config={
                "model": types.HuggingFaceModelInitializer(
                    storage_uri="hf://user/my-model",
                ),
            },
            expected_output={
                "name": "model-initializer",
                "env_storage_uri": "hf://user/my-model",
                "env_output_path": constants.MODEL_PATH,
            },
        ),
        TestCase(
            name="model initializer with S3",
            expected_status=SUCCESS,
            config={
                "model": types.S3ModelInitializer(
                    storage_uri="s3://bucket/model",
                    endpoint="https://s3.example.com",
                ),
            },
            expected_output={
                "name": "model-initializer",
                "env_storage_uri": "s3://bucket/model",
                "env_output_path": constants.MODEL_PATH,
            },
        ),
        TestCase(
            name="unsupported model initializer type",
            expected_status=FAILED,
            config={"model": "not-an-initializer"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_model_initializer(test_case: TestCase):
    """Test model initializer container config generation."""
    print(f"Executing test: {test_case.name}")

    cfg = ContainerBackendConfig()

    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            get_model_initializer(test_case.config["model"], cfg)
    else:
        result = get_model_initializer(test_case.config["model"], cfg)
        assert result.name == test_case.expected_output["name"]
        assert result.env["STORAGE_URI"] == test_case.expected_output["env_storage_uri"]
        assert result.env["OUTPUT_PATH"] == test_case.expected_output["env_output_path"]
        assert result.image == cfg.model_initializer_image

    print("test execution complete")
