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

from pathlib import Path
from unittest.mock import patch

import pytest

from kubeflow.trainer.backends.localprocess import constants as local_exec_constants
from kubeflow.trainer.backends.localprocess.types import LocalRuntimeTrainer
from kubeflow.trainer.backends.localprocess.utils import (
    _canonicalize_name,
    _extract_name,
    get_cleanup_venv_script,
    get_command_using_train_func,
    get_dependencies_command,
    get_install_packages,
    get_local_runtime_trainer,
    get_local_train_job_script,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


# ---------------------------------------------------------------------------
# _extract_name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="simple package name",
            expected_status=SUCCESS,
            config={"requirement": "numpy"},
            expected_output="numpy",
        ),
        TestCase(
            name="package with version specifier",
            expected_status=SUCCESS,
            config={"requirement": "numpy==1.21"},
            expected_output="numpy",
        ),
        TestCase(
            name="package with extras",
            expected_status=SUCCESS,
            config={"requirement": "package[extra1,extra2]"},
            expected_output="package",
        ),
        TestCase(
            name="package with URL",
            expected_status=SUCCESS,
            config={"requirement": "package @ https://example.com/pkg.whl"},
            expected_output="package",
        ),
        TestCase(
            name="None input raises ValueError",
            expected_status=FAILED,
            config={"requirement": None},
            expected_error=ValueError,
        ),
        TestCase(
            name="empty string raises ValueError",
            expected_status=FAILED,
            config={"requirement": ""},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid format raises ValueError",
            expected_status=FAILED,
            config={"requirement": "!!!invalid"},
            expected_error=ValueError,
        ),
    ],
)
def test_extract_name(test_case: TestCase):
    """Test _extract_name with various requirement string formats."""
    print(f"Executing test: {test_case.name}")
    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            _extract_name(test_case.config["requirement"])
    else:
        result = _extract_name(test_case.config["requirement"])
        assert result == test_case.expected_output
    print("test execution complete")


# ---------------------------------------------------------------------------
# _canonicalize_name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="lowercase conversion",
            expected_status=SUCCESS,
            config={"name": "NumPy"},
            expected_output="numpy",
        ),
        TestCase(
            name="underscore collapsed to dash",
            expected_status=SUCCESS,
            config={"name": "my_package"},
            expected_output="my-package",
        ),
        TestCase(
            name="dot collapsed to dash",
            expected_status=SUCCESS,
            config={"name": "my.package"},
            expected_output="my-package",
        ),
    ],
)
def test_canonicalize_name(test_case: TestCase):
    """Test _canonicalize_name PEP 503 normalization."""
    print(f"Executing test: {test_case.name}")
    result = _canonicalize_name(test_case.config["name"])
    assert result == test_case.expected_output
    print("test execution complete")


# ---------------------------------------------------------------------------
# get_install_packages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="no trainer packages returns runtime unchanged",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["numpy==1.21", "pandas"],
                "trainer_packages": None,
            },
            expected_output=["numpy==1.21", "pandas"],
        ),
        TestCase(
            name="trainer overwrites matching runtime package",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["numpy==1.21", "pandas"],
                "trainer_packages": ["NumPy==1.25"],
            },
            expected_output=["pandas", "NumPy==1.25"],
        ),
        TestCase(
            name="duplicate trainer packages raises ValueError",
            expected_status=FAILED,
            config={
                "runtime_packages": ["numpy"],
                "trainer_packages": ["torch==2.0", "Torch==2.1"],
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="runtime-only first then trainer ordering preserved",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["numpy", "scipy", "pandas"],
                "trainer_packages": ["scipy==1.12"],
            },
            expected_output=["numpy", "pandas", "scipy==1.12"],
        ),
        TestCase(
            name="empty runtime with trainer packages",
            expected_status=SUCCESS,
            config={
                "runtime_packages": [],
                "trainer_packages": ["torch==2.0"],
            },
            expected_output=["torch==2.0"],
        ),
    ],
)
def test_get_install_packages(test_case: TestCase):
    """Test get_install_packages merge logic."""
    print(f"Executing test: {test_case.name}")
    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            get_install_packages(**test_case.config)
    else:
        result = get_install_packages(**test_case.config)
        assert result == test_case.expected_output
    print("test execution complete")


# ---------------------------------------------------------------------------
# get_local_runtime_trainer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid default runtime name",
            expected_status=SUCCESS,
            config={
                "runtime_name": constants.DEFAULT_TRAINING_RUNTIME,
                "framework": local_exec_constants.TORCH_FRAMEWORK_TYPE,
            },
        ),
        TestCase(
            name="invalid runtime name raises ValueError",
            expected_status=FAILED,
            config={
                "runtime_name": "nonexistent-runtime",
                "framework": "torch",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="torch framework sets torchrun command",
            expected_status=SUCCESS,
            config={
                "runtime_name": constants.DEFAULT_TRAINING_RUNTIME,
                "framework": local_exec_constants.TORCH_FRAMEWORK_TYPE,
            },
        ),
    ],
)
def test_get_local_runtime_trainer(test_case: TestCase, tmp_path: Path):
    """Test get_local_runtime_trainer runtime lookup and command setting."""
    print(f"Executing test: {test_case.name}")
    venv_dir = str(tmp_path / "venv")
    if test_case.expected_status == FAILED:
        with pytest.raises(test_case.expected_error):
            get_local_runtime_trainer(
                runtime_name=test_case.config["runtime_name"],
                venv_dir=venv_dir,
                framework=test_case.config["framework"],
            )
    else:
        result = get_local_runtime_trainer(
            runtime_name=test_case.config["runtime_name"],
            venv_dir=venv_dir,
            framework=test_case.config["framework"],
        )
        assert isinstance(result, LocalRuntimeTrainer)
        assert result.image == local_exec_constants.LOCAL_RUNTIME_IMAGE

        venv_bin = str(Path(venv_dir) / "bin")
        if test_case.config["framework"] == local_exec_constants.TORCH_FRAMEWORK_TYPE:
            expected_cmd = str(Path(venv_bin) / local_exec_constants.TORCH_COMMAND)
            assert result.command == (expected_cmd,)
        else:
            expected_cmd = str(Path(venv_bin) / local_exec_constants.DEFAULT_COMMAND)
            assert result.command == (expected_cmd,)
    print("test execution complete")


# ---------------------------------------------------------------------------
# get_dependencies_command
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="basic with packages",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["numpy", "pandas"],
                "pip_index_urls": ["https://pypi.org/simple"],
                "trainer_packages": [],
            },
        ),
        TestCase(
            name="with extra index URLs",
            expected_status=SUCCESS,
            config={
                "runtime_packages": ["torch"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://download.pytorch.org/whl/cpu",
                ],
                "trainer_packages": [],
            },
        ),
    ],
)
def test_get_dependencies_command(test_case: TestCase):
    """Test get_dependencies_command pip install string construction."""
    print(f"Executing test: {test_case.name}")
    result = get_dependencies_command(**test_case.config)
    assert "pip install" in result
    assert "--index-url" in result

    for pkg in test_case.config["runtime_packages"]:
        assert pkg in result, f"Expected package '{pkg}' in command output"

    pip_urls = test_case.config["pip_index_urls"]
    assert pip_urls[0] in result
    if len(pip_urls) > 1:
        assert "--extra-index-url" in result
        assert pip_urls[1] in result
    print("test execution complete")


# ---------------------------------------------------------------------------
# get_command_using_train_func
# ---------------------------------------------------------------------------
def _sample_train_func(parameters=None):
    print("training")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="with parameters",
            expected_status=SUCCESS,
            config={
                "train_func_parameters": {"lr": 0.01},
            },
        ),
        TestCase(
            name="without parameters",
            expected_status=SUCCESS,
            config={
                "train_func_parameters": None,
            },
        ),
        TestCase(
            name="non-callable raises ValueError",
            expected_status=FAILED,
            config={
                "train_func": "not_a_function",
                "train_func_parameters": None,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="runtime without trainer raises ValueError",
            expected_status=FAILED,
            config={
                "no_trainer": True,
                "train_func_parameters": None,
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_command_using_train_func(test_case: TestCase, tmp_path: Path):
    """Test get_command_using_train_func entrypoint generation."""
    print(f"Executing test: {test_case.name}")

    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    venv_dir = str(venv_dir)
    train_job_name = "test-job"

    if test_case.config.get("no_trainer"):
        runtime = types.Runtime(name="test", trainer=None)
        with pytest.raises(test_case.expected_error):
            get_command_using_train_func(
                runtime=runtime,
                train_func=_sample_train_func,
                train_func_parameters=test_case.config["train_func_parameters"],
                venv_dir=venv_dir,
                train_job_name=train_job_name,
            )
    elif test_case.expected_status == FAILED:
        rt = LocalRuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="local",
        )
        rt.set_command(("python",))
        runtime = types.Runtime(name="test", trainer=rt)
        with pytest.raises(test_case.expected_error):
            get_command_using_train_func(
                runtime=runtime,
                train_func=test_case.config["train_func"],
                train_func_parameters=test_case.config["train_func_parameters"],
                venv_dir=venv_dir,
                train_job_name=train_job_name,
            )
    else:
        rt = LocalRuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="local",
        )
        rt.set_command(("python",))
        runtime = types.Runtime(name="test", trainer=rt)
        result = get_command_using_train_func(
            runtime=runtime,
            train_func=_sample_train_func,
            train_func_parameters=test_case.config["train_func_parameters"],
            venv_dir=venv_dir,
            train_job_name=train_job_name,
        )
        assert isinstance(result, str)
        func_file = (
            tmp_path / "venv" / local_exec_constants.LOCAL_EXEC_FILENAME.format(train_job_name)
        )
        assert func_file.exists()
    print("test execution complete")


# ---------------------------------------------------------------------------
# get_cleanup_venv_script
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="cleanup_venv true returns script with venv path",
            expected_status=SUCCESS,
            config={"cleanup_venv": True},
        ),
        TestCase(
            name="cleanup_venv false returns newline",
            expected_status=SUCCESS,
            config={"cleanup_venv": False},
            expected_output="\n",
        ),
    ],
)
def test_get_cleanup_venv_script(test_case: TestCase, tmp_path: Path):
    """Test get_cleanup_venv_script conditional cleanup logic."""
    print(f"Executing test: {test_case.name}")
    venv_dir = str(tmp_path / "venv")
    result = get_cleanup_venv_script(
        venv_dir=venv_dir, cleanup_venv=test_case.config["cleanup_venv"]
    )
    if not test_case.config["cleanup_venv"]:
        assert result == test_case.expected_output
    else:
        assert venv_dir in result
        assert "rm -rf" in result
    print("test execution complete")


# ---------------------------------------------------------------------------
# get_local_train_job_script
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid inputs return 3-tuple",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="invalid runtime trainer type raises ValueError",
            expected_status=FAILED,
            config={"use_base_runtime_trainer": True},
            expected_error=ValueError,
        ),
    ],
)
def test_get_local_train_job_script(test_case: TestCase, tmp_path: Path):
    """Test get_local_train_job_script full script generation."""
    print(f"Executing test: {test_case.name}")

    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    venv_dir = str(venv_dir)
    train_job_name = "test-job"

    if test_case.config.get("use_base_runtime_trainer"):
        base_rt = types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="local",
        )
        base_rt.set_command(("python",))
        runtime = types.Runtime(name="test", trainer=base_rt)
        trainer = types.CustomTrainer(
            func=_sample_train_func,
            packages_to_install=["numpy"],
        )
        with (
            pytest.raises(test_case.expected_error),
            patch(
                "kubeflow.trainer.backends.localprocess.utils.shutil.which",
                return_value="/usr/bin/python",
            ),
        ):
            get_local_train_job_script(
                train_job_name=train_job_name,
                venv_dir=venv_dir,
                trainer=trainer,
                runtime=runtime,
            )
    else:
        rt = LocalRuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="local",
            packages=["torch"],
        )
        rt.set_command(("python",))
        runtime = types.Runtime(name="test", trainer=rt)
        trainer = types.CustomTrainer(
            func=_sample_train_func,
            packages_to_install=["numpy"],
        )

        with patch(
            "kubeflow.trainer.backends.localprocess.utils.shutil.which",
            return_value="/usr/bin/python",
        ):
            result = get_local_train_job_script(
                train_job_name=train_job_name,
                venv_dir=venv_dir,
                trainer=trainer,
                runtime=runtime,
            )
        assert len(result) == 3
        assert result[0] == "bash"
        assert result[1] == "-c"
        assert isinstance(result[2], str)
    print("test execution complete")
