# Copyright 2024 The Kubeflow Authors.
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

"""Unit tests for experimental Kubernetes backend."""

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.experimental.backends.kubernetes import ExperimentalKubernetesBackend
from kubeflow.trainer.types.experimental import TransformersTrainer
from kubeflow.trainer.types.types import CustomTrainer, Runtime


def dummy_train_func():
    """Dummy training function."""
    print("Training...")


@pytest.fixture
def mock_runtime():
    """Create a mock runtime for testing."""
    runtime = MagicMock(spec=Runtime)
    runtime.name = "pytorch-distributed"
    runtime.trainer = MagicMock()
    # Use command template with both placeholders - {func_file} triggers injection
    # and {func_code} gets the actual code
    runtime.trainer.command = ["bash", "-c", "python -c '{func_code}' {func_file}"]
    return runtime


def test_experimental_backend_initialization():
    """Test ExperimentalKubernetesBackend can be initialized."""
    with patch(
        "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
        return_value=None,
    ):
        backend = ExperimentalKubernetesBackend()
        assert backend is not None


def test_experimental_backend_with_config():
    """Test ExperimentalKubernetesBackend with custom config."""
    config = KubernetesBackendConfig(namespace="test-namespace")

    with patch(
        "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
        return_value=None,
    ):
        backend = ExperimentalKubernetesBackend(cfg=config)
        assert backend is not None


def test_train_with_custom_trainer_uses_parent():
    """Train with CustomTrainer should use parent backend."""
    with (
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
            return_value=None,
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.train",
            return_value="job-123",
        ) as mock_train,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = CustomTrainer(func=dummy_train_func)
        result = backend.train(trainer=trainer)

        assert result == "job-123"
        mock_train.assert_called_once()


def test_train_with_transformers_trainer_uses_instrumentation(mock_runtime):
    """Train with TransformersTrainer should use instrumentation."""
    with (
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
            return_value=None,
        ),
        patch.object(
            ExperimentalKubernetesBackend,
            "_train_transformers_instrumented",
            return_value="job-456",
        ) as mock_instrumented,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = TransformersTrainer(func=dummy_train_func)
        result = backend.train(runtime=mock_runtime, trainer=trainer)

        assert result == "job-456"
        mock_instrumented.assert_called_once()


def test_get_instrumented_command_generates_wrapper(mock_runtime):
    """Test _get_instrumented_command generates valid command."""
    with patch(
        "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
        return_value=None,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = TransformersTrainer(
            func=dummy_train_func,
            packages_to_install=["transformers"],
        )

        wrapper_script = "# Wrapper\n{{user_func_import_and_call}}\n"

        with patch(
            "kubeflow.trainer.backends.kubernetes.utils.get_script_for_python_packages",
            return_value="pip install",
        ):
            command = backend._get_instrumented_command(mock_runtime, trainer, wrapper_script)

            assert isinstance(command, list)
            assert len(command) > 0


def test_get_instrumented_command_injects_user_code(mock_runtime):
    """Test user code is properly injected into wrapper."""
    with patch(
        "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
        return_value=None,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = TransformersTrainer(func=dummy_train_func)
        wrapper_script = "WRAPPER_START\n{{user_func_import_and_call}}\nWRAPPER_END"

        command = backend._get_instrumented_command(mock_runtime, trainer, wrapper_script)

        # Debug: print command to see what we got
        print(f"\nGenerated command: {command}")

        # The command should have injected the wrapper code
        # Since we use {func_file} placeholder, code gets injected via format()
        assert len(command) > 0
        # Look for the wrapper markers in any command element
        command_str = " ".join(str(c) for c in command)
        assert "WRAPPER_START" in command_str
        assert "WRAPPER_END" in command_str
        assert "dummy_train_func" in command_str


def test_get_instrumented_command_with_func_args(mock_runtime):
    """Test function arguments are properly included."""
    with patch(
        "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
        return_value=None,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = TransformersTrainer(
            func=dummy_train_func,
            func_args={"batch_size": 32, "learning_rate": 0.001},
        )
        wrapper_script = "{{user_func_import_and_call}}"

        command = backend._get_instrumented_command(mock_runtime, trainer, wrapper_script)

        # Should include function call with kwargs
        exec_script = str(command)
        assert "batch_size" in exec_script or "{'batch_size': 32" in exec_script


def test_get_instrumented_command_no_runtime_trainer_raises():
    """Test error when runtime has no trainer."""
    with patch(
        "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
        return_value=None,
    ):
        backend = ExperimentalKubernetesBackend()

        runtime = MagicMock(spec=Runtime)
        runtime.trainer = None

        trainer = TransformersTrainer(func=dummy_train_func)
        wrapper_script = "{{user_func_import_and_call}}"

        with pytest.raises(ValueError, match="Runtime must have a trainer"):
            backend._get_instrumented_command(runtime, trainer, wrapper_script)


def test_get_instrumented_command_non_callable_func_raises(mock_runtime):
    """Test error when trainer func is not callable."""
    with patch(
        "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
        return_value=None,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = TransformersTrainer(func="not_callable")
        wrapper_script = "{{user_func_import_and_call}}"

        with pytest.raises(ValueError, match="must be callable"):
            backend._get_instrumented_command(mock_runtime, trainer, wrapper_script)


def test_train_transformers_adds_annotations(mock_runtime):
    """Test TransformersTrainer adds progression tracking annotations."""
    with (
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
            return_value=None,
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.train",
            return_value="job-789",
        ) as mock_parent_train,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = TransformersTrainer(
            func=dummy_train_func,
            metrics_port=9090,
        )

        with patch.object(backend, "get_runtime", return_value=mock_runtime):
            backend.train(trainer=trainer)

            # Check that annotations were added via options
            call_args = mock_parent_train.call_args
            options = call_args[1]["options"]

            # Should have annotations option
            assert options is not None
            assert len(options) > 0


def test_train_transformers_default_runtime(mock_runtime):
    """Test TransformersTrainer can train without runtime specified."""
    with (
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
            return_value=None,
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.train",
            return_value="job-default",
        ) as mock_parent_train,
    ):
        backend = ExperimentalKubernetesBackend()

        trainer = TransformersTrainer(func=dummy_train_func)
        result = backend.train(trainer=trainer)

        # Should successfully train and return job name
        assert result == "job-default"
        mock_parent_train.assert_called_once()


def test_instrumented_command_cleanup_on_error(mock_runtime):
    """Test _instrumented_command is cleaned up even on error."""
    with (
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
            return_value=None,
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.train",
            side_effect=Exception("Test error"),
        ),
    ):
        backend = ExperimentalKubernetesBackend()
        backend.get_runtime = MagicMock(return_value=mock_runtime)

        trainer = TransformersTrainer(func=dummy_train_func)

        with pytest.raises(Exception, match="Test error"):
            backend.train(trainer=trainer)

        # _instrumented_command should be cleaned up
        assert not hasattr(backend, "_instrumented_command")


def test_transformers_trainer_with_progression_disabled(mock_runtime):
    """Test TransformersTrainer with enable_progression_tracking=False uses parent train."""
    with (
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
            return_value=None,
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.train",
            return_value="job-no-instrumentation",
        ) as mock_parent_train,
    ):
        backend = ExperimentalKubernetesBackend()

        # Create TransformersTrainer with progression tracking disabled
        trainer = TransformersTrainer(
            func=dummy_train_func,
            enable_progression_tracking=False,
        )

        result = backend.train(trainer=trainer, runtime=mock_runtime)

        # Should call parent train directly (not _train_transformers_instrumented)
        assert result == "job-no-instrumentation"
        mock_parent_train.assert_called_once()

        # Verify no instrumentation-related options were added
        call_args = mock_parent_train.call_args
        # Options should be None or empty when passed to parent
        options = call_args[1].get("options")
        assert options is None or len(options) == 0


def test_transformers_trainer_with_progression_enabled(mock_runtime):
    """Test TransformersTrainer with enable_progression_tracking=True adds instrumentation."""
    with (
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.__init__",
            return_value=None,
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.train",
            return_value="job-instrumented",
        ) as mock_parent_train,
    ):
        backend = ExperimentalKubernetesBackend()

        # Create TransformersTrainer with progression tracking enabled (default)
        trainer = TransformersTrainer(func=dummy_train_func)

        result = backend.train(trainer=trainer, runtime=mock_runtime)

        # Should add progression tracking annotations
        assert result == "job-instrumented"
        mock_parent_train.assert_called_once()

        # Verify instrumentation options were added
        call_args = mock_parent_train.call_args
        options = call_args[1]["options"]
        assert options is not None
        assert len(options) > 0
