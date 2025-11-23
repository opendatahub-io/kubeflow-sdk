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

"""Unit tests for JIT checkpoint code."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from kubeflow.trainer.test.common import SUCCESS, TestCase

# ============================================================================
# Mock torch and transformers to avoid dependencies
# ============================================================================


@pytest.fixture
def mock_torch():
    """Mock torch module."""

    # Create a real base class for TrainerCallback
    class TrainerCallback:
        """Mock TrainerCallback base class."""

        pass

    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch.cuda": MagicMock(),
            "transformers": MagicMock(),
            "transformers.trainer_utils": MagicMock(),
        },
    ):
        import sys

        torch_mock = sys.modules["torch"]
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.Stream.return_value = Mock()
        torch_mock.cuda.stream = MagicMock()

        transformers_mock = sys.modules["transformers"]
        transformers_mock.TrainerCallback = TrainerCallback  # Use real class

        trainer_utils_mock = sys.modules["transformers.trainer_utils"]
        trainer_utils_mock.PREFIX_CHECKPOINT_DIR = "checkpoint"

        yield torch_mock


@pytest.fixture
def mock_trainer():
    """Mock HuggingFace Trainer object."""
    trainer = Mock()
    trainer.state.global_step = 100
    trainer._get_output_dir.return_value = "/tmp/output"
    trainer._save_checkpoint = Mock()
    trainer.model = Mock()
    return trainer


# ============================================================================
# CheckpointManager Tests
# ============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="checkpoint manager initializes with CUDA available",
            expected_status=SUCCESS,
            config={"cuda_available": True},
        ),
        TestCase(
            name="checkpoint manager initializes without CUDA",
            expected_status=SUCCESS,
            config={"cuda_available": False},
        ),
    ],
)
def test_checkpoint_manager_initialization(test_case, mock_torch, mock_trainer):
    """Test CheckpointManager initialization."""
    print("Executing test:", test_case.name)

    # Configure CUDA availability
    mock_torch.cuda.is_available.return_value = test_case.config["cuda_available"]

    # Import after mocking
    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    manager = CheckpointManager(mock_trainer)

    assert manager.trainer == mock_trainer
    assert not manager.checkpoint_requested
    assert not manager._in_optimizer_step

    if test_case.config["cuda_available"]:
        assert manager.checkpoint_stream is not None
    else:
        assert manager.checkpoint_stream is None

    print("test execution complete")


def test_signal_handler_registration(mock_torch, mock_trainer):
    """Test signal handler registration."""
    print("Executing test: signal handler registration")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    manager = CheckpointManager(mock_trainer)

    with patch.object(manager, "signal") as mock_signal:
        manager.setup_signal_handler()

        # Verify signal.signal was called with SIGTERM
        mock_signal.signal.assert_called_once()
        call_args = mock_signal.signal.call_args
        assert call_args[0][0] == mock_signal.SIGTERM
        assert callable(call_args[0][1])

    print("test execution complete")


def test_sigterm_handler_sets_checkpoint_requested(mock_torch, mock_trainer):
    """Test SIGTERM handler sets checkpoint_requested flag."""
    print("Executing test: SIGTERM handler sets checkpoint requested")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    manager = CheckpointManager(mock_trainer)

    with patch.object(manager, "threading") as mock_threading:
        mock_thread = Mock()
        mock_threading.Thread.return_value = mock_thread

        # Simulate SIGTERM
        manager._sigterm_handler(15, None)

        assert manager.checkpoint_requested
        mock_threading.Thread.assert_called_once()
        mock_thread.start.assert_called_once()

    print("test execution complete")


def test_sigterm_handler_ignores_duplicate_signals(mock_torch, mock_trainer):
    """Test SIGTERM handler ignores duplicate signals."""
    print("Executing test: SIGTERM handler ignores duplicates")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    manager = CheckpointManager(mock_trainer)
    manager.checkpoint_requested = True  # Already requested

    with patch.object(manager, "threading") as mock_threading:
        # Simulate second SIGTERM
        manager._sigterm_handler(15, None)

        # Should not create new thread
        mock_threading.Thread.assert_not_called()

    print("test execution complete")


def test_should_checkpoint_now(mock_torch, mock_trainer):
    """Test should_checkpoint_now returns correct state."""
    print("Executing test: should checkpoint now")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    manager = CheckpointManager(mock_trainer)

    assert not manager.should_checkpoint_now()

    manager.checkpoint_requested = True
    assert manager.should_checkpoint_now()

    print("test execution complete")


# ============================================================================
# JITCheckpointCallback Tests
# ============================================================================


def test_callback_initialization(mock_torch):
    """Test JITCheckpointCallback initialization."""
    print("Executing test: callback initialization")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    callback = CheckpointManager.JITCheckpointCallback()

    assert callback.jit_manager is None
    assert callback._trainer_ref is None

    print("test execution complete")


def test_callback_on_train_begin_creates_manager(mock_torch, mock_trainer):
    """Test on_train_begin creates CheckpointManager."""
    print("Executing test: on_train_begin creates manager")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    callback = CheckpointManager.JITCheckpointCallback()
    callback._trainer_ref = mock_trainer

    # Mock args, state, control
    args = Mock()
    state = Mock()
    control = Mock()

    with (
        patch.object(CheckpointManager, "__init__", return_value=None),
        patch.object(CheckpointManager, "setup_signal_handler") as mock_setup,
    ):
        callback.on_train_begin(args, state, control)

        assert callback.jit_manager is not None
        mock_setup.assert_called_once()

    print("test execution complete")


def test_callback_on_pre_optimizer_step_sets_flag(mock_torch, mock_trainer):
    """Test on_pre_optimizer_step sets _in_optimizer_step flag."""
    print("Executing test: on_pre_optimizer_step sets flag")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    callback = CheckpointManager.JITCheckpointCallback()
    manager = Mock()
    manager._in_optimizer_step = False
    manager.should_checkpoint_now.return_value = False
    callback.jit_manager = manager

    args = Mock()
    state = Mock()
    control = Mock()

    callback.on_pre_optimizer_step(args, state, control)

    assert manager._in_optimizer_step

    print("test execution complete")


def test_callback_on_optimizer_step_clears_flag(mock_torch, mock_trainer):
    """Test on_optimizer_step clears _in_optimizer_step flag."""
    print("Executing test: on_optimizer_step clears flag")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    callback = CheckpointManager.JITCheckpointCallback()
    manager = Mock()
    manager._in_optimizer_step = True
    callback.jit_manager = manager

    args = Mock()
    state = Mock()
    control = Mock()

    callback.on_optimizer_step(args, state, control)

    assert not manager._in_optimizer_step

    print("test execution complete")


def test_callback_stops_training_when_checkpoint_requested(mock_torch, mock_trainer):
    """Test callback stops training when checkpoint is requested."""
    print("Executing test: callback stops training")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    callback = CheckpointManager.JITCheckpointCallback()
    manager = Mock()
    manager.should_checkpoint_now.return_value = True
    callback.jit_manager = manager

    args = Mock()
    state = Mock()
    control = Mock()
    control.should_training_stop = False
    control.should_save = True

    # Test on_step_end
    callback.on_step_end(args, state, control)
    assert control.should_training_stop
    assert not control.should_save

    # Reset and test on_epoch_end
    control.should_training_stop = False
    control.should_save = True
    callback.on_epoch_end(args, state, control)
    assert control.should_training_stop
    assert not control.should_save

    print("test execution complete")


# ============================================================================
# Async Checkpoint Logic Tests
# ============================================================================


def test_async_checkpoint_waits_for_optimizer_step(mock_torch, mock_trainer):
    """Test _async_checkpoint waits if in optimizer step."""
    print("Executing test: async checkpoint waits for optimizer step")

    from kubeflow.trainer.rhai.jit_checkpoint_code import CheckpointManager

    manager = CheckpointManager(mock_trainer)
    manager._in_optimizer_step = True

    # Use a flag to simulate the wait loop exiting
    wait_count = [0]

    def mock_sleep(duration):
        wait_count[0] += 1
        if wait_count[0] > 2:
            manager._in_optimizer_step = False

    with (
        patch.object(manager.time, "sleep", side_effect=mock_sleep),
        patch.object(manager.os, "makedirs"),
        patch.object(manager.os.path, "join", side_effect=lambda *args: "/".join(args)),
        patch("builtins.open", MagicMock()),
    ):
        manager._async_checkpoint()

        # Verify it waited at least once
        assert wait_count[0] > 0
        # Verify checkpoint was attempted
        mock_trainer._save_checkpoint.assert_called_once()

    print("test execution complete")
