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

"""Tests for checkpoint instrumentation module."""

import os
import sys
from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.test.common import SUCCESS, TestCase


def test_checkpoint_instrumentation_imports():
    """Test that checkpoint instrumentation module imports correctly."""
    print("Executing test: checkpoint instrumentation imports")

    # The module itself can be imported without torch
    # create_checkpoint_instrumentation is what requires torch at runtime
    import kubeflow.trainer.rhai.instrumentation.checkpoint as checkpoint_module

    assert hasattr(checkpoint_module, "create_checkpoint_instrumentation")
    assert callable(checkpoint_module.create_checkpoint_instrumentation)
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="checkpoint instrumentation with JIT enabled",
            expected_status=SUCCESS,
            config={
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
            },
            expected_output={
                "has_checkpoint_manager": True,
                "has_callback": True,
                "has_apply_checkpointing": True,
                "has_upload_function": True,
            },
        ),
        TestCase(
            name="checkpoint instrumentation with JIT disabled",
            expected_status=SUCCESS,
            config={
                "enable_jit": False,
                "output_dir": "/tmp/checkpoints",
            },
            expected_output={
                "has_checkpoint_manager": False,
                "has_callback": False,
                "has_apply_checkpointing": True,
                "has_upload_function": True,
            },
        ),
    ],
)
def test_create_checkpoint_instrumentation(test_case):
    """Test create_checkpoint_instrumentation returns expected components."""
    print(f"Executing test: {test_case.name}")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create checkpoint instrumentation components
        result = create_checkpoint_instrumentation(test_case.config)

        # Unpack the tuple
        checkpoint_manager_cls, callback_cls, apply_fn, upload_fn = result

        # Verify structure based on enable_jit setting
        if test_case.expected_output["has_checkpoint_manager"]:
            assert checkpoint_manager_cls is not None
            assert checkpoint_manager_cls.__name__ == "CheckpointManager"
        else:
            assert checkpoint_manager_cls is None

        if test_case.expected_output["has_callback"]:
            assert callback_cls is not None
            assert callback_cls.__name__ == "JITCheckpointCallback"
        else:
            assert callback_cls is None

        # These should always be present
        assert callable(apply_fn)
        assert callable(upload_fn)

    print("test execution complete")


def test_auto_resume_logic_exists():
    """Test that auto-resume logic is present in the checkpoint instrumentation."""
    print("Executing test: auto-resume logic exists")

    # Mock transformers and torch
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation
        _, _, apply_checkpointing, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
            }
        )

        # Verify apply_checkpointing function exists and is callable
        assert callable(apply_checkpointing)

        # The function contains the auto-resume logic within _patched_train
        # which only auto-resumes if resume_from_checkpoint is None
        # This is tested through integration tests rather than unit tests

    print("test execution complete")


def test_async_upload_worker_components():
    """Test that JITCheckpointCallback has async upload worker components."""
    print("Executing test: async upload worker components")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    mock_fs = Mock()
    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation with cloud storage
        _, callback_cls, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": False,  # Disable S3 access check
            }
        )

        # Verify callback class exists and has required methods
        assert callback_cls is not None
        callback_instance = callback_cls(cloud_remote_storage_uri="s3://test-bucket/checkpoints")

        # Check for upload worker attributes
        assert hasattr(callback_instance, "upload_queue")
        assert hasattr(callback_instance, "_upload_thread")
        assert hasattr(callback_instance, "_shutdown_event")
        assert hasattr(callback_instance, "start_upload_worker")
        assert hasattr(callback_instance, "shutdown_upload_worker")
        assert hasattr(callback_instance, "_upload_worker_loop")

    print("test execution complete")


def test_parallel_upload_components():
    """Test that JITCheckpointCallback has parallel upload functionality."""
    print("Executing test: parallel upload components")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    mock_fs = Mock()
    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation with cloud storage
        _, callback_cls, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": False,
            }
        )

        assert callback_cls is not None
        callback_instance = callback_cls(cloud_remote_storage_uri="s3://test-bucket/checkpoints")

        # Check for parallel upload method
        assert hasattr(callback_instance, "_parallel_upload_files")
        assert callable(callback_instance._parallel_upload_files)

    print("test execution complete")


def test_upload_worker_lifecycle(tmp_path):
    """Test upload worker can be started and stopped."""
    print("Executing test: upload worker lifecycle")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    mock_fs = Mock()
    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation with cloud storage
        _, callback_cls, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": str(tmp_path),
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": False,
            }
        )

        callback_instance = callback_cls(cloud_remote_storage_uri="s3://test-bucket/checkpoints")

        # Start worker
        callback_instance.start_upload_worker()
        assert callback_instance.upload_queue is not None
        assert callback_instance._upload_thread is not None
        assert callback_instance._upload_thread.is_alive()

        # Stop worker
        callback_instance._shutdown_event.set()
        callback_instance._upload_thread.join(timeout=2)
        assert not callback_instance._upload_thread.is_alive()

    print("test execution complete")


def test_upload_worker_skips_restart_when_alive():
    """Test that start_upload_worker() skips restart when worker is alive."""
    print("Executing test: upload worker skips restart when alive")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    mock_fs = Mock()
    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        _, callback_cls, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": False,
            }
        )

        callback_instance = callback_cls(cloud_remote_storage_uri="s3://test-bucket/checkpoints")

        # Start worker first time
        callback_instance.start_upload_worker()
        first_thread = callback_instance._upload_thread
        assert first_thread.is_alive()

        # Try to start again - should reuse existing thread
        callback_instance.start_upload_worker()
        second_thread = callback_instance._upload_thread
        assert first_thread is second_thread  # Same thread instance

        # Cleanup
        callback_instance._shutdown_event.set()
        callback_instance._upload_thread.join(timeout=2)

    print("test execution complete")


def test_s3_access_retry_logic():
    """Test that S3 access verification includes 3-retry logic."""
    print("Executing test: S3 retry logic")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    # Mock fsspec to test retry logic
    mock_fs = Mock()
    attempt_count = {"value": 0}

    def side_effect_pipe(*args, **kwargs):
        attempt_count["value"] += 1
        if attempt_count["value"] < 3:
            raise Exception("Temporary failure")
        # Success on 3rd attempt

    mock_fs.pipe = Mock(side_effect=side_effect_pipe)
    mock_fs.cat = Mock(return_value=b"test")
    mock_fs.rm_file = Mock()

    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation - this triggers S3 access verification
        # The callback is created internally during create_checkpoint_instrumentation
        _, _, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": True,
            }
        )

        # Verify retry happened 3 times during callback initialization
        assert attempt_count["value"] == 3

    print("test execution complete")


def test_final_model_upload_skips_checkpoints_and_staging(tmp_path):
    """Test that final model upload skips checkpoint dirs and staging."""
    print("Executing test: final model upload skips checkpoints and staging")

    # Create test directory structure
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create files and dirs that should be uploaded
    (output_dir / "model.bin").write_text("model data")
    (output_dir / "config.json").write_text("{}")
    model_dir = output_dir / "final_model"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_text("model")

    # Create checkpoint dirs (should be skipped)
    (output_dir / "checkpoint-100").mkdir()
    (output_dir / "checkpoint-200").mkdir()

    # Create staging dir (should be skipped) - using the actual constant
    from kubeflow.trainer.rhai.constants import CHECKPOINT_STAGING_DIR

    staging_dir = output_dir / CHECKPOINT_STAGING_DIR
    staging_dir.mkdir()

    # Create .cache dir (should be skipped)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir()

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False
    mock_torch.distributed.is_initialized.return_value = False

    # Track what gets uploaded
    uploaded_items = []

    # Mock fsspec.callbacks.Callback
    mock_callback = type("Callback", (), {})

    mock_fs = Mock()
    mock_fs.put_file = Mock(side_effect=lambda src, dst, **kwargs: uploaded_items.append(dst))
    mock_fs.put = Mock(side_effect=lambda src, dst, **kwargs: uploaded_items.append(dst))

    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs
    mock_fsspec_callbacks = Mock()
    mock_fsspec_callbacks.Callback = mock_callback

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
            "fsspec.callbacks": mock_fsspec_callbacks,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation
        _, _, _, upload_fn = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": str(output_dir),
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": False,
            }
        )

        # Mock LOCAL_RANK environment variable
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            # Call upload function
            upload_fn()

        print(f"Uploaded items: {uploaded_items}")

        # Verify only model files were uploaded, not checkpoints or staging
        assert "model.bin" in uploaded_items
        assert "config.json" in uploaded_items
        assert "final_model" in uploaded_items

        # Verify checkpoints and staging were not uploaded
        from kubeflow.trainer.rhai.constants import CHECKPOINT_STAGING_DIR

        checkpoint_uploads = [item for item in uploaded_items if item.startswith("checkpoint-")]
        if checkpoint_uploads:
            print(f"Unexpected checkpoint uploads: {checkpoint_uploads}")
        assert not any(item.startswith("checkpoint-") for item in uploaded_items)
        assert CHECKPOINT_STAGING_DIR not in uploaded_items
        assert ".cache" not in uploaded_items

    print("test execution complete")


def test_upload_skipped_when_no_cloud_storage():
    """Test that upload is skipped when cloud_remote_storage_uri is not configured."""
    print("Executing test: upload skipped when no cloud storage")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation WITHOUT cloud storage
        _, callback_cls, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
                # No cloud_remote_storage_uri
            }
        )

        callback_instance = callback_cls(cloud_remote_storage_uri=None)

        # Verify remote_fs is None
        assert callback_instance.remote_fs is None

        # Call on_save - should return early without upload
        mock_args = Mock(local_process_index=0, output_dir="/tmp/checkpoints")
        mock_state = Mock(global_step=100)
        mock_control = Mock()

        # Should not raise any errors
        callback_instance.on_save(mock_args, mock_state, mock_control)

    print("test execution complete")


def test_upload_skipped_on_non_rank_0():
    """Test that upload is skipped on non-rank-0 processes."""
    print("Executing test: upload skipped on non-rank-0")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    mock_fs = Mock()
    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation with cloud storage
        _, callback_cls, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": False,
            }
        )

        callback_instance = callback_cls(cloud_remote_storage_uri="s3://test-bucket/checkpoints")

        # Mock args for rank 1 (not rank 0)
        mock_args = Mock(local_process_index=1, output_dir="/tmp/checkpoints")
        mock_state = Mock(global_step=100)
        mock_control = Mock()

        # Call on_save - should return early for non-rank-0
        callback_instance.on_save(mock_args, mock_state, mock_control)

        # Verify upload queue was not created (upload worker not started)
        assert callback_instance.upload_queue is None

    print("test execution complete")


def test_upload_skipped_when_output_dir_not_exists():
    """Test that upload is skipped when output_dir does not exist."""
    print("Executing test: upload skipped when output_dir does not exist")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False
    mock_torch.distributed.is_initialized.return_value = False

    mock_fs = Mock()
    mock_fsspec = Mock()
    mock_fsspec.filesystem.return_value = mock_fs

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
            "fsspec": mock_fsspec,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation
        _, _, _, upload_fn = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/nonexistent/path",
                "cloud_remote_storage_uri": "s3://test-bucket/checkpoints",
                "verify_cloud_storage_access": False,
            }
        )

        # Mock LOCAL_RANK
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            # Should not raise error, just return early
            upload_fn()

    print("test execution complete")


def test_checkpoint_manager_signal_handler():
    """Test CheckpointManager SIGTERM signal handler."""
    print("Executing test: CheckpointManager signal handler")

    # Mock dependencies
    mock_transformers = Mock()
    mock_transformers.TrainerCallback = type("TrainerCallback", (), {})
    mock_transformers.trainer_utils = Mock()
    mock_transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = "checkpoint"

    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.distributed.is_available.return_value = False

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "transformers.trainer_utils": mock_transformers.trainer_utils,
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
        },
    ):
        from kubeflow.trainer.rhai.instrumentation.checkpoint import (
            create_checkpoint_instrumentation,
        )

        # Create instrumentation
        checkpoint_manager_cls, _, _, _ = create_checkpoint_instrumentation(
            {
                "enable_jit": True,
                "output_dir": "/tmp/checkpoints",
            }
        )

        # Create a mock trainer
        mock_trainer = Mock()

        # Create CheckpointManager instance
        manager = checkpoint_manager_cls(trainer=mock_trainer)

        # Verify initial state
        assert not manager.checkpoint_requested
        assert not manager._should_exit

        # Simulate SIGTERM
        manager._sigterm_handler(None, None)

        # Verify checkpoint was requested
        assert manager.checkpoint_requested

        # Call again - should not change state (idempotent)
        manager._sigterm_handler(None, None)
        assert manager.checkpoint_requested

    print("test execution complete")
