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

"""Checkpoint instrumentation for Kubeflow Trainer.

This module provides checkpoint management functionality that runs inside training containers.
It's imported by training scripts instead of being inlined, reducing command size significantly.
"""

def create_checkpoint_instrumentation(checkpoint_config: dict) -> tuple:
    """Checkpoint instrumentation injected into training pods."""
    from collections.abc import Callable
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    from pathlib import PurePosixPath
    from queue import Empty, LifoQueue
    import re
    import shutil
    import signal
    import sys
    import threading
    import time

    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    from kubeflow.trainer.rhai.constants import CHECKPOINT_INCOMPLETE_MARKER, CHECKPOINT_STAGING_DIR

    _log_prefix = None

    def _log(message: str, args: object | None = None) -> None:
        nonlocal _log_prefix
        if _log_prefix is None:
            node = os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK") or "0"
            rank = getattr(args, "local_process_index", None)
            if rank is None:
                rank = os.environ.get("LOCAL_RANK") or "0"
            _log_prefix = f"[Kubeflow-node{node}-rank{rank}]"
        print(f"{_log_prefix} {message}", flush=True)

    def wait_for_all_ranks(operation: str) -> None:
        """Barrier across ranks to synchronize distributed training."""
        if not dist.is_available() or not dist.is_initialized():
            return
        try:
            # Specify device to avoid "guessing device ID" warning
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        except Exception as e:
            raise RuntimeError(
                f"[Kubeflow] Barrier synchronization failed during {operation}: {e}. "
                "This typically indicates one or more training processes crashed or "
                "exited early. Check logs to identify which rank failed."
            ) from e

    class CheckpointManager:
        """Manages just-in-time checkpointing on SIGTERM signal using CUDA streams."""

        def __init__(self, trainer):
            self.trainer = trainer
            self.checkpoint_requested = False
            self._should_exit = False
            self._original_sigterm_handler = None
            self.checkpoint_stream = None

            # Initialize CUDA stream for async checkpoint operations
            try:
                if torch.cuda.is_available():
                    self.checkpoint_stream = torch.cuda.Stream()
                    _log("CUDA stream initialized for async checkpointing")
            except (AttributeError, RuntimeError, OSError) as e:
                _log(f"CUDA not available, checkpointing will be synchronous: {e}")

        def setup_signal_handler(self):
            """Register SIGTERM signal handler for JIT checkpointing."""
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
            _log("JIT checkpoint signal handler registered for SIGTERM")

        def _sigterm_handler(self, signum, frame):
            """Mark checkpoint requested on SIGTERM.

            Checkpoint is deferred to the next training callback (on_step_end /
            on_epoch_end) to avoid reentrancy deadlocks when SIGTERM arrives
            during an in-progress periodic checkpoint.
            """
            if self.checkpoint_requested:
                return

            _log("SIGTERM received, checkpoint will be saved at next safe point")
            self.checkpoint_requested = True

        def _save_jit_checkpoint(self):
            """Execute checkpoint, saving model state and training artifacts."""
            self.checkpoint_requested = False
            try:
                current_step = self.trainer.state.global_step
                _log(f"Starting JIT checkpoint at step {current_step}")

                # Build per-rank marker filename
                node = os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK") or "0"
                try:
                    local_rank = self.trainer.args.local_process_index
                except Exception:
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

                output_dir = self.trainer._get_output_dir(trial=None)
                checkpoint_path = os.path.join(
                    output_dir, f"{PREFIX_CHECKPOINT_DIR}-{current_step}"
                )
                os.makedirs(checkpoint_path, exist_ok=True)

                # Each rank creates its own sentinel to track per-rank completion
                sentinel_name = f"{CHECKPOINT_INCOMPLETE_MARKER}.node-{node}-rank-{local_rank}"
                sentinel_file = os.path.join(checkpoint_path, sentinel_name)
                try:
                    with open(sentinel_file, "w") as f:
                        f.write(
                            f"Checkpoint started at step {current_step} in node {node} rank {local_rank}"
                        )
                except Exception as e:
                    _log(
                        f"Warning: Failed to write sentinel file: {e}. "
                        "Check local disk space and write permissions in output_dir; "
                        "this checkpoint will be treated as complete during resume. "
                        "Please manually verify if the checkpoint is indeed complete: "
                        f"{checkpoint_path}"
                    )

                # Checkpoint using dedicated CUDA stream
                if self.checkpoint_stream is not None:
                    # Wait for default stream to complete all pending operations
                    self.checkpoint_stream.wait_stream(torch.cuda.default_stream())

                    # Record all model parameters on checkpoint stream to prevent deallocation
                    for param in self.trainer.model.parameters():
                        param.record_stream(self.checkpoint_stream)

                    with torch.cuda.stream(self.checkpoint_stream):
                        self.trainer._save_checkpoint(self.trainer.model, trial=None)
                    self.checkpoint_stream.synchronize()
                else:
                    # Fallback if no CUDA stream
                    self.trainer._save_checkpoint(self.trainer.model, trial=None)

                if os.path.exists(sentinel_file):
                    try:
                        os.remove(sentinel_file)
                    except Exception as e:
                        _log(
                            f"Warning: Failed to remove sentinel file: {e}. "
                            "Check permissions; a stale marker may cause this checkpoint "
                            f"to be skipped. Remove it manually from: {sentinel_file}"
                        )

                # Trigger on_save callback to upload checkpoint if cloud storage
                self.trainer.callback_handler.on_save(
                    self.trainer.args, self.trainer.state, self.trainer.control
                )

                _log(
                    f"JIT checkpoint completed at step {current_step} in node {node} rank {local_rank}"
                )
                self._should_exit = True

            except Exception as e:
                _log(
                    f"Failed to save JIT checkpoint: {e}. "
                    "Check local disk space, write permissions, and model save errors; "
                    "training will resume from lastest periodic checkpoint."
                )
                import traceback

                traceback.print_exc()

        def checkpoint_in_progress(self):
            """Return True if checkpoint requested."""
            return self.checkpoint_requested

    class JITCheckpointCallback(TrainerCallback):
        """Transformers callback that integrates JIT checkpointing with trainer lifecycle."""

        def __init__(self, cloud_remote_storage_uri: str | None = None) -> None:
            self.jit_manager = None
            self._trainer_ref = None
            self.cloud_remote_storage_uri = cloud_remote_storage_uri
            self.remote_fs = None
            # Async upload state
            self.upload_queue = None  # LifoQueue for background uploads
            self._upload_thread = None  # Background worker thread
            self._shutdown_event = None  # Event to signal shutdown
            self._upload_error = None  # Store background thread errors
            self._upload_error_lock = threading.Lock()  # Lock for error propagation

            if cloud_remote_storage_uri and "://" in cloud_remote_storage_uri:
                import fsspec

                protocol, base_path = cloud_remote_storage_uri.split("://", 1)

                fsspec_kwargs = {}
                if protocol == "s3":
                    # AWS_S3_ENDPOINT must be explicitly passed since it's not
                    # a standardized name like other AWS credentials
                    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
                    if endpoint_url:
                        verify_ssl = checkpoint_config.get("verify_cloud_storage_ssl", True)
                        fsspec_kwargs = {
                            "client_kwargs": {"endpoint_url": endpoint_url, "verify": verify_ssl},
                        }

                        # Warn when SSL verification is disabled
                        if not verify_ssl:
                            _log(
                                "WARNING: SSL certificate verification disabled. "
                                "This should only be used with trusted S3-compatible storage."
                            )

                try:
                    # Create underlying filesystem and wrap with directory fs to embed base path
                    underlying_fs = fsspec.filesystem(protocol, **fsspec_kwargs)
                    self.remote_fs = fsspec.filesystem("dir", path=base_path, fs=underlying_fs)

                    # Verify storage access by writing/reading test file (if enabled)
                    if checkpoint_config.get("verify_cloud_storage_access", True):
                        test_file = ".kubeflow-access-test"
                        last_error = None
                        for attempt in range(1, 4):  # 3 attempts with backoff
                            try:
                                self.remote_fs.pipe(test_file, b"test")
                                self.remote_fs.cat(test_file)
                                self.remote_fs.rm_file(test_file)
                                last_error = None
                                break  # Success
                            except Exception as e:
                                last_error = e
                                if attempt < 3:
                                    _log(
                                        f"Cloud storage access verification failed (attempt {attempt}/3), retrying..."
                                    )
                                    time.sleep(1)

                        if last_error:
                            raise last_error  # Re-raise to outer except

                    _log(f"Cloud storage configured: {cloud_remote_storage_uri} ({protocol})")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to check this node has access to the storage path: "
                        f"'{cloud_remote_storage_uri}'. Error: {e}. "
                        f"If using self-signed certificates, "
                        f"set verify_cloud_storage_ssl=False. "
                        f"If experiencing permission issues, check you have read, "
                        f"write and delete permissions to '{cloud_remote_storage_uri}'. "
                        f"This check can be disabled by setting verify_cloud_storage_access=False."
                    ) from e

        def _calculate_local_dir_size(self, path: str) -> int:
            """Calculate total size of local directory."""
            import contextlib

            total = 0
            for root, _, files in os.walk(path):
                for f in files:
                    with contextlib.suppress(OSError, FileNotFoundError):
                        total += os.path.getsize(os.path.join(root, f))
            return total

        def _cloud_storage_progress_callback(self, operation: str, total_size: int):
            """Create progress callback with byte-level tracking via branched()."""
            from fsspec.callbacks import Callback

            class ProgressCallback(Callback):
                def __init__(self, name, total_size):
                    super().__init__()
                    self.name = name
                    self.total_size = total_size
                    self.last = time.time()
                    self.value = 0
                    _log(f"{name} size: {total_size / (1024 * 1024):.1f} MB")

                def branched(self, path_1, path_2, **kwargs):
                    """Return child callback that accumulates bytes to parent."""
                    parent = self

                    class Child(Callback):
                        def relative_update(self, inc=1):
                            super().relative_update(inc)
                            parent.value += inc
                            if time.time() - parent.last >= 1:
                                mb = parent.value / (1024 * 1024)
                                pct = (
                                    int((parent.value / parent.total_size) * 100)
                                    if parent.total_size
                                    else 0
                                )
                                _log(
                                    f"Progress: {mb:.1f}"
                                    f"/{parent.total_size / (1024 * 1024):.1f} MB ({pct}%)"
                                )
                                parent.last = time.time()

                    kwargs["callback"] = Child()
                    return kwargs["callback"]

            return ProgressCallback(operation, total_size)

        def _retry_marker_op(self, op: Callable[[], None]) -> Exception | None:
            """Retry marker operation with backoff."""
            last_error = None
            for attempt in range(1, 4):
                try:
                    op()
                    return None
                except Exception as e:
                    last_error = e
                    if attempt < 3:
                        time.sleep(1)
            return last_error

        def _check_upload_error(self) -> None:
            """Raise any background upload error."""
            with self._upload_error_lock:
                if self._upload_error is not None:
                    error = self._upload_error
                    self._upload_error = None  # Clear after reading
                    raise error

        def start_upload_worker(self) -> None:
            """Start background upload worker."""
            worker_alive = self._upload_thread is not None and self._upload_thread.is_alive()
            if worker_alive:
                return
            if self.upload_queue is None:
                # Use LIFO to prioritize the latest checkpoint
                # when resuming after interruptions the latest state is picked.
                self.upload_queue = LifoQueue()
            self._shutdown_event = threading.Event()
            self._upload_thread = threading.Thread(
                target=self._upload_worker_loop,
                daemon=False,
                name="KubeflowCheckpointUploader",
            )
            self._upload_thread.start()
            _log("Background upload worker started")

        def _upload_worker_loop(self) -> None:
            """Upload worker loop."""
            while True:
                try:
                    # Use timeout to periodically check shutdown event
                    task = self.upload_queue.get(timeout=1.0)
                except Empty:
                    if self._shutdown_event.is_set():
                        break
                    continue

                try:
                    self._upload_checkpoint_to_cloud(task)
                except Exception as e:
                    # Store error for main thread propagation
                    checkpoint_name = task[1]
                    with self._upload_error_lock:
                        self._upload_error = RuntimeError(
                            f"[Kubeflow] Background upload failed for {checkpoint_name}: {e}. "
                            "To fix: verify S3 endpoint/credentials, bucket permissions, "
                            "and network connectivity, then retry the training job."
                        )
                        self._upload_error.__cause__ = e
                    _log(
                        f"ERROR: Background upload failed for {checkpoint_name}: {e}. "
                        "Check S3 connectivity/permissions and retry."
                    )
                finally:
                    self.upload_queue.task_done()

        def _upload_checkpoint_to_cloud(self, task: tuple) -> None:
            """Upload checkpoint."""
            staging_path, checkpoint_name, total_size, incomplete_marker_name = task
            incomplete_marker_path = f"{checkpoint_name}/{incomplete_marker_name}"

            # Upload files in parallel
            _log(f"Starting parallel upload to S3: {checkpoint_name}")

            failed_files = self._parallel_upload_files(staging_path, checkpoint_name, total_size)

            if failed_files:
                raise RuntimeError(
                    f"Upload failed for {len(failed_files)} files: "
                    f"{[f[1] for f in failed_files[:5]]}. "  # Show first 5
                    "To fix: check S3 permissions/space and retry."
                )

            _log(f"Upload complete: {checkpoint_name}")

            # Delete incomplete sentinel for this uploader
            remove_error = self._retry_marker_op(
                lambda: self.remote_fs.rm_file(incomplete_marker_path)
            )
            if remove_error:
                _log(
                    f"Warning: Failed to remove incomplete marker '{incomplete_marker_path}': "
                    f"{remove_error}. "
                    "The checkpoint is complete but cannot be loaded during training resume "
                    f"because the marker still exists. Manually delete: {incomplete_marker_path}"
                )

            # Delete local staging checkpoint
            try:
                shutil.rmtree(staging_path)
                _log(f"Deleted local staging checkpoint: {checkpoint_name}")
            except Exception as cleanup_error:
                _log(
                    f"Warning: Failed to delete local staging checkpoint {checkpoint_name}: "
                    f"{cleanup_error}. Upload succeeded, but local cleanup failed. "
                    "Staging data will be cleaned up when the pod terminates."
                )

        def _parallel_upload_files(
            self, staging_path: str, checkpoint_name: str, total_size: int
        ) -> list[tuple[str, str]]:
            """Upload files in parallel; return failed list."""
            # Collect all files to upload
            files_to_upload = []
            for root, _, files in os.walk(staging_path):
                for f in files:
                    local_file = os.path.join(root, f)
                    # Compute relative path from staging_path
                    rel_path = os.path.relpath(local_file, staging_path)
                    rel_path = rel_path.replace(os.sep, "/")
                    remote_file = str(PurePosixPath(checkpoint_name) / rel_path)
                    files_to_upload.append((local_file, remote_file))

            if not files_to_upload:
                return []

            # Cap workers at min(4, file_count)
            num_workers = min(4, len(files_to_upload))
            failed_files = []

            # Progress tracking
            uploaded_bytes = 0
            last_update = time.time()
            lock = threading.Lock()

            def upload_file(local_file: str, remote_file: str) -> bool:
                """Upload single file, return True on success."""
                nonlocal uploaded_bytes, last_update
                last_error = None
                for attempt in range(1, 4):  # 3 attempts total
                    try:
                        self.remote_fs.put_file(local_file, remote_file)

                        # Update progress
                        file_size = os.path.getsize(local_file)
                        with lock:
                            uploaded_bytes += file_size
                            if time.time() - last_update >= 1:
                                mb = uploaded_bytes / (1024 * 1024)
                                pct = int((uploaded_bytes / total_size) * 100) if total_size else 0
                                _log(
                                    f"Progress: {mb:.1f}"
                                    f"/{total_size / (1024 * 1024):.1f} MB ({pct}%)"
                                )
                                last_update = time.time()

                        return True
                    except Exception as e:
                        last_error = e
                        if attempt < 3:
                            _log(f"Retry {attempt}/3 for {remote_file}")
                            time.sleep(1)

                _log(
                    f"ERROR: Upload failed after 3 attempts for {remote_file}: {last_error}. "
                    "Check S3 connectivity/permissions and retry."
                )
                return False

            # Upload files in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(upload_file, local, remote): (local, remote)
                    for local, remote in files_to_upload
                }

                for future in as_completed(futures):
                    local, remote = futures[future]
                    if not future.result():
                        failed_files.append((local, remote))

            return failed_files

        def shutdown_upload_worker(self) -> None:
            """Stop upload worker."""
            if self.upload_queue is not None and self._upload_thread is not None:
                _log("Waiting for background uploads to complete...")
                self._shutdown_event.set()  # Signal shutdown
                self._upload_thread.join(timeout=3600)  # Wait up to 1 hour
                if self._upload_thread.is_alive():
                    _log(
                        "Warning: Upload worker thread is still running after 1 hour timeout. "
                        "Uploads may still be in progress or stalled. Check S3 logs/progress and "
                        "consider increasing termination grace period or timeout if needed."
                    )
                else:
                    _log("Background upload worker stopped")

        def on_init_end(self, args, state, control, **kwargs):
            """Download latest checkpoint from S3 (local rank 0)."""
            if not self.remote_fs:
                # Return since cloud storage not configured by user
                return

            is_local_rank_0 = args.local_process_index == 0

            if is_local_rank_0:
                try:
                    checkpoint_dirs = self.remote_fs.ls("", detail=False)
                except FileNotFoundError:
                    # Remote storage path doesn't exist yet (first training run)
                    checkpoint_dirs = []

                steps = sorted(
                    [
                        int(m.group(1))
                        for p in checkpoint_dirs
                        if (m := re.search(r"checkpoint-(\d+)$", p))
                    ],
                    reverse=True,
                )

                for step in steps:
                    name = f"checkpoint-{step}"
                    try:
                        entries = self.remote_fs.ls(name, detail=False)
                    except FileNotFoundError:
                        entries = []
                    if any(CHECKPOINT_INCOMPLETE_MARKER in entry for entry in entries):
                        continue

                    try:
                        _log(f"Downloading checkpoint: {name}", args=args)
                        # Calculate remote directory size
                        remote_size = self.remote_fs.du(name, total=True, maxdepth=None)
                        self.remote_fs.get(
                            name,
                            args.output_dir,
                            recursive=True,
                            callback=self._cloud_storage_progress_callback("Download", remote_size),
                        )
                        _log("Download complete", args=args)

                    except Exception as e:
                        raise RuntimeError(
                            "[Kubeflow] Checkpoint download failed for "
                            f"'{name}' to '{args.output_dir}': {e}. "
                            "This may be caused by network issues, insufficient permissions, "
                            "or lack of disk space on the training node. "
                            "Verify access to the remote storage location, ensure adequate "
                            "free disk space, and retry the training job. "
                            "If partial files were downloaded, delete the local checkpoint "
                            "and retry."
                        ) from e
                    break
                else:
                    # Loop completed without break - either no checkpoints or all incomplete
                    _log(
                        "No existing or valid checkpoints found in cloud storage. "
                        "Training will start from scratch.",
                        args=args,
                    )

            # Barrier to wait for local rank 0 checkpoint download to complete
            wait_for_all_ranks("download")

        def on_save(self, args, state, control, **kwargs):
            """Stage checkpoint and queue async upload."""
            if not self.remote_fs:
                # S3 storage not configured, skip upload
                return

            # Barrier before staging checkpoint to ensure all ranks finished saving their files
            wait_for_all_ranks("save")

            # Check for background upload errors and propagate to main thread
            self._check_upload_error()

            is_local_rank_0 = args.local_process_index == 0
            # Only local rank 0 stages and queues uploads
            if is_local_rank_0:
                current_step = state.global_step
                checkpoint_name = f"{PREFIX_CHECKPOINT_DIR}-{current_step}"
                checkpoint_path = os.path.join(args.output_dir, checkpoint_name)

                # Verify checkpoint exists
                if not os.path.exists(checkpoint_path):
                    _log(
                        f"Warning: Checkpoint {checkpoint_path} not found, skipping upload",
                        args=args,
                    )
                else:
                    # Create staging directory (prevents save_total_limit race condition)
                    staging_dir = os.path.join(args.output_dir, CHECKPOINT_STAGING_DIR)
                    os.makedirs(staging_dir, exist_ok=True)
                    staging_checkpoint_path = os.path.join(staging_dir, checkpoint_name)

                    # Move checkpoint to staging (instant metadata operation, not a copy)
                    _log(
                        f"Moving checkpoint to staging: {checkpoint_name}",
                        args=args,
                    )
                    shutil.move(checkpoint_path, staging_checkpoint_path)

                    # Calculate directory size for progress tracking
                    total_size = self._calculate_local_dir_size(staging_checkpoint_path)

                    # Start upload worker if not yet running
                    self.start_upload_worker()
                    # Queue upload task (non-blocking, training continues immediately)
                    node = os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK") or "0"
                    marker_name = (
                        f"{CHECKPOINT_INCOMPLETE_MARKER}.node-{node}-"
                        f"rank-{args.local_process_index}"
                    )
                    # Create marker at enqueue time so resume won't treat queued uploads
                    # as complete if the node specific upload worker hasn't started yet.
                    marker_path = f"{checkpoint_name}/{marker_name}"
                    marker_error = self._retry_marker_op(
                        lambda: self.remote_fs.pipe(
                            marker_path,
                            f"Upload queued for {checkpoint_name}".encode(),
                        )
                    )
                    if marker_error:
                        _log(
                            f"Warning: Failed to write incomplete marker for {checkpoint_name}: "
                            f"{marker_error}. "
                            "Upload will continue, but if it fails, partial uploads cannot be detected during "
                            "resume. Check S3 write permissions to prevent corrupted checkpoints. "
                            "Consider deleting the checkpoint file manually."
                        )
                    task = (
                        staging_checkpoint_path,
                        checkpoint_name,
                        total_size,
                        marker_name,
                    )
                    self.upload_queue.put(task)
                    _log(
                        f"Queued checkpoint for async upload: {checkpoint_name}",
                        args=args,
                    )

        def on_train_begin(self, args, state, control, **kwargs):
            if self._trainer_ref is not None and self.jit_manager is None:
                self.jit_manager = CheckpointManager(trainer=self._trainer_ref)
                self.jit_manager.setup_signal_handler()
                _log("JIT checkpointing enabled", args=args)
            elif self._trainer_ref is None:
                _log(
                    "Warning: Trainer reference not set for JIT checkpoint callback",
                    args=args,
                )

        def on_step_begin(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.checkpoint_in_progress():
                control.should_training_stop = True

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.checkpoint_in_progress():
                control.should_training_stop = True

        def on_optimizer_step(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.checkpoint_in_progress():
                control.should_training_stop = True

        def on_step_end(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.checkpoint_in_progress():
                self.jit_manager._save_jit_checkpoint()
                control.should_save = False
                control.should_training_stop = True

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.checkpoint_in_progress():
                self.jit_manager._save_jit_checkpoint()
                control.should_save = False
                control.should_training_stop = True

        def on_train_end(self, args, state, control, **kwargs):
            """Clean up S3 staging directory and wait for pending uploads.

            If JIT checkpoint was taken, exits via sys.exit(0) to prevent
            redundant user code from running during graceful shutdown.
            """
            if not self.remote_fs:
                if self.jit_manager and self.jit_manager._should_exit:
                    _log(
                        "JIT checkpoint complete. Exiting to avoid redundant "
                        "operations during graceful shutdown.",
                        args=args,
                    )
                    sys.exit(0)
                return

            is_local_rank_0 = args.local_process_index == 0
            if is_local_rank_0:
                # Shutdown background upload worker and wait for all uploads to complete
                self.shutdown_upload_worker()

                # Check for any errors that occurred during final uploads
                self._check_upload_error()

                # Clean up staging directory
                staging_dir = os.path.join(args.output_dir, CHECKPOINT_STAGING_DIR)
                if os.path.exists(staging_dir):
                    try:
                        shutil.rmtree(staging_dir)
                        _log("Deleted staging directory", args=args)
                    except Exception as e:
                        _log(
                            f"Warning: Staging cleanup failed: {e}. "
                            "Staging data will be cleaned up when the pod terminates.",
                            args=args,
                        )

            # After JIT checkpoint + S3 upload + cleanup, exit the process to prevent
            # user code (e.g. trainer.save_model()) from running during shutdown.
            if self.jit_manager and self.jit_manager._should_exit:
                _log(
                    "JIT checkpoint and upload complete. Exiting to avoid redundant "
                    "operations during graceful shutdown.",
                    args=args,
                )
                sys.exit(0)

    # Create callback instance at outer scope so both apply_checkpointing()
    # and upload_final_model_to_cloud() can reference it via closure.
    _jit_checkpoint_callback = JITCheckpointCallback(
        checkpoint_config.get("cloud_remote_storage_uri")
    )

    def apply_checkpointing():
        """Setup monkey patch for Trainer to auto inject JIT checkpoint callback."""
        from transformers import Trainer as _TransformersTrainer

        def _find_latest_checkpoint(output_dir):
            """Find the latest checkpoint and deleting incomplete ones."""
            if not output_dir or not os.path.exists(output_dir):
                return None

            # Only global rank-0 deletes incomplete checkpoints
            is_global_rank_0 = True  # Default for single node process
            if dist.is_available() and dist.is_initialized():
                is_global_rank_0 = dist.get_rank() == 0

            checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
            checkpoints = []

            for name in os.listdir(output_dir):
                match = checkpoint_pattern.match(name)
                if not match or not os.path.isdir(os.path.join(output_dir, name)):
                    continue

                checkpoint_path = os.path.join(output_dir, name)

                try:
                    has_incomplete = any(
                        f.startswith(CHECKPOINT_INCOMPLETE_MARKER)
                        for f in os.listdir(checkpoint_path)
                    )
                except (FileNotFoundError, OSError):
                    _log(f"Skipping checkpoint {name} directory was removed by another rank")
                    continue

                # Delete incomplete checkpoints (global rank 0 only to avoid race condition)
                if has_incomplete:
                    if is_global_rank_0:
                        try:
                            _log(f"Deleting incomplete checkpoint: {checkpoint_path}")
                            shutil.rmtree(checkpoint_path)
                        except Exception as e:
                            _log(
                                "Warning: Failed to delete incomplete checkpoint "
                                f"'{checkpoint_path}': {e}. "
                                "Manually delete it to free up storage."
                            )
                    continue

                checkpoints.append((int(match.group(1)), checkpoint_path))

            if checkpoints:
                checkpoints.sort(reverse=True)
                latest = checkpoints[0][1]
                _log(f"Found latest checkpoint: {latest}")
                return latest

            return None

        # Store original __init__ method
        _original_trainer_init = _TransformersTrainer.__init__

        def _patched_trainer_init(self, *args, **kwargs):
            """Patched Trainer.__init__ that auto-injects JIT checkpoint callback."""
            enable_jit = checkpoint_config.get("enable_jit", False)

            # Extract TrainingArguments to patch
            training_args = kwargs.get("args")
            if not training_args and len(args) > 1:
                training_args = args[1]

            # Apply Kubeflow checkpoint config to training_args
            if training_args and checkpoint_config:
                if getattr(training_args, "save_only_model", False):
                    raise ValueError(
                        "save_only_model=True is incompatible with Kubeflow checkpointing. "
                        "When enabled, only model weights are saved, but optimizer state, "
                        "scheduler state, and RNG state are excluded. This prevents resuming "
                        "training from the exact point of interruption, breaking fault tolerance. "
                        "\nTo fix: Set save_only_model=False in your TrainingArguments, or "
                        "remove it entirely (defaults to False)."
                    )

                # Apply output_dir if provided by user
                if "output_dir" in checkpoint_config:
                    training_args.output_dir = checkpoint_config["output_dir"]
                    _log(
                        f"Applied output_dir: {checkpoint_config['output_dir']}",
                        args=training_args,
                    )

                if "save_strategy" in checkpoint_config:
                    training_args.save_strategy = checkpoint_config["save_strategy"]
                    _log(
                        f"Applied save_strategy: {checkpoint_config['save_strategy']}",
                        args=training_args,
                    )

                if (
                    "save_steps" in checkpoint_config
                    and checkpoint_config["save_steps"] is not None
                ):
                    training_args.save_steps = checkpoint_config["save_steps"]
                    _log(
                        f"Applied save_steps: {checkpoint_config['save_steps']}",
                        args=training_args,
                    )

                if "save_total_limit" in checkpoint_config:
                    training_args.save_total_limit = checkpoint_config["save_total_limit"]
                    _log(
                        f"Applied save_total_limit: {checkpoint_config['save_total_limit']}",
                        args=training_args,
                    )

                if checkpoint_config.get("cloud_remote_storage_uri") and getattr(
                    training_args, "save_on_each_node", False
                ):
                    raise ValueError(
                        "save_on_each_node=True is not supported when output_dir is an S3 URI. "
                        "This would duplicate full checkpoints on every node and waste bandwidth. "
                        "Set save_on_each_node=False or use a PVC-backed output_dir instead."
                    )

            # Inject JIT callback if enabled
            if enable_jit:
                callbacks = kwargs.get("callbacks") or []
                if not isinstance(callbacks, list):
                    callbacks = list(callbacks)
                if not any(isinstance(cb, JITCheckpointCallback) for cb in callbacks):
                    callbacks.append(_jit_checkpoint_callback)
                    _log("Auto-injected JIT checkpoint callback", args=training_args)
                kwargs["callbacks"] = callbacks

            # Call original __init__
            _original_trainer_init(self, *args, **kwargs)

            # Store trainer reference in callback
            if enable_jit:
                _jit_checkpoint_callback._trainer_ref = self

            _original_train = self.train

            def _patched_train(resume_from_checkpoint=None, **train_kwargs):
                """Patched train() that auto-resumes from latest checkpoint if available."""
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
                    world_size = dist.get_world_size()
                    _log(
                        f"Rank {rank}/{world_size} - "
                        "Waiting for all ranks before training starts...",
                        args=training_args,
                    )
                    try:
                        # Specify device to avoid "guessing device ID" warning
                        if torch.cuda.is_available():
                            dist.barrier(device_ids=[torch.cuda.current_device()])
                        else:
                            dist.barrier()
                    except Exception as e:
                        raise RuntimeError(
                            "[Kubeflow] Barrier synchronization failed during train-start: "
                            f"{e}. This typically indicates one or more training processes "
                            "crashed or exited early. Check your training logs to identify "
                            "which rank failed, verify all pods are healthy, and ensure "
                            "distributed training is properly configured. Retrying the "
                            "training job often resolves transient issues."
                        ) from e
                    _log(
                        f"Rank {rank}/{world_size} - All ranks synchronized, proceeding...",
                        args=training_args,
                    )

                # Only auto-resume if user didn't explicitly set it
                if resume_from_checkpoint is None and training_args:
                    latest_checkpoint = _find_latest_checkpoint(training_args.output_dir)
                    if latest_checkpoint:
                        resume_from_checkpoint = latest_checkpoint
                        _log(f"Auto-resuming from: {latest_checkpoint}", args=training_args)
                return _original_train(
                    resume_from_checkpoint=resume_from_checkpoint, **train_kwargs
                )

            self.train = _patched_train

        # Apply monkey-patch
        _TransformersTrainer.__init__ = _patched_trainer_init
        _log("Trainer auto-instrumentation enabled")

    def upload_final_model_to_cloud():
        """Upload final model artifacts from output_dir to cloud storage."""

        if not checkpoint_config.get("cloud_remote_storage_uri"):
            return

        # Ensure all ranks finish writing final artifacts before upload starts.
        wait_for_all_ranks("final_model_upload")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            return

        output_dir = checkpoint_config.get("output_dir")
        if not output_dir or not os.path.exists(output_dir):
            return

        _log("Uploading final model artifacts to S3")

        try:
            uploaded_count = 0
            for item in os.listdir(output_dir):
                if item.startswith("checkpoint-") or item in (
                    CHECKPOINT_STAGING_DIR,
                    ".cache",
                ):
                    continue

                local_path = os.path.join(output_dir, item)
                if os.path.isfile(local_path):
                    size = os.path.getsize(local_path)
                    _jit_checkpoint_callback.remote_fs.put_file(
                        local_path,
                        item,
                        callback=_jit_checkpoint_callback._cloud_storage_progress_callback(
                            "Upload", size
                        ),
                    )
                    uploaded_count += 1
                elif os.path.isdir(local_path):
                    size = _jit_checkpoint_callback._calculate_local_dir_size(local_path)
                    _jit_checkpoint_callback.remote_fs.put(
                        local_path,
                        item,
                        recursive=True,
                        callback=_jit_checkpoint_callback._cloud_storage_progress_callback(
                            "Upload", size
                        ),
                    )
                    uploaded_count += 1

            if uploaded_count == 0:
                _log("No final model artifacts to upload")
                return

            _log("Final model upload complete")

        except Exception as e:
            _log(
                f"Warning: Final model upload failed: {e}. "
                "Training completed successfully but final model artifacts "
                "may not be fully available in cloud storage."
            )

    enable_jit = checkpoint_config.get("enable_jit", False)
    return (
        CheckpointManager if enable_jit else None,
        JITCheckpointCallback if enable_jit else None,
        apply_checkpointing,
        upload_final_model_to_cloud,
    )
