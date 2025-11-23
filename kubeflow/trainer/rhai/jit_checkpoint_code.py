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


class CheckpointManager:
    """Manages async just-in-time checkpointing on SIGTERM signal using CUDA streams."""

    import os
    import signal
    import threading
    import time

    import torch
    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    def __init__(self, trainer):
        self.trainer = trainer
        self.checkpoint_requested = False
        self._original_sigterm_handler = None
        self.checkpoint_stream = None
        self.checkpoint_thread = None
        self._in_optimizer_step = False

        # Initialize CUDA stream for async checkpoint operations
        try:
            if self.torch.cuda.is_available():
                self.checkpoint_stream = self.torch.cuda.Stream()
                print("[Kubeflow] CUDA stream initialized for async checkpointing", flush=True)
        except (ImportError, AttributeError):
            print("[Kubeflow] CUDA not available, checkpointing will be synchronous", flush=True)

    def setup_signal_handler(self):
        """Register SIGTERM signal handler for JIT checkpointing."""
        self._original_sigterm_handler = self.signal.signal(
            self.signal.SIGTERM, self._sigterm_handler
        )
        print("[Kubeflow] JIT checkpoint signal handler registered for SIGTERM", flush=True)

    def _sigterm_handler(self, signum, frame):
        """Handle SIGTERM by starting async checkpoint immediately."""
        if self.checkpoint_requested:
            return

        print("[Kubeflow] SIGTERM received, starting async checkpoint", flush=True)
        self.checkpoint_requested = True

        # Start checkpoint thread immediately
        self.checkpoint_thread = self.threading.Thread(
            target=self._async_checkpoint, daemon=True, name="KubeflowJITCheckpoint"
        )
        self.checkpoint_thread.start()

    def _async_checkpoint(self):
        """Execute checkpoint asynchronously, waiting if in optimizer step."""
        try:
            # Wait if we're currently in optimizer step (unsafe to checkpoint)
            while self._in_optimizer_step:
                self.time.sleep(0.01)

            current_step = self.trainer.state.global_step
            print(f"[Kubeflow] Starting JIT checkpoint at step {current_step}", flush=True)

            output_dir = self.trainer._get_output_dir(trial=None)
            checkpoint_path = self.os.path.join(
                output_dir, f"{self.PREFIX_CHECKPOINT_DIR}-{current_step}"
            )
            self.os.makedirs(checkpoint_path, exist_ok=True)

            # Create sentinel file to mark incomplete checkpoint
            sentinel_file = self.os.path.join(checkpoint_path, "checkpoint-is-incomplete.txt")
            with open(sentinel_file, "w") as f:
                f.write(f"Checkpoint started at step {current_step}")

            # Checkpoint using dedicated CUDA stream
            if self.checkpoint_stream is not None:
                with self.torch.cuda.stream(self.checkpoint_stream):
                    self.trainer._save_checkpoint(self.trainer.model, trial=None)
                self.checkpoint_stream.synchronize()
            else:
                # Fallback if no CUDA stream
                self.trainer._save_checkpoint(self.trainer.model, trial=None)

            # Remove sentinel on success
            if self.os.path.exists(sentinel_file):
                self.os.remove(sentinel_file)

            print(f"[Kubeflow] JIT checkpoint completed at step {current_step}", flush=True)

        except Exception as e:
            print(f"[Kubeflow] Failed to save JIT checkpoint: {e}", flush=True)
            import traceback

            traceback.print_exc()

    def should_checkpoint_now(self):
        """Check if a checkpoint has been requested."""
        return self.checkpoint_requested

    class JITCheckpointCallback(TrainerCallback):
        """Transformers callback that integrates JIT checkpointing with trainer lifecycle."""

        def __init__(self):
            self.jit_manager = None
            self._trainer_ref = None

        def on_train_begin(self, args, state, control, **kwargs):
            if self._trainer_ref is not None and self.jit_manager is None:
                self.jit_manager = CheckpointManager(trainer=self._trainer_ref)
                self.jit_manager.setup_signal_handler()
                print("[Kubeflow] JIT checkpointing enabled", flush=True)
            elif self._trainer_ref is None:
                print(
                    "[Kubeflow] Warning: Trainer reference not set for JIT checkpoint callback",
                    flush=True,
                )

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if self.jit_manager:
                # Mark that we're entering optimizer step (unsafe for checkpoint)
                self.jit_manager._in_optimizer_step = True

                if self.jit_manager.should_checkpoint_now():
                    control.should_training_stop = True

        def on_optimizer_step(self, args, state, control, **kwargs):
            if self.jit_manager:
                # Mark that optimizer step completed (safe for checkpoint again)
                self.jit_manager._in_optimizer_step = False

        def on_step_end(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.should_checkpoint_now():
                control.should_save = False
                control.should_training_stop = True

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.jit_manager and self.jit_manager.should_checkpoint_now():
                control.should_save = False
                control.should_training_stop = True


def setup_jit_checkpoint_monkey_patch():
    """Setup monkey patch for Trainer to auto inject JIT checkpoint callback."""
    from transformers import Trainer as _TransformersTrainer

    _jit_checkpoint_callback = CheckpointManager.JITCheckpointCallback()

    # Store original __init__ method
    _original_trainer_init = _TransformersTrainer.__init__

    def _patched_trainer_init(self, *args, **kwargs):
        """Patched Trainer.__init__ that auto-injects JIT checkpoint callback."""
        config = globals().get("_KUBEFLOW_CHECKPOINT_CONFIG", {})
        enable_jit = config.get("enable_jit", False)

        # Extract TrainingArguments to patch
        training_args = kwargs.get("args")
        if not training_args and len(args) > 1:
            training_args = args[1]

        # Apply Kubeflow checkpoint config to training_args
        if training_args and config:
            # Apply output_dir if provided by user
            if "output_dir" in config:
                training_args.output_dir = config["output_dir"]
                print(f"[Kubeflow] Applied output_dir: {config['output_dir']}", flush=True)

            if "save_strategy" in config:
                training_args.save_strategy = config["save_strategy"]
                print(f"[Kubeflow] Applied save_strategy: {config['save_strategy']}", flush=True)

            if "save_steps" in config and config["save_steps"] is not None:
                training_args.save_steps = config["save_steps"]
                print(f"[Kubeflow] Applied save_steps: {config['save_steps']}", flush=True)

            if "save_total_limit" in config:
                training_args.save_total_limit = config["save_total_limit"]
                print(
                    f"[Kubeflow] Applied save_total_limit: {config['save_total_limit']}", flush=True
                )

        # Inject JIT callback if enabled
        if enable_jit:
            callbacks = kwargs.get("callbacks") or []
            if not isinstance(callbacks, list):
                callbacks = list(callbacks)
            if not any(isinstance(cb, CheckpointManager.JITCheckpointCallback) for cb in callbacks):
                callbacks.append(_jit_checkpoint_callback)
                print("[Kubeflow] Auto-injected JIT checkpoint callback", flush=True)
            kwargs["callbacks"] = callbacks

        # Call original __init__
        _original_trainer_init(self, *args, **kwargs)

        # Store trainer reference in callback
        if enable_jit:
            _jit_checkpoint_callback._trainer_ref = self

    # Apply monkey-patch
    _TransformersTrainer.__init__ = _patched_trainer_init
    print("[Kubeflow] Trainer auto-instrumentation enabled", flush=True)
