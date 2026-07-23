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

"""Unsloth/HuggingFace adapter for TrainingHubCallback.

Translates TrainingHubCallback instances into HuggingFace TrainerCallback objects
so they can be passed to any HuggingFace Trainer, TRL SFTTrainer, or Unsloth trainer.

Note: The transformers import is deferred to class body/usage time to match the
existing pattern in this codebase (transformers is not always installed).
"""

from __future__ import annotations

import logging

from kubeflow.trainer.rhai.callbacks import TrainingHubCallback, TrainingHubContext

logger = logging.getLogger(__name__)


class UnslothCallbackAdapter:
    """Adapts a TrainingHubCallback to HuggingFace TrainerCallback interface.

    Wraps a single TrainingHubCallback, building a normalized TrainingHubContext
    from HuggingFace's TrainerState/TrainerControl/TrainingArguments and delegating
    to the corresponding hub callback hook. Exceptions are caught and logged to
    ensure callbacks never crash training.

    This class dynamically inherits from transformers.TrainerCallback at instantiation
    time to avoid a top-level import of transformers (which may not be installed in
    all environments).

    Args:
        hub_callback: A TrainingHubCallback instance to adapt.
    """

    def __init__(self, hub_callback: TrainingHubCallback) -> None:
        self._hub_callback = hub_callback

    def __getattr__(self, name: str):
        """Return a no-op for any HF callback hook we don't explicitly handle."""
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def _build_context(
        self,
        args,
        state,
        control,
        logs: dict | None = None,
    ) -> TrainingHubContext:
        """Build TrainingHubContext from HuggingFace state objects.

        Args:
            args: HuggingFace TrainingArguments (immutable config).
            state: HuggingFace TrainerState (training progress).
            control: HuggingFace TrainerControl (flow control flags).
            logs: Optional log dictionary with current metrics.

        Returns:
            Normalized TrainingHubContext for the hub callback.
        """
        return TrainingHubContext(
            step=state.global_step or 0,
            epoch=int(state.epoch) if state.epoch is not None else 0,
            loss=logs.get("loss") if logs else None,
            learning_rate=logs.get("learning_rate") if logs else None,
            is_main_process=state.is_world_process_zero,
            output_dir=args.output_dir,
            metrics=dict(logs) if logs else {},
        )

    def _safe_call(self, method_name: str, *args, **kwargs) -> None:
        """Call a hub callback method with exception isolation.

        Args:
            method_name: Name of the hook method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
        """
        try:
            getattr(self._hub_callback, method_name)(*args, **kwargs)
        except Exception:
            logger.warning(
                "Exception in TrainingHubCallback.%s (callback=%s), ignoring.",
                method_name,
                type(self._hub_callback).__name__,
                exc_info=True,
            )

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_train_begin."""
        self._safe_call("on_train_begin", self._build_context(args, state, control))

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_epoch_begin."""
        self._safe_call("on_epoch_begin", self._build_context(args, state, control))

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_step_begin."""
        self._safe_call("on_step_begin", self._build_context(args, state, control))

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """Delegate to hub callback on_log."""
        self._safe_call("on_log", self._build_context(args, state, control, logs))

    def on_evaluate(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_evaluate."""
        self._safe_call("on_evaluate", self._build_context(args, state, control))

    def on_save(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_save."""
        self._safe_call("on_save", self._build_context(args, state, control))

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_step_end."""
        self._safe_call("on_step_end", self._build_context(args, state, control))

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_epoch_end."""
        self._safe_call("on_epoch_end", self._build_context(args, state, control))

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """Delegate to hub callback on_train_end."""
        self._safe_call("on_train_end", self._build_context(args, state, control))


def adapt_hub_callbacks(
    callbacks: list[TrainingHubCallback],
) -> list[UnslothCallbackAdapter]:
    """Convert a list of TrainingHubCallback instances to HuggingFace-compatible adapters.

    Args:
        callbacks: List of TrainingHubCallback instances.

    Returns:
        List of UnslothCallbackAdapter instances ready for HuggingFace Trainer.
    """
    return [UnslothCallbackAdapter(cb) for cb in callbacks]
