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

"""Unified callback abstraction for Training Hub.

Provides a single callback interface that works across all Training Hub backends
(InstructLab Training, Mini-Trainer, Unsloth/HuggingFace). Users subclass
TrainingHubCallback and override the hooks they need. The adapter layer
translates to each backend's native callback system.

Example:
    class MyCallback(TrainingHubCallback):
        def on_log(self, context: TrainingHubContext) -> None:
            if context.is_main_process:
                print(f"Step {context.step}: loss={context.loss}")

    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.SFT,
        func_args={"model": "granite-3b"},
        callbacks=[MyCallback],
    )
"""

from dataclasses import dataclass, field


@dataclass
class TrainingHubContext:
    """Normalized training state passed to all callback hooks.

    Provides unified access to training progress regardless of which backend
    (InstructLab Training, Mini-Trainer, Unsloth/HuggingFace) is running.
    Each adapter builds this from its backend-specific state objects.

    Args:
        step: Current global training step.
        epoch: Current epoch number.
        loss: Most recent training loss value.
        learning_rate: Current learning rate from the scheduler.
        is_main_process: Whether this is the rank-0 (main) process.
        output_dir: Directory where checkpoints are saved.
        metrics: Flattened dictionary of all available metrics.
    """

    step: int = 0
    epoch: int = 0
    loss: float | None = None
    learning_rate: float | None = None
    is_main_process: bool = True
    output_dir: str | None = None
    metrics: dict = field(default_factory=dict)


class TrainingHubCallback:
    """Base class for Training Hub callbacks.

    Subclass and override any hook methods you need. All hooks receive a
    TrainingHubContext with normalized training state. Methods default to
    no-op so you only implement what you need.

    Callbacks are fire-and-forget: exceptions in callbacks do not crash
    training. This matches the upstream design in both InstructLab Training
    and Mini-Trainer.

    Example:
        class MetricsLogger(TrainingHubCallback):
            def on_train_begin(self, context: TrainingHubContext) -> None:
                print(f"Training started")

            def on_log(self, context: TrainingHubContext) -> None:
                if context.is_main_process:
                    print(f"Step {context.step}: loss={context.loss}")

            def on_train_end(self, context: TrainingHubContext) -> None:
                print(f"Training complete at step {context.step}")
    """

    def on_train_begin(self, context: TrainingHubContext) -> None:
        """Called after initialization, before the training loop starts."""

    def on_epoch_begin(self, context: TrainingHubContext) -> None:
        """Called at the start of each epoch."""

    def on_step_begin(self, context: TrainingHubContext) -> None:
        """Called at the start of each training step."""

    def on_log(self, context: TrainingHubContext) -> None:
        """Called when metrics are logged."""

    def on_evaluate(self, context: TrainingHubContext) -> None:
        """Called after validation/evaluation."""

    def on_save(self, context: TrainingHubContext) -> None:
        """Called after a checkpoint is saved."""

    def on_step_end(self, context: TrainingHubContext) -> None:
        """Called at the end of each training step."""

    def on_epoch_end(self, context: TrainingHubContext) -> None:
        """Called at the end of each epoch."""

    def on_train_end(self, context: TrainingHubContext) -> None:
        """Called after training completes."""
