"""Test fixture callbacks for e2e callback injection tests."""

from kubeflow.trainer.rhai.callbacks import TrainingHubCallback, TrainingHubContext


class LoggingCallback(TrainingHubCallback):
    """Fixture callback that logs metrics."""

    def on_log(self, context: TrainingHubContext) -> None:
        print(f"Loss at step {context.step}: {context.loss}")

    def on_train_begin(self, context: TrainingHubContext) -> None:
        print("Training started")


class EarlyStopCallback(TrainingHubCallback):
    """Fixture callback for early stopping checks."""

    def on_step_end(self, context: TrainingHubContext) -> None:
        if context.loss and context.loss < 0.01:
            print("Early stop threshold reached")
