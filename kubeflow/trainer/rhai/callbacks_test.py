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

"""Tests for TrainingHubCallback, TrainingHubContext, and Unsloth adapter."""

from unittest.mock import MagicMock

from kubeflow.trainer.rhai.adapters.unsloth import UnslothCallbackAdapter, adapt_hub_callbacks
from kubeflow.trainer.rhai.callbacks import TrainingHubCallback, TrainingHubContext

# ---------- Test helpers ----------


class RecordingCallback(TrainingHubCallback):
    """Test callback that records which hooks were called and with what context."""

    def __init__(self):
        self.calls: list[tuple[str, TrainingHubContext]] = []

    def on_train_begin(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_train_begin", context))

    def on_epoch_begin(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_epoch_begin", context))

    def on_step_begin(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_step_begin", context))

    def on_log(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_log", context))

    def on_evaluate(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_evaluate", context))

    def on_save(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_save", context))

    def on_step_end(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_step_end", context))

    def on_epoch_end(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_epoch_end", context))

    def on_train_end(self, context: TrainingHubContext) -> None:
        self.calls.append(("on_train_end", context))


class CrashingCallback(TrainingHubCallback):
    """Test callback that raises on every hook to verify exception isolation."""

    def on_train_begin(self, context: TrainingHubContext) -> None:
        raise RuntimeError("deliberate crash in on_train_begin")

    def on_log(self, context: TrainingHubContext) -> None:
        raise ValueError("deliberate crash in on_log")

    def on_train_end(self, context: TrainingHubContext) -> None:
        raise TypeError("deliberate crash in on_train_end")


def _make_hf_mocks(
    global_step: int = 100,
    epoch: float = 2.0,
    max_steps: int = 500,
    output_dir: str = "/tmp/output",
    is_world_process_zero: bool = True,
    logs: dict | None = None,
) -> tuple:
    """Create mock HuggingFace args/state/control for testing."""
    args = MagicMock()
    args.output_dir = output_dir
    args.num_train_epochs = 3

    state = MagicMock()
    state.global_step = global_step
    state.epoch = epoch
    state.max_steps = max_steps
    state.is_world_process_zero = is_world_process_zero
    state.is_local_process_zero = is_world_process_zero

    control = MagicMock()

    return args, state, control, logs


# ---------- TrainingHubContext tests ----------


class TestTrainingHubContext:
    """Tests for TrainingHubContext dataclass."""

    def test_default_values(self):
        """Test context has sensible defaults."""
        print("Executing test: TrainingHubContext default values")
        ctx = TrainingHubContext()
        assert ctx.step == 0
        assert ctx.epoch == 0
        assert ctx.loss is None
        assert ctx.learning_rate is None
        assert ctx.is_main_process is True
        assert ctx.output_dir is None
        assert ctx.metrics == {}
        print("test execution complete")

    def test_custom_values(self):
        """Test context with all fields set."""
        print("Executing test: TrainingHubContext custom values")
        ctx = TrainingHubContext(
            step=42,
            epoch=3,
            loss=0.5,
            learning_rate=1e-4,
            is_main_process=False,
            output_dir="/checkpoints",
            metrics={"grad_norm": 1.2},
        )
        assert ctx.step == 42
        assert ctx.epoch == 3
        assert ctx.loss == 0.5
        assert ctx.learning_rate == 1e-4
        assert ctx.is_main_process is False
        assert ctx.output_dir == "/checkpoints"
        assert ctx.metrics == {"grad_norm": 1.2}
        print("test execution complete")

    def test_metrics_dict_not_shared(self):
        """Test that metrics default factory creates independent dicts."""
        print("Executing test: TrainingHubContext metrics isolation")
        ctx1 = TrainingHubContext()
        ctx2 = TrainingHubContext()
        ctx1.metrics["key"] = "value"
        assert "key" not in ctx2.metrics
        print("test execution complete")


# ---------- TrainingHubCallback ABC tests ----------


class TestTrainingHubCallback:
    """Tests for TrainingHubCallback abstract base class."""

    def test_all_hooks_are_noop_by_default(self):
        """Test that a minimal subclass can be instantiated and all hooks are callable."""
        print("Executing test: TrainingHubCallback all hooks noop")

        class MinimalCallback(TrainingHubCallback):
            pass

        cb = MinimalCallback()
        ctx = TrainingHubContext()
        # None of these should raise
        cb.on_train_begin(ctx)
        cb.on_epoch_begin(ctx)
        cb.on_step_begin(ctx)
        cb.on_log(ctx)
        cb.on_evaluate(ctx)
        cb.on_save(ctx)
        cb.on_step_end(ctx)
        cb.on_epoch_end(ctx)
        cb.on_train_end(ctx)
        print("test execution complete")

    def test_selective_override(self):
        """Test that only overridden hooks fire custom logic."""
        print("Executing test: TrainingHubCallback selective override")

        class OnlyLogCallback(TrainingHubCallback):
            def __init__(self):
                self.logged = False

            def on_log(self, context: TrainingHubContext) -> None:
                self.logged = True

        cb = OnlyLogCallback()
        ctx = TrainingHubContext(step=10, loss=0.3)
        cb.on_train_begin(ctx)  # no-op
        cb.on_log(ctx)
        assert cb.logged is True
        print("test execution complete")

    def test_recording_callback_captures_all_hooks(self):
        """Test RecordingCallback records every hook call."""
        print("Executing test: TrainingHubCallback recording all hooks")
        cb = RecordingCallback()
        ctx = TrainingHubContext(step=5)

        hooks = [
            "on_train_begin",
            "on_epoch_begin",
            "on_step_begin",
            "on_log",
            "on_evaluate",
            "on_save",
            "on_step_end",
            "on_epoch_end",
            "on_train_end",
        ]
        for hook in hooks:
            getattr(cb, hook)(ctx)

        assert len(cb.calls) == 9
        assert [name for name, _ in cb.calls] == hooks
        print("test execution complete")


# ---------- UnslothCallbackAdapter tests ----------


class TestUnslothCallbackAdapter:
    """Tests for UnslothCallbackAdapter."""

    def test_context_mapping(self):
        """Test that HF state/args correctly map to TrainingHubContext fields."""
        print("Executing test: UnslothCallbackAdapter context mapping")
        cb = RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args, state, control, _ = _make_hf_mocks(global_step=150, epoch=2.5, output_dir="/out")
        adapter.on_train_begin(args, state, control)

        assert len(cb.calls) == 1
        _, ctx = cb.calls[0]
        assert ctx.step == 150
        assert ctx.epoch == 2
        assert ctx.output_dir == "/out"
        assert ctx.is_main_process is True
        print("test execution complete")

    def test_log_metrics_mapping(self):
        """Test that logs dict maps to context.loss, learning_rate, and metrics."""
        print("Executing test: UnslothCallbackAdapter log metrics")
        cb = RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args, state, control, _ = _make_hf_mocks(global_step=50)
        logs = {"loss": 0.42, "learning_rate": 5e-5, "grad_norm": 1.1}
        adapter.on_log(args, state, control, logs=logs)

        _, ctx = cb.calls[0]
        assert ctx.loss == 0.42
        assert ctx.learning_rate == 5e-5
        assert ctx.metrics == {"loss": 0.42, "learning_rate": 5e-5, "grad_norm": 1.1}
        print("test execution complete")

    def test_all_hooks_dispatch(self):
        """Test that every adapter hook dispatches to the correct hub callback hook."""
        print("Executing test: UnslothCallbackAdapter all hooks dispatch")
        cb = RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args, state, control, _ = _make_hf_mocks()

        adapter.on_train_begin(args, state, control)
        adapter.on_epoch_begin(args, state, control)
        adapter.on_step_begin(args, state, control)
        adapter.on_log(args, state, control, logs={"loss": 0.1})
        adapter.on_evaluate(args, state, control)
        adapter.on_save(args, state, control)
        adapter.on_step_end(args, state, control)
        adapter.on_epoch_end(args, state, control)
        adapter.on_train_end(args, state, control)

        hook_names = [name for name, _ in cb.calls]
        assert hook_names == [
            "on_train_begin",
            "on_epoch_begin",
            "on_step_begin",
            "on_log",
            "on_evaluate",
            "on_save",
            "on_step_end",
            "on_epoch_end",
            "on_train_end",
        ]
        print("test execution complete")

    def test_exception_isolation(self):
        """Test that exceptions in hub callbacks don't propagate through adapter."""
        print("Executing test: UnslothCallbackAdapter exception isolation")
        cb = CrashingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args, state, control, _ = _make_hf_mocks()

        # None of these should raise despite the callback crashing
        adapter.on_train_begin(args, state, control)
        adapter.on_log(args, state, control, logs={"loss": 0.5})
        adapter.on_train_end(args, state, control)
        print("test execution complete")

    def test_non_main_process(self):
        """Test context correctly reflects non-main process."""
        print("Executing test: UnslothCallbackAdapter non-main process")
        cb = RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args, state, control, _ = _make_hf_mocks(is_world_process_zero=False)
        adapter.on_step_end(args, state, control)

        _, ctx = cb.calls[0]
        assert ctx.is_main_process is False
        print("test execution complete")

    def test_none_logs_handled(self):
        """Test adapter handles None logs gracefully."""
        print("Executing test: UnslothCallbackAdapter None logs")
        cb = RecordingCallback()
        adapter = UnslothCallbackAdapter(cb)

        args, state, control, _ = _make_hf_mocks()
        adapter.on_log(args, state, control, logs=None)

        _, ctx = cb.calls[0]
        assert ctx.loss is None
        assert ctx.learning_rate is None
        assert ctx.metrics == {}
        print("test execution complete")


# ---------- adapt_hub_callbacks utility tests ----------


class TestAdaptHubCallbacks:
    """Tests for adapt_hub_callbacks helper."""

    def test_converts_list(self):
        """Test that a list of hub callbacks becomes a list of adapters."""
        print("Executing test: adapt_hub_callbacks converts list")
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        adapters = adapt_hub_callbacks([cb1, cb2])

        assert len(adapters) == 2
        assert all(isinstance(a, UnslothCallbackAdapter) for a in adapters)
        print("test execution complete")

    def test_empty_list(self):
        """Test that empty input returns empty output."""
        print("Executing test: adapt_hub_callbacks empty list")
        assert adapt_hub_callbacks([]) == []
        print("test execution complete")


# =============================================================================
# E2E: _build_callback_injection_code
# =============================================================================


class TestBuildCallbackInjectionCode:
    """Tests for the code generation that injects callbacks into HF Trainer pods."""

    def test_single_callback_generates_valid_code(self):
        """Generated code imports SDK, defines user class, and patches Trainer."""
        print("Executing test: single callback injection code generation")
        from kubeflow.trainer.rhai.test_fixtures.sample_callbacks import LoggingCallback
        from kubeflow.trainer.rhai.transformers import _build_callback_injection_code

        code = _build_callback_injection_code([LoggingCallback])

        # Should import SDK classes from installed package
        assert "from kubeflow.trainer.rhai.callbacks import" in code
        assert "from kubeflow.trainer.rhai.adapters.unsloth import" in code
        # Should contain the user's class definition
        assert "class LoggingCallback" in code
        # Should monkey-patch Trainer.__init__
        assert "_cb_transformers.Trainer.__init__" in code
        # Should instantiate the callback
        assert "LoggingCallback()" in code
        print("test execution complete")

    def test_multiple_callbacks(self):
        """Multiple callbacks are all serialized and instantiated."""
        print("Executing test: multiple callbacks injection code generation")
        from kubeflow.trainer.rhai.test_fixtures.sample_callbacks import (
            EarlyStopCallback,
            LoggingCallback,
        )
        from kubeflow.trainer.rhai.transformers import _build_callback_injection_code

        code = _build_callback_injection_code([LoggingCallback, EarlyStopCallback])

        assert "class LoggingCallback" in code
        assert "class EarlyStopCallback" in code
        assert "LoggingCallback(), EarlyStopCallback()" in code
        print("test execution complete")

    def test_generated_code_is_syntactically_valid(self):
        """Generated code can be compiled without syntax errors."""
        print("Executing test: generated code syntax validation")
        from kubeflow.trainer.rhai.test_fixtures.sample_callbacks import LoggingCallback
        from kubeflow.trainer.rhai.transformers import _build_callback_injection_code

        code = _build_callback_injection_code([LoggingCallback])
        # compile() raises SyntaxError if invalid
        compile(code, "<generated>", "exec")
        print("test execution complete")

    def test_instance_passed_uses_class(self):
        """Passing an instance serializes its class instead."""
        print("Executing test: instance callback uses class source")
        from kubeflow.trainer.rhai.test_fixtures.sample_callbacks import LoggingCallback
        from kubeflow.trainer.rhai.transformers import _build_callback_injection_code

        instance = LoggingCallback()
        code = _build_callback_injection_code([instance])

        assert "class LoggingCallback" in code
        assert "LoggingCallback()" in code
        print("test execution complete")

    def test_transformers_trainer_callbacks_field(self):
        """TransformersTrainer accepts callbacks field."""
        print("Executing test: TransformersTrainer callbacks field")
        from kubeflow.trainer.rhai.test_fixtures.sample_callbacks import LoggingCallback
        from kubeflow.trainer.rhai.transformers import TransformersTrainer

        def dummy_train():
            pass

        trainer = TransformersTrainer(func=dummy_train, callbacks=[LoggingCallback])
        assert trainer.callbacks == [LoggingCallback]

        # Without callbacks defaults to None
        trainer_no_cb = TransformersTrainer(func=dummy_train)
        assert trainer_no_cb.callbacks is None
        print("test execution complete")

    def test_get_trainer_cr_includes_callback_injection(self):
        """get_trainer_cr_from_transformers_trainer injects callback code into command."""
        print("Executing test: CRD generation includes callback injection")
        from unittest.mock import MagicMock

        from kubeflow.trainer.constants import constants
        from kubeflow.trainer.rhai.test_fixtures.sample_callbacks import LoggingCallback
        from kubeflow.trainer.rhai.transformers import (
            TransformersTrainer,
            get_trainer_cr_from_transformers_trainer,
        )

        def my_train_func():
            pass

        trainer = TransformersTrainer(
            func=my_train_func,
            callbacks=[LoggingCallback],
            enable_progression_tracking=False,
        )

        # Mock the runtime with actual TORCH_COMMAND template
        runtime = MagicMock()
        runtime.trainer.command = list(constants.TORCH_COMMAND)
        runtime.trainer.framework = "pytorch"

        trainer_cr = get_trainer_cr_from_transformers_trainer(runtime, trainer)

        # The command's bash script should contain callback injection code
        bash_script = trainer_cr.command[2]  # "bash", "-c", <script>
        assert "Callback Injection" in bash_script
        assert "LoggingCallback" in bash_script
        assert "UnslothCallbackAdapter" in bash_script
        print("test execution complete")

    def test_get_trainer_cr_no_callbacks_no_injection(self):
        """Without callbacks, no injection code is added."""
        print("Executing test: CRD generation without callbacks has no injection")
        from unittest.mock import MagicMock

        from kubeflow.trainer.constants import constants
        from kubeflow.trainer.rhai.transformers import (
            TransformersTrainer,
            get_trainer_cr_from_transformers_trainer,
        )

        def my_train_func():
            pass

        trainer = TransformersTrainer(
            func=my_train_func,
            enable_progression_tracking=False,
        )

        runtime = MagicMock()
        runtime.trainer.command = list(constants.TORCH_COMMAND)
        runtime.trainer.framework = "pytorch"

        trainer_cr = get_trainer_cr_from_transformers_trainer(runtime, trainer)

        bash_script = trainer_cr.command[2]
        assert "Callback Injection" not in bash_script
        assert "UnslothCallbackAdapter" not in bash_script
        print("test execution complete")
