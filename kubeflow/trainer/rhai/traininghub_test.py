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

"""Tests for TrainingHubTrainer and instrumentation wrapper generation."""

import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.traininghub import (
    TrainingHubAlgorithms,
    TrainingHubTrainer,
    get_training_hub_instrumentation_wrapper,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


def test_traininghub_trainer_initialization():
    """Test TrainingHubTrainer initialization with default values."""
    print("Executing test: TrainingHubTrainer initialization with defaults")

    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.SFT,
        func_args={"data_path": "/data/train.jsonl", "ckpt_output_dir": "/tmp/checkpoints"},
    )

    assert trainer.func is None
    assert trainer.func_args == {
        "data_path": "/data/train.jsonl",
        "ckpt_output_dir": "/tmp/checkpoints",
    }
    assert trainer.packages_to_install is None
    assert trainer.pip_index_urls == list(constants.DEFAULT_PIP_INDEX_URLS)
    assert trainer.env is None
    assert trainer.algorithm == TrainingHubAlgorithms.SFT
    # Enabled by default for consistency with TransformersTrainer
    assert trainer.enable_progression_tracking is True
    assert trainer.metrics_port == 28080
    assert trainer.metrics_poll_interval_seconds == 30

    print("test execution complete")


def test_traininghub_trainer_with_custom_config():
    """Test TrainingHubTrainer with custom configuration."""
    print("Executing test: TrainingHubTrainer with custom configuration")

    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.OSFT,
        func_args={
            "data_path": "/data/train.jsonl",
            "ckpt_output_dir": "/tmp/checkpoints",
            "num_epochs": 5,
        },
        packages_to_install=["training-hub"],
        pip_index_urls=["https://custom.pypi.org/simple"],
        env={"WANDB_DISABLED": "true"},
        enable_progression_tracking=True,
        metrics_port=28090,
        metrics_poll_interval_seconds=60,
    )

    assert trainer.algorithm == TrainingHubAlgorithms.OSFT
    assert trainer.func_args["num_epochs"] == 5
    assert trainer.packages_to_install == ["training-hub"]
    assert trainer.pip_index_urls == ["https://custom.pypi.org/simple"]
    assert trainer.env == {"WANDB_DISABLED": "true"}
    assert trainer.enable_progression_tracking is True
    assert trainer.metrics_port == 28090
    assert trainer.metrics_poll_interval_seconds == 60

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="poll interval too low (4 seconds)",
            expected_status="failed",
            config={"metrics_poll_interval_seconds": 4},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval too high (301 seconds)",
            expected_status="failed",
            config={"metrics_poll_interval_seconds": 301},
            expected_error=ValueError,
        ),
        TestCase(
            name="poll interval minimum boundary (5 seconds)",
            expected_status="success",
            config={"metrics_poll_interval_seconds": 5},
            expected_output=5,
        ),
        TestCase(
            name="poll interval maximum boundary (300 seconds)",
            expected_status="success",
            config={"metrics_poll_interval_seconds": 300},
            expected_output=300,
        ),
    ],
)
def test_metrics_poll_interval_validation(test_case):
    """Test metrics_poll_interval_seconds validation."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = TrainingHubTrainer(
            algorithm=TrainingHubAlgorithms.SFT,
            func_args={"ckpt_output_dir": "/tmp"},
            metrics_poll_interval_seconds=test_case.config["metrics_poll_interval_seconds"],
        )

        assert test_case.expected_status == "success"
        assert trainer.metrics_poll_interval_seconds == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == "failed"
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="port too low (1023)",
            expected_status="failed",
            config={"metrics_port": 1023},
            expected_error=ValueError,
        ),
        TestCase(
            name="port too high (65536)",
            expected_status="failed",
            config={"metrics_port": 65536},
            expected_error=ValueError,
        ),
        TestCase(
            name="port minimum boundary (1024)",
            expected_status="success",
            config={"metrics_port": 1024},
            expected_output=1024,
        ),
        TestCase(
            name="port maximum boundary (65535)",
            expected_status="success",
            config={"metrics_port": 65535},
            expected_output=65535,
        ),
    ],
)
def test_metrics_port_validation(test_case):
    """Test metrics_port validation."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = TrainingHubTrainer(
            algorithm=TrainingHubAlgorithms.SFT,
            func_args={"ckpt_output_dir": "/tmp"},
            metrics_port=test_case.config["metrics_port"],
        )

        assert test_case.expected_status == "success"
        assert trainer.metrics_port == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == "failed"
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="basic SFT generation - returns string",
            expected_status="success",
            config={"algorithm": "sft", "ckpt_output_dir": "/tmp/checkpoints", "port": 28080},
            expected_output=[
                ("isinstance", str),
                ("class TrainingHubMetricsHandler", True),
                ("def apply_progression_tracking", True),
                ("def _read_sft_metrics", True),
                ("def _transform_sft", True),
                ('algorithm="sft"', True),
                ("/tmp/checkpoints", True),
                ("metrics_port=28080", True),
            ],
        ),
        TestCase(
            name="basic OSFT generation - returns string",
            expected_status="success",
            config={"algorithm": "osft", "ckpt_output_dir": "/tmp/outputs", "port": 28090},
            expected_output=[
                ("isinstance", str),
                ("class TrainingHubMetricsHandler", True),
                ("def apply_progression_tracking", True),
                ("def _read_osft_metrics", True),
                ("def _transform_osft", True),
                ('algorithm="osft"', True),
                ("/tmp/outputs", True),
                ("metrics_port=28090", True),
            ],
        ),
        TestCase(
            name="self-contained - no SDK imports",
            expected_status="success",
            config={"algorithm": "sft", "ckpt_output_dir": "/tmp", "port": 28080},
            expected_output=[
                ("from kubeflow", False),
                ("import kubeflow", False),
                ("class TrainingHubMetricsHandler", True),
                ("def apply_progression_tracking", True),
            ],
        ),
        TestCase(
            name="structure - function call and constants",
            expected_status="success",
            config={"algorithm": "sft", "ckpt_output_dir": "/tmp", "port": 28080},
            expected_output=[
                ("apply_progression_tracking", True),
                ("apply_progression_tracking()", True),
                ("SFT_METRICS_FILE_RANK0", True),
                ("OSFT_METRICS_FILE_RANK0", True),
                ("OSFT_CONFIG_FILE", True),
            ],
        ),
        TestCase(
            name="completeness - all methods present",
            expected_status="success",
            config={"algorithm": "osft", "ckpt_output_dir": "/tmp", "port": 28080},
            expected_output=[
                ("def _read_latest_metrics", True),
                ("def _read_osft_metrics", True),
                ("def _read_sft_metrics", True),
                ("def _transform_schema", True),
                ("def _transform_osft", True),
                ("def _transform_sft", True),
                ("def do_GET", True),
            ],
        ),
    ],
)
def test_instrumentation_wrapper_content(test_case):
    """Test instrumentation wrapper content contains expected elements."""
    print(f"Executing test: {test_case.name}")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm=test_case.config["algorithm"],
        ckpt_output_dir=test_case.config["ckpt_output_dir"],
        metrics_port=test_case.config["port"],
    )

    assert test_case.expected_status == "success"

    for check_item, expected in test_case.expected_output:
        if check_item == "isinstance":
            assert isinstance(wrapper, expected)
        elif expected:
            assert check_item in wrapper, f"Expected '{check_item}' to be in wrapper"
        else:
            assert check_item not in wrapper, f"Expected '{check_item}' to NOT be in wrapper"

    print("test execution complete")


def test_instrumentation_wrapper_no_syntax_errors():
    """Test that generated wrapper has no obvious syntax errors."""
    print("Executing test: Wrapper syntax validation")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    try:
        compile(wrapper, "<string>", "exec")
        print("  ✓ Wrapper compiles successfully")
    except SyntaxError as e:
        pytest.fail(f"Generated wrapper has syntax error: {e}")

    print("test execution complete")


def test_instrumentation_constants_embedded():
    """Test that file path constants are embedded in the wrapper."""
    print("Executing test: Constants are embedded")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify constants are defined in the wrapper
    assert 'SFT_METRICS_FILE_PATTERN = "training_params_and_metrics_global*.jsonl"' in wrapper
    assert 'SFT_METRICS_FILE_RANK0 = "training_params_and_metrics_global0.jsonl"' in wrapper
    assert 'OSFT_METRICS_FILE_PATTERN = "training_metrics_*.jsonl"' in wrapper
    assert 'OSFT_METRICS_FILE_RANK0 = "training_metrics_0.jsonl"' in wrapper
    assert 'OSFT_CONFIG_FILE = "training_params.json"' in wrapper

    print("test execution complete")


def test_algorithm_parameter_used_not_detected():
    """Test that algorithm parameter is used directly, not detected from metrics."""
    print("Executing test: Algorithm parameter used (no heuristic detection)")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify algorithm parameter is used in conditionals
    assert 'if algorithm == "osft"' in wrapper
    assert 'if algorithm == "sft"' in wrapper or "else:  # sft" in wrapper

    # Verify NO heuristic detection based on metrics keys
    assert 'if "tokens_per_second" in metrics' not in wrapper
    assert 'if "samples_per_second" in metrics' not in wrapper

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="trainer with progression tracking disabled",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": False,
            },
        ),
        TestCase(
            name="trainer with progression tracking enabled (defaults)",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
            },
        ),
        TestCase(
            name="trainer with custom port",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
                "metrics_port": 8888,
            },
        ),
        TestCase(
            name="trainer with long poll interval",
            expected_status=SUCCESS,
            config={
                "enable_progression_tracking": True,
                "metrics_poll_interval_seconds": 120,
            },
        ),
    ],
)
def test_traininghub_trainer_configurations(test_case):
    """Test various TrainingHubTrainer configurations."""
    print(f"Executing test: {test_case.name}")

    try:
        trainer = TrainingHubTrainer(
            algorithm=TrainingHubAlgorithms.SFT,
            func_args={"ckpt_output_dir": "/tmp"},
            **test_case.config,
        )

        assert test_case.expected_status == SUCCESS

        for key, value in test_case.config.items():
            assert getattr(trainer, key) == value

    except Exception as e:
        if test_case.expected_error:
            assert type(e) is test_case.expected_error
        else:
            raise

    print("test execution complete")


def test_progression_tracking_disabled_no_server():
    """Test that wrapper isn't generated when progression tracking is disabled."""
    print("Executing test: Progression tracking disabled")

    from kubeflow.trainer.rhai.traininghub import get_trainer_cr_from_training_hub_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.SFT,
        func_args={"ckpt_output_dir": "/tmp"},
        enable_progression_tracking=False,
    )

    trainer_crd = get_trainer_cr_from_training_hub_trainer(runtime, trainer)

    # Command is split into command and args, script is in args[0]
    script = trainer_crd.args[0] if trainer_crd.args else ""
    # Should NOT contain progression tracking code
    assert "[Kubeflow] Initializing Training Hub progression tracking" not in script
    assert "TrainingHubMetricsHandler" not in script

    print("test execution complete")


def test_progression_tracking_enabled_has_server():
    """Test that wrapper is generated when progression tracking is enabled."""
    print("Executing test: Progression tracking enabled")

    from kubeflow.trainer.rhai.traininghub import get_trainer_cr_from_training_hub_trainer
    from kubeflow.trainer.types import types

    runtime = types.Runtime(
        name="test-runtime",
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="pytorch",
            image="pytorch/pytorch:2.0.0",
        ),
    )
    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.SFT,
        func_args={"ckpt_output_dir": "/tmp"},
        enable_progression_tracking=True,
    )

    trainer_crd = get_trainer_cr_from_training_hub_trainer(runtime, trainer)

    # Command is split into command and args, script is in args[0]
    script = trainer_crd.args[0] if trainer_crd.args else ""
    # Should contain progression tracking code
    assert "[Kubeflow]" in script  # Message is in the script
    assert "TrainingHubMetricsHandler" in script

    print("test execution complete")


def test_instrumentation_wrapper_termination_log_path():
    """Test that termination log path is used in wrapper."""
    print("Executing test: Termination log path used")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify /dev/termination-log path is used directly
    assert '"/dev/termination-log"' in wrapper
    print("test execution complete")


def test_instrumentation_wrapper_termination_method():
    """Test that _maybe_write_termination_message method is in wrapper."""
    print("Executing test: Termination message method in HTTP handler")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify termination message method exists in HTTP handler
    assert "def _maybe_write_termination_message(self, metrics)" in wrapper
    # Verify it checks for 100% progress
    assert "progress >= 100" in wrapper
    # Verify it writes to termination log path
    assert '"/dev/termination-log"' in wrapper
    # Verify it has write-once flag
    assert "_termination_message_written" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_explicit_bind_address():
    """Test that HTTP server binds to 0.0.0.0 explicitly."""
    print("Executing test: Explicit 0.0.0.0 bind address")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify explicit 0.0.0.0 bind (not empty string)
    assert '("0.0.0.0", metrics_port)' in wrapper
    # Verify NOT using empty string bind
    assert '("", metrics_port)' not in wrapper

    print("test execution complete")


def _install_dummy_training_hub(raises: Exception) -> None:
    """Install a dummy training_hub module in sys.modules for wrapper exec().

    The wrapper imports `from training_hub import <algo>` and calls it.
    We provide an implementation that raises the supplied exception to
    verify the wrapper re-raises and surfaces failures.
    """
    import sys
    import types

    module = types.ModuleType("training_hub")

    def sft(**kwargs):  # type: ignore[unused-argument]
        raise raises

    def osft(**kwargs):  # type: ignore[unused-argument]
        raise raises

    module.sft = sft  # type: ignore[attr-defined]
    module.osft = osft  # type: ignore[attr-defined]
    sys.modules["training_hub"] = module


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="sft ValueError is re-raised via CR script",
            expected_status="failed",
            config={"algorithm": "sft", "exception": ValueError("bad config")},
            expected_error=ValueError,
            expected_output="Configuration error:",
        ),
        TestCase(
            name="sft RuntimeError is re-raised via CR script",
            expected_status="failed",
            config={"algorithm": "sft", "exception": RuntimeError("boom")},
            expected_error=RuntimeError,
            expected_output="Training failed with error:",
        ),
    ],
)
def test_traininghub_wrapper_reraises_failure(test_case: TestCase, capsys):
    """Generate wrapper code and assert it re-raises exceptions."""
    print(f"Executing test: {test_case.name}")

    from kubeflow.trainer.rhai.traininghub import _render_algorithm_wrapper

    # Arrange: fake training_hub module that raises
    _install_dummy_training_hub(test_case.config["exception"])

    # Generate the self-contained Python wrapper that the SDK injects into the container
    # command. Executing this code simulates what actually runs inside the training pod
    # (without the surrounding bash heredoc). The dummy training_hub installed above
    # ensures the selected algorithm raises so we can assert failure propagation.
    code = _render_algorithm_wrapper(test_case.config["algorithm"], {"ckpt_output_dir": "/tmp"})

    # Act / Assert
    # Execute the generated wrapper code in-process. Because we installed a dummy
    # training_hub module that raises, the wrapper should re-raise the same exception.
    # This validates failure propagation (so the container exits non‑zero in real runs).
    with pytest.raises(test_case.expected_error):
        exec(code, {})  # nosec: B102 - executing generated test-only code

    # Additionally, verify that the wrapper printed the expected human-readable
    # error message (useful for debugging in container logs).
    out, err = capsys.readouterr()
    combined = out + err
    assert test_case.expected_output in combined

    print("test execution complete")


def test_instrumentation_wrapper_oserror_handling():
    """Test that wrapper has OSError handling for port binding issues."""
    print("Executing test: OSError handling for server start")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify OSError is specifically caught
    assert "except OSError as e:" in wrapper
    # Verify generic Exception is also caught as fallback
    assert "except Exception as e:" in wrapper
    # Verify helpful error message mentions port and server
    assert "Failed to start metrics server on port" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_do_get_try_except_else():
    """Test that do_GET uses try/except/else pattern for clean error handling."""
    print("Executing test: do_GET try/except/else pattern")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Verify try/except/else pattern (send_error in except, send_response after else)
    # The pattern should have send_error(500) in except block
    assert "self.send_error(500)" in wrapper
    # And send_response(200) for success case
    assert "self.send_response(200)" in wrapper

    print("test execution complete")


def test_algorithm_wrapper_termination_message():
    """Test that algorithm wrapper includes termination message writing after training."""
    print("Executing test: Algorithm wrapper termination message (on_train_end)")

    from kubeflow.trainer.rhai.traininghub import _render_algorithm_wrapper

    wrapper = _render_algorithm_wrapper("sft", {"ckpt_output_dir": "/tmp/checkpoints"})

    # Verify _write_termination_message function is defined
    assert "def _write_termination_message(ckpt_output_dir, algorithm):" in wrapper
    # Verify it's called after training completes
    assert "_write_termination_message(ckpt_output_dir, algorithm)" in wrapper
    # Verify termination log path is used
    assert '"/dev/termination-log"' in wrapper
    # Verify it reads metrics files
    assert "training_params_and_metrics_global0.jsonl" in wrapper  # SFT
    assert "training_metrics_0.jsonl" in wrapper  # OSFT
    # Verify docstring explains purpose
    assert "Kubernetes reads /dev/termination-log after container exit" in wrapper

    print("test execution complete")


def test_algorithm_wrapper_termination_handles_errors():
    """Test that algorithm wrapper termination handles errors gracefully."""
    print("Executing test: Algorithm wrapper termination error handling")

    from kubeflow.trainer.rhai.traininghub import _render_algorithm_wrapper

    wrapper = _render_algorithm_wrapper("osft", {"ckpt_output_dir": "/tmp"})

    # Verify PermissionError is handled (not in container)
    assert "except PermissionError:" in wrapper
    # Verify generic Exception is handled
    assert "except Exception as e:" in wrapper
    # Verify helpful messages
    assert "not in container" in wrapper

    print("test execution complete")


def test_instrumentation_wrapper_flush_all_prints():
    """Test that all print statements use flush=True for real-time logging."""
    print("Executing test: All prints use flush=True")

    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/tmp/checkpoints",
        metrics_port=28080,
    )

    # Count print statements and verify they have flush=True
    import re

    print_statements = re.findall(r"print\([^)]+\)", wrapper)
    for stmt in print_statements:
        # Logging statements with [Kubeflow] should have flush=True
        if "flush" not in stmt and "[Kubeflow]" in stmt:
            pytest.fail(f"Print statement missing flush=True: {stmt}")

    print("test execution complete")


def test_instrumentation_cleans_sft_metrics_on_startup(tmp_path):
    """Test that SFT metrics files are cleaned before training starts."""
    print("Executing test: SFT metrics cleanup on startup")

    # Create stale metrics files for multiple ranks (distributed training scenario)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    stale_metrics_file_rank0 = ckpt_dir / "training_params_and_metrics_global0.jsonl"
    stale_metrics_file_rank0.write_text('{"step": 100, "loss": 1.5}\n')

    stale_metrics_file_rank1 = ckpt_dir / "training_params_and_metrics_global1.jsonl"
    stale_metrics_file_rank1.write_text('{"step": 100, "loss": 1.4}\n')

    stale_metrics_file_rank2 = ckpt_dir / "training_params_and_metrics_global2.jsonl"
    stale_metrics_file_rank2.write_text('{"step": 100, "loss": 1.3}\n')

    # Call instrumentation directly (no exec) with port 0 for random available port
    from kubeflow.trainer.rhai.traininghub import _create_training_hub_progression_instrumentation

    apply_fn, _handler = _create_training_hub_progression_instrumentation(
        algorithm="sft",
        ckpt_output_dir=str(ckpt_dir),
        metrics_port=0,  # Use port 0 for random available port
    )

    # Call the function to trigger cleanup and start server
    server = apply_fn()

    try:
        # Verify all stale files from all ranks were removed
        assert not stale_metrics_file_rank0.exists(), "Stale SFT metrics rank 0 should be removed"
        assert not stale_metrics_file_rank1.exists(), "Stale SFT metrics rank 1 should be removed"
        assert not stale_metrics_file_rank2.exists(), "Stale SFT metrics rank 2 should be removed"
    finally:
        # Always shut down server to avoid port conflicts
        if server:
            server.shutdown()

    print("test execution complete")


def test_instrumentation_cleans_osft_metrics_on_startup(tmp_path):
    """Test that OSFT metrics files are cleaned before training starts."""
    print("Executing test: OSFT metrics cleanup on startup")

    # Create stale metrics files for multiple ranks (distributed training scenario)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    stale_metrics_file_rank0 = ckpt_dir / "training_metrics_0.jsonl"
    stale_metrics_file_rank0.write_text('{"step": 50, "loss": 1.2}\n')

    stale_metrics_file_rank1 = ckpt_dir / "training_metrics_1.jsonl"
    stale_metrics_file_rank1.write_text('{"step": 50, "loss": 1.1}\n')

    stale_metrics_file_rank2 = ckpt_dir / "training_metrics_2.jsonl"
    stale_metrics_file_rank2.write_text('{"step": 50, "loss": 1.0}\n')

    stale_config_file = ckpt_dir / "training_params.json"
    stale_config_file.write_text('{"max_epochs": 10}\n')

    # Call instrumentation directly (no exec) with port 0 for random available port
    from kubeflow.trainer.rhai.traininghub import _create_training_hub_progression_instrumentation

    apply_fn, _handler = _create_training_hub_progression_instrumentation(
        algorithm="osft",
        ckpt_output_dir=str(ckpt_dir),
        metrics_port=0,  # Use port 0 for random available port
    )

    # Call the function to trigger cleanup and start server
    server = apply_fn()

    try:
        # Verify all stale metrics from all ranks were removed
        assert not stale_metrics_file_rank0.exists(), "Stale OSFT metrics rank 0 should be removed"
        assert not stale_metrics_file_rank1.exists(), "Stale OSFT metrics rank 1 should be removed"
        assert not stale_metrics_file_rank2.exists(), "Stale OSFT metrics rank 2 should be removed"
        # Config file should NOT be removed (backend overwrites it)
        assert stale_config_file.exists(), "OSFT config should not be removed"
    finally:
        # Always shut down server to avoid port conflicts
        if server:
            server.shutdown()

    print("test execution complete")


def test_instrumentation_cleanup_handles_missing_files(tmp_path):
    """Test that cleanup handles missing files gracefully."""
    print("Executing test: Cleanup handles missing files")

    # Create empty directory (no files to clean)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Call instrumentation directly (no exec) with port 0 for random available port
    from kubeflow.trainer.rhai.traininghub import _create_training_hub_progression_instrumentation

    apply_fn, _handler = _create_training_hub_progression_instrumentation(
        algorithm="sft",
        ckpt_output_dir=str(ckpt_dir),
        metrics_port=0,  # Use port 0 for random available port
    )

    # Should not raise exception when no files exist
    server = apply_fn()

    # Always shut down server
    if server:
        server.shutdown()

    print("test execution complete")


def test_instrumentation_cleanup_continues_on_error():
    """Test that training continues even if cleanup fails."""
    print("Executing test: Training continues on cleanup failure")

    # Get instrumentation wrapper with non-existent directory
    wrapper = get_training_hub_instrumentation_wrapper(
        algorithm="sft",
        ckpt_output_dir="/nonexistent/path",
        metrics_port=28080,
    )

    # Verify error handling is present
    assert "Warning: Metrics cleanup failed" in wrapper, "Should log cleanup failures"

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
