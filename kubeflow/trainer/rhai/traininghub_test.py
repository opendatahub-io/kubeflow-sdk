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

"""Tests for TrainingHubTrainer and instrumentation wrapper generation."""

import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai import constants as rhai_constants
from kubeflow.trainer.rhai.traininghub import (
    TrainingHubAlgorithms,
    TrainingHubTrainer,
    get_progress_tracking_annotations,
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
            name="builds CRD for SFT algorithm wrapper with explicit resources",
            expected_status=SUCCESS,
            config={
                "runtime": create_runtime_type(name="torch"),
                "trainer": TrainingHubTrainer(
                    func=None,
                    func_args={"nnodes": 2, "nproc_per_node": 2, "data_path": "/data/file.json"},
                    packages_to_install=["training_hub"],
                    algorithm=TrainingHubAlgorithms.SFT,
                    resources_per_node={"gpu": "2"},
                ),
            },
            expected_output=models.TrainerV1alpha1Trainer(
                numNodes=2,
                numProcPerNode=models.IoK8sApimachineryPkgUtilIntstrIntOrString(2),
                resourcesPerNode=get_expected_resources_per_node(2),
                command=["bash", "-c"],
            ),
        ),
        TestCase(
            name="embeds user function and maps env",
            expected_status=SUCCESS,
            config={
                "runtime": create_runtime_type(name="torch"),
                "trainer": TrainingHubTrainer(
                    func=_simple_training_fn,
                    func_args={"param": 1},
                    packages_to_install=["some_pkg"],
                    algorithm=None,
                    env={"A": "1", "B": "two"},
                ),
            },
            expected_output=models.TrainerV1alpha1Trainer(
                command=["bash", "-c"],
                env=[
                    models.IoK8sApiCoreV1EnvVar(name="A", value="1"),
                    models.IoK8sApiCoreV1EnvVar(name="B", value="two"),
                ],
            ),
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

        assert test_case.expected_status == SUCCESS
        assert isinstance(crd, models.TrainerV1alpha1Trainer)

        # Validate baseline fields if provided in expected_output
        exp: Optional[models.TrainerV1alpha1Trainer] = test_case.expected_output
        if exp is not None:
            if exp.num_nodes is not None:
                assert crd.num_nodes == exp.num_nodes
            if exp.resources_per_node is not None:
                # Compare gpu quantities
                assert crd.resources_per_node is not None
                assert crd.resources_per_node.limits is not None
                assert (
                    crd.resources_per_node.limits.get("nvidia.com/gpu").actual_instance
                    == exp.resources_per_node.limits.get("nvidia.com/gpu").actual_instance  # type: ignore
                )
            if exp.num_proc_per_node is not None:
                assert crd.num_proc_per_node is not None
                assert (
                    crd.num_proc_per_node.actual_instance == exp.num_proc_per_node.actual_instance
                )
            if exp.command is not None:
                assert crd.command == exp.command

        # Specific content checks
        if trainer.func is None:
            # Algorithm wrapper path; ensure algorithm import appears
            assert crd.args is not None and len(crd.args) == 1
            script = crd.args[0]
            algo_name = trainer.algorithm.value if trainer.algorithm else ""
            assert f"from training_hub import {algo_name}" in script
            assert "training_script.py" in script
        else:
            # User function path; ensure function name appears in generated script
            assert crd.args is not None and len(crd.args) == 1
            script = crd.args[0]
            assert "_simple_training_fn" in script

        # Env mapping check (if env was provided)
        if trainer.env:
            # Compare (name, value) pairs for stability
            actual_env = sorted([(e.name, e.value) for e in crd.env])  # type: ignore
            expected_env = sorted([(e.name, e.value) for e in test_case.expected_output.env])  # type: ignore
            assert actual_env == expected_env

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
        crd = get_trainer_cr_from_training_hub_trainer(runtime, trainer_cfg)

        assert test_case.expected_status == SUCCESS
        # Validate topology mapping: when nnodes / nproc_per_node are provided in func_args,
        # they should be propagated; otherwise they should be left unset so that the
        # TrainingRuntime ML policy can supply defaults.
        nnodes_expected = trainer_cfg.func_args.get("nnodes") if trainer_cfg.func_args else None
        nproc_expected = (
            trainer_cfg.func_args.get("nproc_per_node") if trainer_cfg.func_args else None
        )

        if nnodes_expected is not None:
            assert crd.num_nodes == nnodes_expected
        else:
            assert crd.num_nodes is None

        if nproc_expected is not None:
            assert crd.num_proc_per_node is not None
            assert crd.num_proc_per_node.actual_instance == nproc_expected
        else:
            assert crd.num_proc_per_node is None

        # Validate command/args structure.
        assert crd.command == ["bash", "-c"]
        assert isinstance(crd.args, list) and len(crd.args) == 1 and isinstance(crd.args[0], str)
        args_str = crd.args[0]

        # Algorithm wrapper should import training_hub.<algo>
        if trainer_cfg.algorithm:
            assert f"from training_hub import {trainer_cfg.algorithm.value}" in args_str
            # PIP install header if packages requested.
            if trainer_cfg.packages_to_install:
                assert "pip install" in args_str

        # Validate env mapping.
        if trainer_cfg.env:
            assert crd.env is not None
            env_dict = {e.name: e.value for e in crd.env}
            assert env_dict == trainer_cfg.env

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
        # Topology parameters are not set when not provided in func_args; ML policy will
        # supply defaults instead of the SDK.
        assert crd.num_nodes is None
        assert crd.num_proc_per_node is None
        # No explicit resources_per_node were provided, so resources should be unset.
        assert crd.resources_per_node is None

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


def test_get_progress_tracking_annotations_enabled():
    """Test that annotations are generated when progression tracking is enabled."""
    print("Executing test: get_progress_tracking_annotations with enabled=True")

    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.SFT,
        func_args={"ckpt_output_dir": "/tmp"},
        enable_progression_tracking=True,
        metrics_port=28080,
        metrics_poll_interval_seconds=30,
    )

    annotations = get_progress_tracking_annotations(trainer)

    # Verify all expected annotations are present
    assert rhai_constants.ANNOTATION_PROGRESSION_TRACKING in annotations
    assert annotations[rhai_constants.ANNOTATION_PROGRESSION_TRACKING] == "true"

    assert rhai_constants.ANNOTATION_METRICS_PORT in annotations
    assert annotations[rhai_constants.ANNOTATION_METRICS_PORT] == "28080"

    assert rhai_constants.ANNOTATION_METRICS_POLL_INTERVAL in annotations
    assert annotations[rhai_constants.ANNOTATION_METRICS_POLL_INTERVAL] == "30s"

    assert rhai_constants.ANNOTATION_FRAMEWORK in annotations
    assert annotations[rhai_constants.ANNOTATION_FRAMEWORK] == "traininghub"

    print("test execution complete")


def test_get_progress_tracking_annotations_disabled():
    """Test that empty dict is returned when progression tracking is disabled."""
    print("Executing test: get_progress_tracking_annotations with enabled=False")

    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.OSFT,
        func_args={"ckpt_output_dir": "/tmp"},
        enable_progression_tracking=False,
    )

    annotations = get_progress_tracking_annotations(trainer)

    # Should return empty dict when disabled
    assert annotations == {}

    print("test execution complete")


def test_get_progress_tracking_annotations_custom_values():
    """Test that custom port and interval values are correctly set in annotations."""
    print("Executing test: get_progress_tracking_annotations with custom values")

    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.OSFT,
        func_args={"ckpt_output_dir": "/tmp"},
        enable_progression_tracking=True,
        metrics_port=9999,
        metrics_poll_interval_seconds=120,
    )

    annotations = get_progress_tracking_annotations(trainer)

    assert annotations[rhai_constants.ANNOTATION_METRICS_PORT] == "9999"
    assert annotations[rhai_constants.ANNOTATION_METRICS_POLL_INTERVAL] == "120s"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
