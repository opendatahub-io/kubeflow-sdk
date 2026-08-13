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

"""Tests for Transformers instrumentation functions."""

import pytest

from kubeflow.trainer.rhai.transformers import (
    get_jit_checkpoint_injection_code,
    get_transformers_instrumentation_wrapper,
)
from kubeflow.trainer.test.common import SUCCESS, TestCase


class TestGetTransformersInstrumentationWrapper:
    """Tests for progression tracking wrapper generation."""

    def test_wrapper_contains_placeholder(self) -> None:
        """Generated wrapper must contain the user code placeholder."""
        print("Executing test: wrapper_contains_placeholder")
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        assert "{{user_func_import_and_call}}" in wrapper

    def test_wrapper_contains_metrics_port(self) -> None:
        """Generated wrapper must embed the provided metrics port."""
        print("Executing test: wrapper_contains_metrics_port")
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=9090)
        assert "metrics_port=9090" in wrapper

    def test_wrapper_contains_instrumentation_function(self) -> None:
        """Generated wrapper must contain the progression instrumentation function."""
        print("Executing test: wrapper_contains_instrumentation_function")
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        assert "_create_progression_instrumentation" in wrapper

    def test_wrapper_contains_apply_call(self) -> None:
        """Generated wrapper must call apply_progression_tracking."""
        print("Executing test: wrapper_contains_apply_call")
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        assert "apply_progression_tracking()" in wrapper

    def test_wrapper_is_valid_python_structure(self) -> None:
        """Generated wrapper should have header and footer markers."""
        print("Executing test: wrapper_is_valid_python_structure")
        wrapper = get_transformers_instrumentation_wrapper(metrics_port=28080)
        assert "Kubeflow SDK" in wrapper
        assert "USER TRAINING CODE" in wrapper


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="checkpoint code with JIT enabled contains checkpoint function",
            expected_status=SUCCESS,
            config={
                "output_dir": "/mnt/checkpoints",
                "enable_jit_checkpoint": True,
            },
            expected_output="_create_checkpoint_instrumentation",
        ),
        TestCase(
            name="checkpoint code embeds output_dir",
            expected_status=SUCCESS,
            config={
                "output_dir": "/mnt/my-checkpoints",
                "enable_jit_checkpoint": True,
            },
            expected_output="/mnt/my-checkpoints",
        ),
        TestCase(
            name="checkpoint code with S3 URI embeds cloud storage URI",
            expected_status=SUCCESS,
            config={
                "output_dir": "/mnt/local",
                "cloud_remote_storage_uri": "s3://my-bucket/models",
                "enable_jit_checkpoint": True,
            },
            expected_output="s3://my-bucket/models",
        ),
        TestCase(
            name="checkpoint code with periodic config embeds save_strategy",
            expected_status=SUCCESS,
            config={
                "output_dir": "/mnt/checkpoints",
                "enable_jit_checkpoint": True,
                "periodic_checkpoint_config": {
                    "save_strategy": "steps",
                    "save_steps": 100,
                    "save_total_limit": 5,
                },
            },
            expected_output="save_strategy",
        ),
    ],
)
def test_jit_checkpoint_injection_code_content(test_case: TestCase) -> None:
    """Verify JIT checkpoint injection code contains expected elements."""
    print(f"Executing test: {test_case.name}")
    header, footer = get_jit_checkpoint_injection_code(**test_case.config)
    assert test_case.expected_output in header


def test_jit_checkpoint_injection_without_jit_still_generates_code() -> None:
    """Even with enable_jit_checkpoint=False, instrumentation code is generated."""
    print("Executing test: jit_checkpoint_injection_without_jit_still_generates_code")
    header, footer = get_jit_checkpoint_injection_code(enable_jit_checkpoint=False)
    assert "_create_checkpoint_instrumentation" in header
    assert "enable_jit" in header


def test_jit_checkpoint_injection_with_jit_sets_enable_jit_true() -> None:
    """With enable_jit_checkpoint=True, config dict must set enable_jit to True."""
    print("Executing test: jit_checkpoint_injection_with_jit_sets_enable_jit_true")
    header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/checkpoints",
        enable_jit_checkpoint=True,
    )
    assert "'enable_jit': True" in header


def test_jit_checkpoint_footer_contains_upload() -> None:
    """Footer must contain the final model upload call when JIT is enabled."""
    print("Executing test: jit_checkpoint_footer_contains_upload")
    header, footer = get_jit_checkpoint_injection_code(
        output_dir="/mnt/checkpoints",
        enable_jit_checkpoint=True,
    )
    assert "upload_final_model_to_cloud" in footer


def test_jit_checkpoint_header_contains_constants() -> None:
    """Header must inline constants and exclude the SDK constants import."""
    print("Executing test: jit_checkpoint_header_contains_constants")
    header, _ = get_jit_checkpoint_injection_code(
        output_dir="/mnt/checkpoints",
        enable_jit_checkpoint=True,
    )
    assert "CHECKPOINT_INCOMPLETE_MARKER" in header
    assert "CHECKPOINT_STAGING_DIR" in header
    assert "kubeflow.trainer.rhai.constants" not in header
    assert "# Constants defined globally above" in header
