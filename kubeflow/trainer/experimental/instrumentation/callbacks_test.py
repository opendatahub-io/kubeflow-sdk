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

"""Unit tests for instrumentation callback generators."""

import pytest

from kubeflow.trainer.experimental.instrumentation.callbacks import (
    _validate_metric_name,
    get_transformers_trainer_wrapper_script,
)


def test_transformers_wrapper_compiles():
    """Generated Transformers wrapper script should be valid Python."""
    script = get_transformers_trainer_wrapper_script()
    compile(script, "<string>", "exec")

    # Verify key components are present
    assert "KubeflowProgressCallback" in script
    assert "start_metrics_server" in script
    assert "{user_func_import_and_call}" in script


def test_transformers_wrapper_custom_port():
    """Custom metrics port should be configurable."""
    script = get_transformers_trainer_wrapper_script(metrics_port=9090)
    assert "port=9090" in script
    compile(script, "<string>", "exec")


def test_transformers_wrapper_custom_metrics():
    """Custom metrics should be injected and tracked."""
    script = get_transformers_trainer_wrapper_script(
        custom_metrics={"eval_accuracy": "accuracy", "eval_f1": "f1_score"}
    )

    # Should include custom metrics mapping
    assert "eval_accuracy" in script
    assert "eval_f1" in script
    assert "accuracy" in script
    assert "f1_score" in script
    compile(script, "<string>", "exec")

    # Edge case: None/empty should not break
    compile(get_transformers_trainer_wrapper_script(custom_metrics=None), "<string>", "exec")
    compile(get_transformers_trainer_wrapper_script(custom_metrics={}), "<string>", "exec")


def test_validate_metric_name_valid():
    """Valid metric names should pass validation."""
    # Valid identifiers
    _validate_metric_name("accuracy")
    _validate_metric_name("eval_loss")
    _validate_metric_name("f1_score")
    _validate_metric_name("_private_metric")
    _validate_metric_name("metric123")
    _validate_metric_name("UPPERCASE_METRIC")


def test_validate_metric_name_invalid_empty():
    """Empty or whitespace-only metric names should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_metric_name("")

    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_metric_name("   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_metric_name("\t\n")


def test_validate_metric_name_invalid_quotes():
    """Metric names with quotes should be rejected (code injection risk)."""
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_metric_name('metric"name')

    with pytest.raises(ValueError, match="invalid characters"):
        _validate_metric_name("metric'name")

    with pytest.raises(ValueError, match="invalid characters"):
        _validate_metric_name('"; DROP TABLE metrics; --')


def test_validate_metric_name_invalid_backslash():
    """Metric names with backslashes should be rejected."""
    with pytest.raises(ValueError, match="invalid characters"):
        _validate_metric_name("metric\\name")


def test_validate_metric_name_invalid_identifier():
    """Metric names that aren't valid Python identifiers should be rejected."""
    with pytest.raises(ValueError, match="valid Python identifier"):
        _validate_metric_name("metric-name")  # Hyphen not allowed

    with pytest.raises(ValueError, match="valid Python identifier"):
        _validate_metric_name("metric name")  # Space not allowed

    with pytest.raises(ValueError, match="valid Python identifier"):
        _validate_metric_name("123metric")  # Can't start with number

    with pytest.raises(ValueError, match="valid Python identifier"):
        _validate_metric_name("metric@name")  # Special char not allowed

    with pytest.raises(ValueError, match="valid Python identifier"):
        _validate_metric_name("metric.name")  # Dot not allowed


def test_transformers_wrapper_rejects_invalid_metrics():
    """Wrapper script generation should reject invalid custom metric names."""
    with pytest.raises(ValueError, match="invalid characters"):
        get_transformers_trainer_wrapper_script(
            custom_metrics={"eval_loss": "loss'; malicious_code()"}
        )

    with pytest.raises(ValueError, match="valid Python identifier"):
        get_transformers_trainer_wrapper_script(custom_metrics={"eval": "metric-name"})

    with pytest.raises(ValueError, match="cannot be empty"):
        get_transformers_trainer_wrapper_script(custom_metrics={"eval": "   "})
