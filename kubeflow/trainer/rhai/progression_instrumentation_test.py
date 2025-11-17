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

"""Tests for progression tracking instrumentation."""

import time
from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.rhai.progression_instrumentation import (
    ProgressCallback,
    _metrics_state,
    _update_metrics,
)


def test_server_starts_only_on_rank_zero():
    """Test server only starts on rank-0 process in distributed training."""
    print("Executing test: Server starts only on rank-0")

    # Test rank-0: should start server
    callback_rank0 = ProgressCallback(custom_metrics_map={}, metrics_port=28080)
    state_rank0 = Mock(
        is_world_process_zero=True,
        global_step=0,
        max_steps=100,
        epoch=0,
    )
    args_rank0 = Mock(num_train_epochs=3)

    with patch("kubeflow.trainer.rhai.progression_instrumentation.start_server") as mock_start:
        callback_rank0.on_train_begin(args_rank0, state_rank0, Mock())
        assert mock_start.called, "Server should start on rank-0"
        mock_start.assert_called_once_with(28080)

    # Test rank-1: should NOT start server
    callback_rank1 = ProgressCallback(custom_metrics_map={}, metrics_port=28080)
    state_rank1 = Mock(
        is_world_process_zero=False,
        global_step=0,
        max_steps=100,
        epoch=0,
    )
    args_rank1 = Mock(num_train_epochs=3)

    with patch("kubeflow.trainer.rhai.progression_instrumentation.start_server") as mock_start:
        callback_rank1.on_train_begin(args_rank1, state_rank1, Mock())
        assert not mock_start.called, "Server should NOT start on rank-1"

    print("test execution complete")


def test_custom_metrics_mapping():
    """Test custom metrics are correctly mapped and categorized."""
    print("Executing test: Custom metrics mapping")

    callback = ProgressCallback(
        custom_metrics_map={
            "perplexity": "perplexity",  # train metric
            "custom_score": "eval_custom_score",  # eval metric (by prefix)
            "bleu": "eval_bleu",  # eval metric
        }
    )

    args = Mock()
    state = Mock()
    control = Mock()

    logs = {
        "perplexity": 15.2,
        "custom_score": 0.89,
        "bleu": 0.45,
        "unmapped_metric": 123.4,  # Not in custom_metrics_map, should be ignored
    }

    # Clear metrics before test
    _metrics_state["trainMetrics"] = {}
    _metrics_state["evalMetrics"] = {}

    callback.on_log(args, state, control, logs=logs)

    assert _metrics_state["trainMetrics"]["perplexity"] == 15.2
    assert _metrics_state["evalMetrics"]["eval_custom_score"] == 0.89
    assert _metrics_state["evalMetrics"]["eval_bleu"] == 0.45
    assert "unmapped_metric" not in _metrics_state["trainMetrics"]
    assert "unmapped_metric" not in _metrics_state["evalMetrics"]

    print("test execution complete")


def test_progress_percentage_calculation():
    """Test progress percentage is calculated correctly."""
    print("Executing test: Progress percentage calculation")

    callback = ProgressCallback(custom_metrics_map={})
    callback.start_time = time.time()

    state = Mock(
        is_world_process_zero=True,
        global_step=50,
        max_steps=200,
        epoch=1,
    )
    args = Mock(num_train_epochs=3)

    # Clear metrics
    _metrics_state["progressPercentage"] = None
    _metrics_state["currentStep"] = 0

    callback.on_step_end(args, state, Mock())

    assert _metrics_state["progressPercentage"] == 25  # 50/200 = 25%
    assert _metrics_state["currentStep"] == 50
    assert _metrics_state["totalSteps"] == 200

    print("test execution complete")


def test_progress_callback_handles_numeric_types():
    """Test ProgressCallback handles different numeric types (int, float)."""
    print("Executing test: Numeric types handling")

    callback = ProgressCallback(custom_metrics_map={})

    args = Mock()
    state = Mock()
    control = Mock()

    # Test basic numeric types
    logs = {
        "loss": 0.5,  # float
        "learning_rate": 1e-4,  # float in scientific notation
        "eval_accuracy": 0.95,  # float
        "step": 100,  # int
        "message": "training",  # string - should be skipped
        "config": {"lr": 0.01},  # dict - should be skipped
    }

    # Clear metrics
    _metrics_state["trainMetrics"] = {}
    _metrics_state["evalMetrics"] = {}

    callback.on_log(args, state, control, logs=logs)

    assert _metrics_state["trainMetrics"]["loss"] == 0.5
    assert _metrics_state["trainMetrics"]["learning_rate"] == 1e-4
    assert _metrics_state["evalMetrics"]["eval_accuracy"] == 0.95
    # Non-numeric values should not be present
    assert "message" not in _metrics_state["trainMetrics"]
    assert "config" not in _metrics_state["trainMetrics"]

    print("test execution complete")


def test_progress_callback_handles_torch_tensors():
    """Test tensor handling (skipped if torch not installed)."""
    print("Executing test: PyTorch tensor handling")

    torch = pytest.importorskip("torch")  # Skip if torch not available

    callback = ProgressCallback(custom_metrics_map={})

    args = Mock()
    state = Mock()
    control = Mock()

    # Clear metrics
    _metrics_state["trainMetrics"] = {}
    _metrics_state["evalMetrics"] = {}

    logs = {
        "loss": torch.tensor(0.42),  # Scalar tensor
        "learning_rate": torch.tensor(1e-5),  # Scalar tensor
        "eval_f1": torch.tensor(0.88),  # Scalar tensor
        "multi": torch.tensor([1, 2, 3]),  # Multi-element (should skip)
        "matrix": torch.tensor([[1, 2], [3, 4]]),  # 2D tensor (should skip)
    }

    callback.on_log(args, state, control, logs=logs)

    assert _metrics_state["trainMetrics"]["loss"] == pytest.approx(0.42)
    assert _metrics_state["trainMetrics"]["learning_rate"] == pytest.approx(1e-5)
    assert _metrics_state["evalMetrics"]["eval_f1"] == pytest.approx(0.88)
    # Multi-element tensors should not be present
    assert "multi" not in _metrics_state["trainMetrics"]
    assert "matrix" not in _metrics_state["trainMetrics"]

    print("test execution complete")


def test_time_estimation():
    """Test remaining time estimation."""
    print("Executing test: Time estimation")

    callback = ProgressCallback(custom_metrics_map={})

    # Simulate training start
    callback.start_time = time.time() - 10  # Started 10 seconds ago

    state = Mock(
        is_world_process_zero=True,
        global_step=25,  # 25% complete
        max_steps=100,
        epoch=0,
    )
    args = Mock(num_train_epochs=1)

    # Clear metrics
    _metrics_state["estimatedRemainingSeconds"] = None

    callback.on_step_end(args, state, Mock())

    # If 25% took 10s, 100% should take ~40s, so ~30s remaining
    assert _metrics_state["estimatedRemainingSeconds"] is not None
    assert 25 <= _metrics_state["estimatedRemainingSeconds"] <= 35  # Allow some variance

    print("test execution complete")


def test_metrics_update_thread_safe():
    """Test concurrent metric updates don't corrupt state."""
    print("Executing test: Thread-safe metric updates")

    import threading

    # Clear metrics
    _metrics_state["trainMetrics"] = {}

    def update_metrics_thread(thread_id):
        for i in range(100):
            _update_metrics({"trainMetrics": {f"metric_{thread_id}_{i}": i}})

    threads = []
    for tid in range(5):
        thread = threading.Thread(target=update_metrics_thread, args=(tid,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Verify all metrics were recorded
    # Should have 5 threads * 100 metrics = 500 metrics
    assert len(_metrics_state["trainMetrics"]) == 500

    print("test execution complete")


def test_standard_metrics_auto_tracked():
    """Test that standard metrics are automatically tracked."""
    print("Executing test: Standard metrics auto-tracking")

    callback = ProgressCallback(custom_metrics_map={})

    args = Mock()
    state = Mock()
    control = Mock()

    # Clear metrics
    _metrics_state["trainMetrics"] = {}
    _metrics_state["evalMetrics"] = {}

    logs = {
        "loss": 0.5,
        "learning_rate": 1e-4,
        "grad_norm": 2.3,
        "train_loss": 0.48,
        "train_samples_per_second": 4.2,
        "eval_loss": 0.52,
        "eval_accuracy": 0.95,
    }

    callback.on_log(args, state, control, logs=logs)

    # Standard train metrics
    assert _metrics_state["trainMetrics"]["loss"] == 0.5
    assert _metrics_state["trainMetrics"]["learning_rate"] == 1e-4
    assert _metrics_state["trainMetrics"]["grad_norm"] == 2.3
    assert _metrics_state["trainMetrics"]["train_loss"] == 0.48
    assert _metrics_state["trainMetrics"]["throughput_samples_sec"] == 4.2
    # Eval metrics (auto-detected by prefix)
    assert _metrics_state["evalMetrics"]["eval_loss"] == 0.52
    assert _metrics_state["evalMetrics"]["eval_accuracy"] == 0.95

    print("test execution complete")


def test_on_train_end_sets_completion():
    """Test that on_train_end marks training as 100% complete."""
    print("Executing test: Training completion")

    callback = ProgressCallback(custom_metrics_map={})

    # Clear metrics
    _metrics_state["progressPercentage"] = 50
    _metrics_state["estimatedRemainingSeconds"] = 100

    callback.on_train_end(Mock(), Mock(), Mock())

    assert _metrics_state["progressPercentage"] == 100
    assert _metrics_state["estimatedRemainingSeconds"] == 0

    print("test execution complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
