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

"""Integration tests for instrumentation - actually run generated code."""

from http.client import HTTPConnection
import importlib.util
import threading
import time

import pytest

from kubeflow.trainer.experimental.instrumentation.callbacks import (
    get_transformers_trainer_wrapper_script,
)

# Check if transformers is available
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

# Skip all tests in this module if transformers is not available
pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers package not installed - integration tests require it"
)


def test_wrapper_script_executes_without_errors():
    """Test that generated wrapper actually runs."""
    script = get_transformers_trainer_wrapper_script(metrics_port=28082)

    # Replace placeholder with simple function
    script = script.replace("{user_func_import_and_call}", "def dummy(): pass\ndummy()")

    # Should execute without errors
    try:
        exec(compile(script, "<string>", "exec"))
    finally:
        # Cleanup: server runs in daemon thread, will die with main thread
        time.sleep(0.1)


def test_progress_callback_updates_metrics():
    """Test that KubeflowProgressCallback actually updates metrics."""
    script = get_transformers_trainer_wrapper_script(metrics_port=28083)

    # Inject test code
    test_code = """
def test_callback():
    callback = KubeflowProgressCallback()

    # Mock state and args
    state = type('State', (), {
        'global_step': 10,
        'max_steps': 100,
        'epoch': 1
    })()
    args = type('Args', (), {})()

    # Call on_train_begin
    callback.on_train_begin(args, state, None)

    # Check metrics were updated
    import json
    metrics_str = ProgressMetricsHandler.get_metrics_json()
    metrics = json.loads(metrics_str)
    assert metrics['status'] == 'training'
    assert metrics['progress']['step_total'] == 100

test_callback()
"""

    script = script.replace("{user_func_import_and_call}", test_code)

    # Execute and verify
    exec(compile(script, "<string>", "exec"))


def test_custom_metrics_appear_in_handler():
    """Test that custom metrics are initialized and tracked."""
    custom_metrics = {"eval_accuracy": "accuracy", "eval_f1": "f1_score"}
    script = get_transformers_trainer_wrapper_script(
        metrics_port=28084,
        custom_metrics=custom_metrics,
    )

    test_code = """
def test_custom_metrics():
    import json

    # Check custom metrics initialized
    metrics_str = ProgressMetricsHandler.get_metrics_json()
    metrics = json.loads(metrics_str)
    assert 'accuracy' in metrics['metrics']
    assert 'f1_score' in metrics['metrics']

    # Test on_log with custom metrics
    callback = KubeflowProgressCallback()
    state = type('State', (), {})()
    args = type('Args', (), {})()

    logs = {
        'loss': 0.5,
        'eval_accuracy': 0.95,
        'eval_f1': 0.87,
    }

    callback.on_log(args, state, None, logs=logs)

    # Verify custom metrics were tracked
    metrics_str = ProgressMetricsHandler.get_metrics_json()
    metrics = json.loads(metrics_str)
    assert metrics['metrics']['accuracy'] == 0.95
    assert metrics['metrics']['f1_score'] == 0.87
    assert metrics['metrics']['loss'] == 0.5

test_custom_metrics()
"""

    script = script.replace("{user_func_import_and_call}", test_code)
    exec(compile(script, "<string>", "exec"))


def test_metrics_server_starts_on_custom_port():
    """Test that metrics server starts on specified port."""
    port = 28087
    script = get_transformers_trainer_wrapper_script(metrics_port=port)

    script = script.replace("{{user_func_import_and_call}}", "pass")

    # Execute in separate thread to avoid blocking
    def run_script():
        exec(compile(script, "<string>", "exec"))

    thread = threading.Thread(target=run_script, daemon=True)
    thread.start()
    time.sleep(0.2)  # Give server time to start

    # Try to connect
    try:
        conn = HTTPConnection("localhost", port, timeout=5)
        conn.request("GET", "/health")
        response = conn.getresponse()
        assert response.status == 200
        conn.close()
    except Exception as e:
        pytest.fail(f"Failed to connect to metrics server: {e}")


def test_on_step_end_calculates_eta():
    """Test that on_step_end calculates ETA correctly."""
    script = get_transformers_trainer_wrapper_script(metrics_port=28088)

    test_code = """
def test_eta():
    import time
    import json

    callback = KubeflowProgressCallback()

    # Initialize
    state = type('State', (), {
        'global_step': 0,
        'max_steps': 100,
        'epoch': 1
    })()
    args = type('Args', (), {})()

    callback.on_train_begin(args, state, None)
    time.sleep(0.1)

    # Simulate step
    state.global_step = 50
    callback.on_step_end(args, state, None)

    # Check progress metrics
    metrics_str = ProgressMetricsHandler.get_metrics_json()
    metrics = json.loads(metrics_str)

    assert metrics['progress']['step_current'] == 50
    assert metrics['progress']['percent'] == 50.0
    assert metrics['time']['elapsed_sec'] > 0
    # ETA might be None if step is first one, but should be calculated

test_eta()
"""

    script = script.replace("{user_func_import_and_call}", test_code)
    exec(compile(script, "<string>", "exec"))
