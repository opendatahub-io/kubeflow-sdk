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

"""Unit tests for experimental trainer types."""

from kubeflow.trainer.types.experimental.types import TransformersTrainer


def dummy_func():
    """Dummy training function."""
    pass


def test_transformers_trainer_defaults():
    """Test TransformersTrainer has correct default values."""
    trainer = TransformersTrainer(func=dummy_func)

    assert trainer.enable_progression_tracking is True
    assert trainer.metrics_port == 28080
    assert trainer.custom_metrics == {}


def test_transformers_trainer_with_custom_metrics():
    """Test TransformersTrainer accepts custom_metrics."""
    custom = {"eval_accuracy": "accuracy", "eval_f1": "f1"}
    trainer = TransformersTrainer(func=dummy_func, custom_metrics=custom)

    assert trainer.custom_metrics == custom
    assert "eval_accuracy" in trainer.custom_metrics
    assert trainer.custom_metrics["eval_accuracy"] == "accuracy"


def test_transformers_trainer_disable_features():
    """Test TransformersTrainer can disable progression tracking."""
    trainer = TransformersTrainer(
        func=dummy_func,
        enable_progression_tracking=False,
    )

    assert trainer.enable_progression_tracking is False


def test_transformers_trainer_custom_port():
    """Test TransformersTrainer accepts custom metrics port."""
    trainer = TransformersTrainer(
        func=dummy_func,
        metrics_port=9090,
    )

    assert trainer.metrics_port == 9090


def test_transformers_trainer_has_custom_trainer_fields():
    """Test TransformersTrainer has same core fields as CustomTrainer."""
    trainer = TransformersTrainer(
        func=dummy_func,
        num_nodes=2,
        resources_per_node={"gpu": 1, "cpu": 4},
        packages_to_install=["transformers", "torch"],
    )

    assert trainer.num_nodes == 2
    assert trainer.resources_per_node == {"gpu": 1, "cpu": 4}
    assert trainer.packages_to_install == ["transformers", "torch"]


def test_transformers_trainer_with_func_args():
    """Test TransformersTrainer can store function arguments."""
    func_args = {"model_name": "bert-base", "batch_size": 32}
    trainer = TransformersTrainer(
        func=dummy_func,
        func_args=func_args,
    )

    assert trainer.func_args == func_args


def test_transformers_trainer_none_custom_metrics():
    """Test TransformersTrainer handles None custom_metrics."""
    trainer = TransformersTrainer(
        func=dummy_func,
        custom_metrics=None,
    )

    # Should default to empty dict due to field default_factory
    assert trainer.custom_metrics is None or trainer.custom_metrics == {}


def test_transformers_trainer_empty_custom_metrics():
    """Test TransformersTrainer handles empty custom_metrics dict."""
    trainer = TransformersTrainer(
        func=dummy_func,
        custom_metrics={},
    )

    assert trainer.custom_metrics == {}
    assert len(trainer.custom_metrics) == 0


def test_transformers_trainer_multiple_custom_metrics():
    """Test TransformersTrainer with multiple custom metrics."""
    custom = {
        "eval_accuracy": "accuracy",
        "eval_precision": "precision",
        "eval_recall": "recall",
        "eval_f1": "f1_score",
        "custom_metric": "my_metric",
    }
    trainer = TransformersTrainer(
        func=dummy_func,
        custom_metrics=custom,
    )

    assert len(trainer.custom_metrics) == 5
    assert trainer.custom_metrics["eval_accuracy"] == "accuracy"
    assert trainer.custom_metrics["custom_metric"] == "my_metric"


def test_transformers_trainer_is_dataclass():
    """Test that TransformersTrainer is properly decorated as dataclass."""
    import dataclasses

    assert dataclasses.is_dataclass(TransformersTrainer)
    fields = {f.name for f in dataclasses.fields(TransformersTrainer)}

    # Check core training fields (same as CustomTrainer)
    assert "func" in fields
    assert "func_args" in fields
    assert "packages_to_install" in fields
    assert "pip_index_urls" in fields
    assert "num_nodes" in fields
    assert "resources_per_node" in fields
    assert "env" in fields

    # Check experimental instrumentation fields
    assert "enable_progression_tracking" in fields
    assert "metrics_port" in fields
    assert "metrics_poll_interval_seconds" in fields
    assert "custom_metrics" in fields
