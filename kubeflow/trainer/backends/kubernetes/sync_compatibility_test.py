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

"""Regression proof for the upstream/downstream integration boundary."""

from collections.abc import Iterator
from unittest.mock import Mock, patch

from kubeflow_trainer_api import models
from kubernetes import client
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.options import ActiveDeadlineSeconds, Annotations, Labels, Name
from kubeflow.trainer.rhai import (
    RHAITrainer,
    SpeculativeDecodingTrainer,
    SpeculatorConfig,
    SpeculatorMode,
    TrainingHubAlgorithms,
    TrainingHubTrainer,
    TransformersTrainer,
    utils as rhai_utils,
)
from kubeflow.trainer.rhai.constants import (
    ANNOTATION_METRICS_PORT,
    ANNOTATION_PROGRESSION_TRACKING,
    CHECKPOINT_MOUNT_PATH,
    CHECKPOINT_VOLUME_NAME,
)
from kubeflow.trainer.types import types


def sync_train_function() -> None:
    """Inspectable training function for the sync regression tests."""
    pass


@pytest.fixture
def sync_backend() -> Iterator[KubernetesBackend]:
    """Use mocked APIs so these integration checks never contact a cluster."""
    with (
        patch("kubernetes.client.CustomObjectsApi"),
        patch("kubernetes.client.CoreV1Api"),
    ):
        yield KubernetesBackend(
            KubernetesBackendConfig(
                namespace="sync-test", client_configuration=client.Configuration()
            )
        )


@pytest.fixture
def sync_runtime() -> types.Runtime:
    """A Kubernetes runtime with the downstream-compatible optional-kind model."""
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="torch",
        image="trainer:test",
    )
    trainer.set_command(("torchrun",))
    return types.Runtime(
        name="sync-runtime", trainer=trainer, kind=types.RuntimeKind.CLUSTER_TRAINING_RUNTIME
    )


@pytest.mark.parametrize(
    "trainer",
    [
        TransformersTrainer(func=sync_train_function, output_dir="pvc://checkpoints/run"),
        TrainingHubTrainer(algorithm=TrainingHubAlgorithms.SFT),
        SpeculativeDecodingTrainer(
            verifier_model="test/model",
            mode=SpeculatorMode.TRAIN_ONLY,
            training_resources={"nvidia.com/gpu": 1, "memory": "16Gi", "cpu": "2"},
            hidden_states_path="pvc://checkpoints/hidden",
            data_path="pvc://checkpoints/data",
            output_dir="pvc://checkpoints/output",
            config=SpeculatorConfig(target_layer_ids=[0, 7, 14, 27]),
        ),
    ],
    ids=["transformers", "traininghub", "speculator"],
)
def test_rhai_dispatch_storage_progression_and_deadline(
    sync_backend: KubernetesBackend, sync_runtime: types.Runtime, trainer: RHAITrainer
) -> None:
    """Every RHAI type keeps its own dispatch and storage path with upstream deadlines."""
    trainer_cr = models.TrainerV1alpha1Trainer(command=["python", "train.py"])
    with (
        patch.object(
            rhai_utils, "get_trainer_cr_from_rhai_trainer", return_value=trainer_cr
        ) as build_rhai,
        patch.object(
            rhai_utils, "setup_rhai_trainer_storage", wraps=rhai_utils.setup_rhai_trainer_storage
        ) as setup_storage,
        patch(
            "kubeflow.trainer.backends.kubernetes.utils.validate_python_function",
            side_effect=AssertionError("RHAI must not use CustomTrainer validation"),
        ) as validate_custom,
    ):
        name = sync_backend.train(
            runtime=sync_runtime,
            trainer=trainer,
            options=[Name("sync-job"), ActiveDeadlineSeconds(600), Labels({"team": "training"})],
        )
    assert name == "sync-job"
    build_rhai.assert_called_once_with(sync_runtime, trainer, None)
    setup_storage.assert_called_once_with(
        trainer, trainer_cr, None, sync_backend.core_api, "sync-test"
    )
    validate_custom.assert_not_called()
    body = sync_backend.custom_api.create_namespaced_custom_object.call_args.args[4]
    assert body["spec"]["activeDeadlineSeconds"] == 600
    assert body["spec"]["trainer"]["command"] == ["python", "train.py"]
    assert body["metadata"]["labels"] == {"team": "training"}
    assert body["metadata"]["annotations"][ANNOTATION_PROGRESSION_TRACKING] == "true"
    if isinstance(trainer, (TransformersTrainer, SpeculativeDecodingTrainer)):
        pod = body["spec"]["runtimePatches"][0]["trainingRuntimeSpec"]["template"]["spec"][
            "replicatedJobs"
        ][0]["template"]["spec"]["template"]["spec"]
        assert pod["volumes"] == [
            {"name": CHECKPOINT_VOLUME_NAME, "persistentVolumeClaim": {"claimName": "checkpoints"}}
        ]
        assert pod["containers"][0]["volumeMounts"][0]["mountPath"] == CHECKPOINT_MOUNT_PATH


def test_s3_credentials_and_annotation_overrides_with_deadline(
    sync_backend: KubernetesBackend, sync_runtime: types.Runtime
) -> None:
    """S3 credentials remain secret references and user annotation overrides survive."""
    trainer = TransformersTrainer(
        func=sync_train_function,
        output_dir="s3://checkpoint-bucket/run",
        data_connection_name="checkpoint-connection",
    )
    sync_backend.core_api.read_namespaced_secret.return_value = Mock(
        data={"AWS_ACCESS_KEY_ID": "unused", "AWS_SECRET_ACCESS_KEY": "unused"}
    )
    with patch.object(
        rhai_utils,
        "get_trainer_cr_from_rhai_trainer",
        return_value=models.TrainerV1alpha1Trainer(command=["python", "train.py"]),
    ):
        sync_backend.train(
            runtime=sync_runtime,
            trainer=trainer,
            options=[ActiveDeadlineSeconds(120), Annotations({ANNOTATION_METRICS_PORT: "28081"})],
        )
    body = sync_backend.custom_api.create_namespaced_custom_object.call_args.args[4]
    assert body["spec"]["activeDeadlineSeconds"] == 120
    assert body["metadata"]["annotations"][ANNOTATION_METRICS_PORT] == "28081"
    sync_backend.core_api.read_namespaced_secret.assert_called_once_with(
        name="checkpoint-connection", namespace="sync-test"
    )
    env = body["spec"]["trainer"]["env"]
    assert {entry["name"] for entry in env} == {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
    for entry in env:
        assert entry["valueFrom"]["secretKeyRef"] == {
            "name": "checkpoint-connection",
            "key": entry["name"],
        }
        assert "value" not in entry
    pod = body["spec"]["runtimePatches"][0]["trainingRuntimeSpec"]["template"]["spec"][
        "replicatedJobs"
    ][0]["template"]["spec"]["template"]["spec"]
    assert pod["volumes"] == [{"name": CHECKPOINT_VOLUME_NAME, "emptyDir": {}}]


def test_custom_trainer_validation_stays_separate(
    sync_backend: KubernetesBackend, sync_runtime: types.Runtime
) -> None:
    """The upstream lambda rejection applies to CustomTrainer, without RHAI dispatch."""
    with (
        patch.object(rhai_utils, "get_trainer_cr_from_rhai_trainer") as build_rhai,
        patch.object(rhai_utils, "setup_rhai_trainer_storage") as setup_storage,
        pytest.raises(ValueError, match="Lambda functions are not supported"),
    ):
        sync_backend.train(runtime=sync_runtime, trainer=types.CustomTrainer(func=lambda: None))
    build_rhai.assert_not_called()
    setup_storage.assert_not_called()
    sync_backend.custom_api.create_namespaced_custom_object.assert_not_called()
