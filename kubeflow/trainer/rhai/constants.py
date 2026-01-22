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

"""Constants for RHAI trainers"""

# Progression tracking annotations for OpenDataHub/RHAI
# These annotations enable real-time training progress monitoring via HTTP metrics
ANNOTATION_PROGRESSION_TRACKING = "trainer.opendatahub.io/progression-tracking"
ANNOTATION_METRICS_PORT = "trainer.opendatahub.io/metrics-port"
ANNOTATION_METRICS_POLL_INTERVAL = "trainer.opendatahub.io/metrics-poll-interval"
ANNOTATION_TRAINER_STATUS = "trainer.opendatahub.io/trainerStatus"

# Checkpointing storage constants (for PVC)
PVC_URI_SCHEME = "pvc://"
CHECKPOINT_MOUNT_PATH = "/mnt/kubeflow-checkpoints"
CHECKPOINT_VOLUME_NAME = "checkpoint-storage"
CHECKPOINT_INCOMPLETE_MARKER = "checkpoint-is-incomplete.txt"

# Ephemeral volume for cloud storage checkpoint staging (for S3)
CHECKPOINT_EPHEMERAL_VOLUME_SIZE = "50Gi"
CHECKPOINT_EPHEMERAL_STORAGE_CLASS = "default"

# Cloud storage URI schemes
S3_URI_SCHEME = "s3://"

# S3 credential environment variable keys (used in data connection secrets)
S3_CREDENTIAL_KEYS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "AWS_S3_ENDPOINT",
]
