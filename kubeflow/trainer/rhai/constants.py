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
ANNOTATION_FRAMEWORK = "trainer.opendatahub.io/framework"

# Checkpointing storage constants
PVC_URI_SCHEME = "pvc://"
CHECKPOINT_MOUNT_PATH = "/mnt/kubeflow-checkpoints"
CHECKPOINT_VOLUME_NAME = "checkpoint-storage"
CHECKPOINT_INCOMPLETE_MARKER = "checkpoint-is-incomplete.txt"

# Training Hub specific file names and patterns
# These are the JSONL metrics files written by Training Hub backends
TRAININGHUB_SFT_METRICS_FILE_PATTERN = "training_params_and_metrics_global*.jsonl"
TRAININGHUB_SFT_METRICS_FILE_RANK0 = "training_params_and_metrics_global0.jsonl"
TRAININGHUB_OSFT_METRICS_FILE_PATTERN = "training_metrics_*.jsonl"
TRAININGHUB_OSFT_METRICS_FILE_RANK0 = "training_metrics_0.jsonl"
TRAININGHUB_OSFT_CONFIG_FILE = "training_params.json"
