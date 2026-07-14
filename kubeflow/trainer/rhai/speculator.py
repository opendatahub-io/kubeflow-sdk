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

"""SpeculativeDecodingTrainer for custom draft model training via the speculators library.

This module provides the SpeculativeDecodingTrainer and SpeculatorConfig dataclasses
for training speculative decoding draft models (e.g., Eagle3) using the speculators
library. Supports TRAIN_ONLY and DATA_ONLY modes.
"""

from dataclasses import dataclass, field
from enum import Enum

from kubeflow_trainer_api import models

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.constants import PVC_URI_SCHEME
from kubeflow.trainer.types import types


class SpeculatorMode(Enum):
    """Training mode for speculator training.

    Args:
        TRAIN_ONLY: Train draft model from pre-extracted hidden states on PVC.
        DATA_ONLY: Extract hidden states from verifier model via managed vLLM sidecar.
        OFFLINE: Extract hidden states via user-managed vLLM endpoint, then train.
        ONLINE: Online training with a custom runtime image that includes all dependencies.
    """

    TRAIN_ONLY = "train_only"
    DATA_ONLY = "data_only"
    OFFLINE = "offline"
    ONLINE = "online"


class SpeculatorType(Enum):
    """Draft model architecture for speculative decoding.

    Args:
        EAGLE3: Eagle3 draft model architecture.
        DFLASH: DFlash draft model architecture.
        MTP: Multi-Token Prediction draft model architecture.
        PEAGLE: PEAGLE draft model architecture.
    """

    EAGLE3 = "eagle3"
    DFLASH = "dflash"
    MTP = "mtp"
    PEAGLE = "peagle"


_SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}


@dataclass
class SpeculatorConfig:
    """Advanced configuration for speculator training.

    Args:
        num_layers: Number of draft model layers (default: 1).
        ttt_steps: Test-time training steps (default: 3).
        norm_before_residual: Apply normalization before residual connection (default: True).
        norm_before_fc: Apply normalization before fully-connected layer (default: False).
        embed_requires_grad: Whether embedding layer requires gradients (default: False).
        hidden_states_dtype: PyTorch dtype for hidden states tensors (default: "bfloat16").
            Must match the verifier model's dtype. Supported: "bfloat16", "float16", "float32".
        scheduler_type: Learning rate scheduler type (default: "linear").
            Supported: "linear", "cosine", "none".
        scheduler_warmup_steps: Number of warmup steps for the learning rate scheduler.
            When ``None`` (default), computed as 1% of total training steps.
        scheduler_total_steps: Total number of steps for the learning rate scheduler.
            When ``None`` (default), computed as ``num_epochs * steps_per_epoch``.
        scheduler_num_cosine_cycles: Number of cosine cycles for the cosine scheduler
            (default: 0.5). Only used when ``scheduler_type="cosine"``.
        checkpoint_freq: Checkpoint frequency in epochs (default: 1.0).
        save_best: Save only the best model checkpoint by validation loss (default: False).
        log_freq: Logging frequency in steps (default: 1).
        resume_from_checkpoint: Resume training from an existing checkpoint (default: False).
        datagen_concurrency: Number of concurrent requests to vLLM for hidden state
            extraction (default: 4).
        target_layer_ids: Specific layer IDs for hidden state extraction. When ``None``,
            auto-selected from the verifier model architecture.
        from_pretrained: Path to a pretrained draft model to resume training from.
    """

    num_layers: int = 1
    ttt_steps: int = 3
    norm_before_residual: bool = True
    norm_before_fc: bool = False
    embed_requires_grad: bool = False
    hidden_states_dtype: str = "bfloat16"
    scheduler_type: str = "linear"
    scheduler_warmup_steps: int | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5
    checkpoint_freq: float = 1.0
    save_best: bool = False
    log_freq: int = 1
    resume_from_checkpoint: bool = False
    datagen_concurrency: int = 4
    target_layer_ids: list[int] | None = None
    from_pretrained: str | None = None


@dataclass
class SpeculativeDecodingTrainer:
    """RHAI trainer for custom draft model training via the speculators library.

    Args:
        verifier_model: HuggingFace model ID (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``)
            or PVC URI (``pvc://<name>/<path>``) pointing to a pre-downloaded model.
        mode: Training mode (TRAIN_ONLY or DATA_ONLY).
        speculator_type: Draft model architecture (default: EAGLE3).
        hidden_states_path: PVC URI (``pvc://<name>/<path>``) to pre-extracted hidden states
            (required for TRAIN_ONLY).
        data_path: PVC URI (``pvc://<name>/<path>``) to preprocessed Arrow dataset
            (required for TRAIN_ONLY).
        dataset_name: Dataset for hidden state extraction (required for DATA_ONLY).
            Built-in names (``"sharegpt"``, ``"ultrachat"``, ``"gsm8k"``), a HuggingFace
            dataset ID, or a local ``.json``/``.jsonl`` file path.
        max_samples: Maximum number of dataset samples to use for data generation.
            Useful for quick testing. When ``None`` (default), all samples are used.
        epochs: Training epochs (default: 3).
        lr: Learning rate (default: 1e-4).
        total_seq_len: Maximum sequence length for dataset preprocessing,
            vLLM context window, and training (default: 8192).
        draft_vocab_size: Vocabulary size for the draft model. When ``None`` (default),
            auto-computed as ``min(8192, verifier_vocab_size)``.
        training_gpu_count: Number of GPUs for training (default: 1).
        vllm_gpu_count: Number of GPUs for vLLM sidecar (default: 1).
        vllm_gpu_memory_utilization: Fraction of GPU memory for vLLM (default: 0.9).
        config: Advanced training configuration. See ``SpeculatorConfig``.
        packages_to_install: Python packages to install before training.
        pip_index_urls: PyPI index URLs for package installation.
        env: Environment variables to set in training pods.
        output_dir: Directory for saving outputs. Supports PVC URIs
            (pvc://<name>/<path>). The SDK auto-mounts the volume and resolves the path.
        regenerate_responses: When True, send dataset prompts to the verifier model
            and use its responses instead of the original dataset responses before
            preprocessing. Only supported in DATA_ONLY mode (default: False).
        enable_progression_tracking: Enable progression tracking (default: True).
        metrics_port: HTTP server port for metrics endpoint (default: 28080).
        metrics_poll_interval_seconds: How often controller polls metrics (default: 30).
    """

    verifier_model: str
    mode: SpeculatorMode
    speculator_type: SpeculatorType = SpeculatorType.EAGLE3
    hidden_states_path: str | None = None
    data_path: str | None = None
    dataset_name: str | None = None
    max_samples: int | None = None
    epochs: int = 3
    lr: float = 1e-4
    total_seq_len: int = 8192
    draft_vocab_size: int | None = None
    training_gpu_count: int = 1
    vllm_gpu_count: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    config: SpeculatorConfig | None = None
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: dict[str, str] | None = None
    output_dir: str | None = None
    regenerate_responses: bool = False

    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        supported_modes = {
            SpeculatorMode.TRAIN_ONLY,
            SpeculatorMode.DATA_ONLY,
        }
        if self.mode not in supported_modes:
            raise NotImplementedError(
                f"Mode '{self.mode.value}' is not yet supported. "
                f"Currently supported modes: {', '.join(m.value for m in supported_modes)}."
            )

        if self.speculator_type != SpeculatorType.EAGLE3:
            raise NotImplementedError(
                f"Speculator type '{self.speculator_type.value}' is not yet supported. "
                f"Currently only '{SpeculatorType.EAGLE3.value}' is supported."
            )

        if self.mode == SpeculatorMode.TRAIN_ONLY and not self.hidden_states_path:
            raise ValueError(
                "hidden_states_path is required for TRAIN_ONLY mode. "
                "Provide a PVC URI (pvc://<name>/<path>) to the pre-extracted hidden states."
            )

        if (
            self.mode == SpeculatorMode.TRAIN_ONLY
            and self.hidden_states_path
            and not self.hidden_states_path.startswith(PVC_URI_SCHEME)
        ):
            raise ValueError(
                f"hidden_states_path must use a PVC URI (pvc://<name>/<path>), "
                f"got {self.hidden_states_path!r}."
            )

        if self.mode == SpeculatorMode.TRAIN_ONLY and not self.data_path:
            raise ValueError(
                "data_path is required for TRAIN_ONLY mode. "
                "Provide a PVC URI (pvc://<name>/<path>) to the preprocessed Arrow dataset."
            )

        if self.data_path and not self.data_path.startswith(PVC_URI_SCHEME):
            raise ValueError(
                f"data_path must use a PVC URI (pvc://<name>/<path>), got {self.data_path!r}."
            )

        if not self.output_dir:
            raise ValueError(
                f"output_dir is required for {self.mode.name} mode. "
                "Provide a PVC URI (pvc://<name>/<path>)."
            )

        if not self.output_dir.startswith(PVC_URI_SCHEME):
            raise ValueError(
                f"output_dir must use a PVC URI (pvc://<name>/<path>), got {self.output_dir!r}. "
                "Shared PVC storage is required for the vLLM sidecar and main container "
                "to exchange hidden states."
            )

        if self.mode == SpeculatorMode.DATA_ONLY and not self.dataset_name:
            raise ValueError(
                "dataset_name is required for DATA_ONLY mode. "
                "Provide a HuggingFace dataset ID or name (e.g. 'sharegpt')."
            )

        if self.max_samples is not None:
            if self.mode != SpeculatorMode.DATA_ONLY:
                raise ValueError("max_samples is only supported in DATA_ONLY mode.")
            if not isinstance(self.max_samples, int) or self.max_samples < 1:
                raise ValueError(
                    f"max_samples must be a positive integer, got {self.max_samples!r}."
                )

        if self.regenerate_responses and self.mode != SpeculatorMode.DATA_ONLY:
            raise ValueError("regenerate_responses is only supported in DATA_ONLY mode.")

        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {self.epochs!r}.")

        if not isinstance(self.lr, (int, float)) or self.lr <= 0:
            raise ValueError(f"lr must be a positive number, got {self.lr!r}.")

        if not isinstance(self.total_seq_len, int) or self.total_seq_len < 1:
            raise ValueError(
                f"total_seq_len must be a positive integer, got {self.total_seq_len!r}."
            )

        if not isinstance(self.training_gpu_count, int) or self.training_gpu_count < 1:
            raise ValueError(
                f"training_gpu_count must be a positive integer, got {self.training_gpu_count!r}."
            )

        if not isinstance(self.vllm_gpu_count, int) or self.vllm_gpu_count < 1:
            raise ValueError(
                f"vllm_gpu_count must be a positive integer, got {self.vllm_gpu_count!r}."
            )

        if (
            not isinstance(self.vllm_gpu_memory_utilization, (int, float))
            or self.vllm_gpu_memory_utilization <= 0
            or self.vllm_gpu_memory_utilization > 1.0
        ):
            raise ValueError(
                f"vllm_gpu_memory_utilization must be in range (0, 1.0], "
                f"got {self.vllm_gpu_memory_utilization!r}."
            )

        if self.config is not None and self.config.hidden_states_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"config.hidden_states_dtype must be one of {_SUPPORTED_DTYPES}, "
                f"got '{self.config.hidden_states_dtype}'."
            )

        if not isinstance(self.metrics_port, int):
            raise ValueError(
                f"metrics_port must be an integer, got {type(self.metrics_port).__name__}"
            )
        if self.metrics_port < 1024 or self.metrics_port > 65535:
            raise ValueError(f"metrics_port must be in range 1024-65535, got {self.metrics_port}")

        if not isinstance(self.metrics_poll_interval_seconds, int):
            raise ValueError(
                f"metrics_poll_interval_seconds must be an integer, "
                f"got {type(self.metrics_poll_interval_seconds).__name__}"
            )
        if self.metrics_poll_interval_seconds < 5 or self.metrics_poll_interval_seconds > 300:
            raise ValueError(
                f"metrics_poll_interval_seconds must be in range 5-300 seconds, "
                f"got {self.metrics_poll_interval_seconds}"
            )

        if self.output_dir:
            from kubeflow.trainer.rhai.utils import normalize_and_validate_output_dir

            self.output_dir = normalize_and_validate_output_dir(self.output_dir)

        if (
            self.output_dir
            and "://" in self.output_dir
            and not self.output_dir.startswith(PVC_URI_SCHEME)
        ):
            raise NotImplementedError(
                f"output_dir scheme '{self.output_dir.split('://')[0]}://' is not yet supported "
                f"for SpeculativeDecodingTrainer. Currently only PVC URIs (pvc://<name>/<path>) "
                f"or direct paths are supported."
            )

        if self.hidden_states_path:
            from kubeflow.trainer.rhai.utils import normalize_and_validate_output_dir

            self.hidden_states_path = normalize_and_validate_output_dir(self.hidden_states_path)

        if (
            self.hidden_states_path
            and "://" in self.hidden_states_path
            and not self.hidden_states_path.startswith(PVC_URI_SCHEME)
        ):
            raise NotImplementedError(
                f"hidden_states_path scheme "
                f"'{self.hidden_states_path.split('://')[0]}://' is not yet supported "
                f"for SpeculativeDecodingTrainer. Currently only PVC URIs (pvc://<name>/<path>) "
                f"or direct paths are supported."
            )

        if self.dataset_name and self.dataset_name.startswith(PVC_URI_SCHEME):
            from kubeflow.trainer.rhai.utils import normalize_and_validate_output_dir

            self.dataset_name = normalize_and_validate_output_dir(self.dataset_name)

        if (
            self.dataset_name
            and "://" in self.dataset_name
            and not self.dataset_name.startswith(PVC_URI_SCHEME)
        ):
            raise NotImplementedError(
                f"dataset_name scheme "
                f"'{self.dataset_name.split('://')[0]}://' is not yet supported "
                f"for SpeculativeDecodingTrainer. Currently only PVC URIs (pvc://<name>/<path>), "
                f"direct paths, or HuggingFace dataset names are supported."
            )

        if self.mode == SpeculatorMode.DATA_ONLY:
            cfg = self.config or SpeculatorConfig()
            if self.verifier_model.startswith(PVC_URI_SCHEME):
                if cfg.target_layer_ids is None:
                    raise ValueError(
                        "config.target_layer_ids is required when verifier_model is a "
                        "PVC URI. The SDK cannot read the model config from the PVC to "
                        "auto-detect layers. Provide target_layer_ids explicitly via "
                        "SpeculatorConfig(target_layer_ids=[2, n//2, n-3])."
                    )
            else:
                if cfg.target_layer_ids is None:
                    try:
                        from transformers import AutoConfig

                        model_config = AutoConfig.from_pretrained(
                            self.verifier_model, trust_remote_code=False
                        )
                    except Exception as e:
                        raise ValueError(
                            f"verifier_model {self.verifier_model!r} is not a valid "
                            f"HuggingFace model ID. For DATA_ONLY mode, verifier_model "
                            f"must be either a HuggingFace model ID "
                            f"(e.g. 'meta-llama/Llama-3.1-8B-Instruct') or a PVC URI "
                            f"(pvc://<name>/<path>)."
                        ) from e
                    if hasattr(model_config, "text_config"):
                        model_config = model_config.text_config
                    n = model_config.num_hidden_layers
                    cfg.target_layer_ids = [2, n // 2, n - 3]
                    self.config = cfg


def _speculator_data_only(
    verifier_model: str,
    dataset_name: str,
    save_path: str,
    total_seq_len: int,
    max_samples: int | None = None,
    vllm_endpoint: str = "http://localhost:8234/v1",
    concurrency: int = 4,
    regenerate_responses: bool = False,
) -> None:
    """Data extraction function injected into pods via inspect.getsource().

    Extracts hidden states from the verifier model via a vLLM endpoint.
    The vLLM server is provided either by the Kubernetes sidecar container
    (DATA_ONLY mode).

    This function is NOT called directly in the SDK. It is extracted as source
    code and injected into the script that runs inside the container.
    """
    import os
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request

    from speculators.data_generation.preprocessing import load_and_preprocess_dataset

    hidden_states_dir = os.path.join(save_path, "hidden_states")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(hidden_states_dir, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    marker_name = f"{EXTRACTION_INCOMPLETE_MARKER}.rank-{rank}"  # noqa: F821
    incomplete_marker = os.path.join(hidden_states_dir, marker_name)

    if os.path.exists(incomplete_marker):
        print(
            f"[Kubeflow] Warning: Previous data extraction for rank {rank} did not complete. "
            "Restarting extraction from the beginning.",
            flush=True,
        )
    elif any(f.endswith(".safetensors") for f in os.listdir(hidden_states_dir)):
        print("[Kubeflow] Data extraction already completed. Skipping.", flush=True)
        if "_mark_data_complete" in globals():
            _mark_data_complete()  # noqa: F821
        return

    try:
        with open(incomplete_marker, "w") as f:
            f.write(f"Data extraction in progress (rank {rank})")
    except Exception as e:
        print(
            f"[Kubeflow] Warning: Failed to write sentinel file: {e}. "
            "Check local disk space and write permissions in output_dir.",
            flush=True,
        )

    if "_set_phase" in globals():
        _set_phase("checking_vllm", 5)  # noqa: F821

    endpoint = vllm_endpoint
    print(f"[Kubeflow] Using vLLM endpoint: {endpoint}", flush=True)
    health = vllm_endpoint.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    timeout_secs = 600
    start = time.time()
    print(f"[Kubeflow] Waiting for vLLM sidecar at {health} (timeout={timeout_secs}s)", flush=True)
    while time.time() - start < timeout_secs:
        try:
            urllib.request.urlopen(health, timeout=5)
            print("[Kubeflow] vLLM sidecar is ready", flush=True)
            break
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    else:
        sys.exit(f"vLLM endpoint not reachable within {timeout_secs}s")

    if regenerate_responses:
        from pathlib import Path

        if "_set_phase" in globals():
            _set_phase("regenerating_responses", 5)  # noqa: F821

        print(
            f"[Kubeflow] Regenerating responses from verifier model '{verifier_model}'", flush=True
        )

        regen_dataset_map = {
            "sharegpt": "magpie",
            "magpie": "magpie",
            "ultrachat": "ultrachat",
            "gsm8k": "gsm8k",
        }
        regen_dataset = regen_dataset_map.get(dataset_name, "magpie")
        regen_output = str(Path(save_path) / "regenerated_responses.jsonl")
        chat_endpoint = endpoint.rstrip("/").rsplit("/v1", 1)[0] + "/v1/chat/completions"

        regen_script_path = "/tmp/response_regeneration.py"
        if not os.path.exists(regen_script_path):
            import base64

            regen_content = base64.b64decode(_REGEN_SCRIPT_B64).decode("utf-8")  # noqa: F821
            with open(regen_script_path, "w") as f:
                f.write(regen_content)

        regen_cmd = [
            sys.executable,
            regen_script_path,
            "--endpoint",
            chat_endpoint,
            "--dataset",
            regen_dataset,
            "--outfile",
            regen_output,
        ]
        if max_samples is not None:
            regen_cmd.extend(["--limit", str(max_samples)])
        print(f"[Kubeflow] Regenerating responses using dataset '{regen_dataset}'", flush=True)
        regen_result = subprocess.run(regen_cmd, capture_output=False)
        if regen_result.returncode != 0:
            raise RuntimeError(
                f"response_regeneration.py exited with code {regen_result.returncode}"
            )
        print(f"[Kubeflow] Responses saved to {regen_output}", flush=True)
        dataset_name = regen_output

    if "_set_phase" in globals():
        _set_phase("preprocessing", 10)  # noqa: F821

    print(
        f"[Kubeflow] Preprocessing dataset '{dataset_name}' (seq_len={total_seq_len})", flush=True
    )

    token_freq_path = os.path.join(save_path, "token_freq.pt")
    preprocess_kwargs = {
        "target_model_path": verifier_model,
        "train_data_paths": [dataset_name],
        "seq_length": total_seq_len,
        "token_freq_path": token_freq_path,
    }
    if max_samples is not None:
        preprocess_kwargs["max_samples"] = max_samples
    dataset, processor = load_and_preprocess_dataset(**preprocess_kwargs)
    dataset.save_to_disk(save_path)
    print(
        f"[Kubeflow] Saved preprocessed dataset to {save_path} ({len(dataset)} samples)", flush=True
    )

    if "_start_data_progress_server" in globals():
        _start_data_progress_server(hidden_states_dir, len(dataset))  # noqa: F821

    if "_set_phase" in globals():
        _set_phase("extracting", 15)  # noqa: F821

    print(
        f"[Kubeflow] Extracting hidden states from '{verifier_model}' to '{hidden_states_dir}'",
        flush=True,
    )

    script_path = "/tmp/data_generation_offline.py"
    if not os.path.exists(script_path):
        import base64

        script_content = base64.b64decode(_DATAGEN_SCRIPT_B64).decode("utf-8")  # noqa: F821
        with open(script_path, "w") as f:
            f.write(script_content)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    datagen_cmd = [
        sys.executable,
        script_path,
        "--model",
        verifier_model,
        "--preprocessed-data",
        save_path,
        "--endpoint",
        endpoint,
        "--output",
        hidden_states_dir,
        "--concurrency",
        str(concurrency),
        "--world-size",
        str(world_size),
        "--rank",
        str(rank),
    ]
    if max_samples is not None:
        datagen_cmd.extend(["--max-samples", str(max_samples)])
    print(
        f"[Kubeflow] Running hidden state extraction (concurrency={concurrency}, rank={rank}/{world_size})",
        flush=True,
    )
    result = subprocess.run(datagen_cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"data_generation_offline.py exited with code {result.returncode}")

    if os.path.exists(incomplete_marker):
        try:
            os.remove(incomplete_marker)
        except Exception as e:
            print(
                f"[Kubeflow] Warning: Failed to remove sentinel file: {e}. "
                f"A stale marker may cause re-extraction on next run. "
                f"Remove it manually from: {incomplete_marker}",
                flush=True,
            )
    if "_set_phase" in globals():
        _set_phase("complete", 100)  # noqa: F821

    print(
        f"[Kubeflow] Data extraction complete. Hidden states saved to '{hidden_states_dir}'",
        flush=True,
    )


def _speculator_train_only(
    verifier_model: str,
    data_path: str,
    hidden_states_path: str,
    save_path: str,
    epochs: int,
    lr: float,
    total_seq_len: int,
    draft_vocab_size: int | None = None,
    hidden_states_dtype: str = "bfloat16",
    num_layers: int = 1,
    ttt_steps: int = 3,
    norm_before_residual: bool = True,
    norm_before_fc: bool = False,
    embed_requires_grad: bool = False,
    scheduler_type: str = "linear",
    scheduler_warmup_steps: int | None = None,
    scheduler_total_steps: int | None = None,
    scheduler_num_cosine_cycles: float = 0.5,
    checkpoint_freq: float = 1.0,
    save_best: bool = False,
    log_freq: int = 1,
    resume_from_checkpoint: bool = False,
    from_pretrained: str | None = None,
    target_layer_ids: list[int] | None = None,
) -> None:
    """Training function injected into pods via inspect.getsource().

    This function is NOT called directly in the SDK. It is extracted as source
    code and injected into the training script that runs inside the container.
    """
    import contextlib
    import os
    from pathlib import Path

    import numpy as np

    if "_set_phase" in globals():
        _set_phase("initializing", 0)  # noqa: F821
    from speculators.models.eagle3.core import Eagle3DraftModel
    from speculators.models.eagle3.data import shift_batch
    from speculators.train.data import ArrowDataset, create_collate_fn
    from speculators.train.distributed_batch_sampler import (
        MultipackDistributedBatchSamplerV2,
    )
    from speculators.train.noise_transforms import AddUniformNoise
    from speculators.train.trainer import Trainer, TrainerConfig
    from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    rank = int(os.environ.get("RANK", "0"))
    marker_name = f"{EXTRACTION_INCOMPLETE_MARKER}.rank-{rank}"  # noqa: F821
    hs_dir = os.path.join(hidden_states_path, "hidden_states")
    own_marker = os.path.join(hs_dir, marker_name)
    if os.path.exists(own_marker):
        raise RuntimeError(
            f"Incomplete data extraction detected at '{hidden_states_path}' "
            f"for rank {rank}. "
            "Re-run data extraction (DATA_ONLY mode) before training."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    verifier_config = AutoConfig.from_pretrained(verifier_model)
    target_vocab_size = verifier_config.vocab_size

    d2t_path = Path(data_path) / "d2t.npy"
    t2d_path = Path(data_path) / "t2d.npy"

    if d2t_path.exists() and t2d_path.exists():
        d2t = torch.from_numpy(np.load(str(d2t_path)))
        t2d = torch.from_numpy(np.load(str(t2d_path)))
        resolved_draft_vocab = len(d2t)
    else:
        resolved_draft_vocab = draft_vocab_size or min(8192, target_vocab_size)
        token_freq_path = Path(data_path) / "token_freq.pt"
        token_freq_dict = torch.load(str(token_freq_path), weights_only=True)
        d2t, t2d = build_vocab_mappings_from_distribution(
            token_freq_dict=token_freq_dict,
            draft_vocab_size=resolved_draft_vocab,
            target_vocab_size=target_vocab_size,
        )
        np.save(str(d2t_path), d2t.cpu().numpy())
        np.save(str(t2d_path), t2d.cpu().numpy())

    from_training_kwargs = {
        "draft_vocab_size": resolved_draft_vocab,
        "num_layers": num_layers,
        "norm_before_residual": norm_before_residual,
        "norm_before_fc": norm_before_fc,
        "embed_requires_grad": embed_requires_grad,
        "ttt_steps": ttt_steps,
        "verifier_name_or_path": verifier_model,
        "d2t": d2t,
        "t2d": t2d,
    }
    if from_pretrained is not None:
        from_training_kwargs["from_pretrained"] = from_pretrained
    if target_layer_ids is not None:
        from_training_kwargs["target_layer_ids"] = target_layer_ids
    model = Eagle3DraftModel.from_training_args(verifier_config, **from_training_kwargs)

    max_len = total_seq_len
    collate_fn = create_collate_fn(max_len, verifier_config.hidden_size)
    hs_dtype = getattr(torch, hidden_states_dtype)

    train_dataset = ArrowDataset(
        max_len=max_len,
        datapath=data_path,
        hidden_states_path=hs_dir,
        split_ratio=0.9,
        on_missing="skip",
        transform=AddUniformNoise(),
        hidden_states_dtype=hs_dtype,
    )
    val_dataset = ArrowDataset(
        max_len=max_len,
        datapath=data_path,
        hidden_states_path=hs_dir,
        split_ratio=-0.1,
        on_missing="skip",
        hidden_states_dtype=hs_dtype,
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    train_batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=max_len,
        lengths=train_dataset.approx_lengths,
        num_replicas=world_size,
        rank=rank,
    )
    val_batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=max_len,
        lengths=val_dataset.approx_lengths,
        num_replicas=world_size,
        rank=rank,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn)

    with contextlib.suppress(NameError):
        _set_steps_per_epoch(len(train_loader))  # noqa: F821

    config = TrainerConfig(
        lr=lr,
        num_epochs=epochs,
        save_path=save_path,
        resume_from_checkpoint=resume_from_checkpoint,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs={"shift_fn": shift_batch},
        val_call_kwargs={"shift_fn": shift_batch},
        scheduler_type=scheduler_type,
        scheduler_warmup_steps=scheduler_warmup_steps,
        scheduler_total_steps=scheduler_total_steps,
        scheduler_num_cosine_cycles=scheduler_num_cosine_cycles,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
        hidden_states_dtype=hs_dtype,
        log_freq=log_freq,
    )

    if "_set_phase" in globals():
        _set_phase("training", 15)  # noqa: F821

    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.run_training()

    if "_set_phase" in globals():
        _set_phase("complete", 100)  # noqa: F821


def _render_speculator_training_script(trainer: SpeculativeDecodingTrainer) -> str:
    """Generate a training script via inspect.getsource().

    Builds the script by composing shared pieces based on what each mode needs:
    - DATA_ONLY: data extraction only
    - TRAIN_ONLY: training only

    Args:
        trainer: SpeculativeDecodingTrainer configuration.

    Returns:
        Python source code string for the training script.
    """
    import inspect
    import textwrap

    from kubeflow.trainer.rhai.utils import parse_output_dir_uri

    resolved_output_dir, _ = parse_output_dir_uri(trainer.output_dir)

    needs_data = trainer.mode == SpeculatorMode.DATA_ONLY
    needs_train = trainer.mode == SpeculatorMode.TRAIN_ONLY

    cfg = trainer.config or SpeculatorConfig()

    from kubeflow.trainer.rhai.constants import EXTRACTION_INCOMPLETE_MARKER

    script = f"EXTRACTION_INCOMPLETE_MARKER = {EXTRACTION_INCOMPLETE_MARKER!r}\n\n"

    if needs_data:
        import base64
        from pathlib import Path

        datagen_script_path = Path(__file__).parent / "scripts" / "data_generation_offline.py"
        datagen_b64 = base64.b64encode(datagen_script_path.read_bytes()).decode("ascii")

        script += f'_DATAGEN_SCRIPT_B64 = "{datagen_b64}"\n\n'
        script += textwrap.dedent(inspect.getsource(_speculator_data_only))

    if needs_train:
        script += textwrap.dedent(inspect.getsource(_speculator_train_only))

    if trainer.verifier_model.startswith(PVC_URI_SCHEME):
        resolved_verifier_model, _ = parse_output_dir_uri(trainer.verifier_model)
    else:
        resolved_verifier_model = trainer.verifier_model

    if trainer.dataset_name and trainer.dataset_name.startswith(PVC_URI_SCHEME):
        resolved_dataset_name, _ = parse_output_dir_uri(trainer.dataset_name)
    else:
        resolved_dataset_name = trainer.dataset_name

    if trainer.hidden_states_path and trainer.hidden_states_path.startswith(PVC_URI_SCHEME):
        resolved_hidden_states, _ = parse_output_dir_uri(trainer.hidden_states_path)
    else:
        resolved_hidden_states = trainer.hidden_states_path

    if trainer.data_path and trainer.data_path.startswith(PVC_URI_SCHEME):
        resolved_data_path, _ = parse_output_dir_uri(trainer.data_path)
    else:
        resolved_data_path = trainer.data_path

    from kubeflow.trainer.rhai.constants import VLLM_SIDECAR_ENDPOINT

    data_call = (
        f"_speculator_data_only(\n"
        f"    verifier_model={resolved_verifier_model!r},\n"
        f"    dataset_name={resolved_dataset_name!r},\n"
        f"    save_path={resolved_output_dir!r},\n"
        f"    total_seq_len={trainer.total_seq_len!r},\n"
        f"    max_samples={trainer.max_samples!r},\n"
        f"    vllm_endpoint={VLLM_SIDECAR_ENDPOINT!r},\n"
        f"    concurrency={cfg.datagen_concurrency!r},\n"
        f"    regenerate_responses={trainer.regenerate_responses!r},\n"
        f")\n"
    )

    train_call = (
        f"_speculator_train_only(\n"
        f"    verifier_model={resolved_verifier_model!r},\n"
        f"    data_path={resolved_data_path!r},\n"
        f"    hidden_states_path={resolved_hidden_states!r},\n"
        f"    save_path={resolved_output_dir!r},\n"
        f"    epochs={trainer.epochs!r},\n"
        f"    lr={trainer.lr!r},\n"
        f"    total_seq_len={trainer.total_seq_len!r},\n"
        f"    draft_vocab_size={trainer.draft_vocab_size!r},\n"
        f"    hidden_states_dtype={cfg.hidden_states_dtype!r},\n"
        f"    num_layers={cfg.num_layers!r},\n"
        f"    ttt_steps={cfg.ttt_steps!r},\n"
        f"    norm_before_residual={cfg.norm_before_residual!r},\n"
        f"    norm_before_fc={cfg.norm_before_fc!r},\n"
        f"    embed_requires_grad={cfg.embed_requires_grad!r},\n"
        f"    scheduler_type={cfg.scheduler_type!r},\n"
        f"    scheduler_warmup_steps={cfg.scheduler_warmup_steps!r},\n"
        f"    scheduler_total_steps={cfg.scheduler_total_steps!r},\n"
        f"    scheduler_num_cosine_cycles={cfg.scheduler_num_cosine_cycles!r},\n"
        f"    checkpoint_freq={cfg.checkpoint_freq!r},\n"
        f"    save_best={cfg.save_best!r},\n"
        f"    log_freq={cfg.log_freq!r},\n"
        f"    resume_from_checkpoint={cfg.resume_from_checkpoint!r},\n"
        f"    from_pretrained={cfg.from_pretrained!r},\n"
        f"    target_layer_ids={cfg.target_layer_ids!r},\n"
        f")\n"
    )

    if trainer.mode == SpeculatorMode.DATA_ONLY:
        script += f"\n{data_call}"

    elif trainer.mode == SpeculatorMode.TRAIN_ONLY:
        script += f"\n{train_call}"

    return script


def _create_speculator_progression_instrumentation(
    metrics_port: int,
    mode: str,
    num_epochs: int = 0,
) -> tuple:
    """Unified instrumentation for all speculator modes (extracted via inspect.getsource).

    Handles progression tracking for DATA_ONLY (file counting) and TRAIN_ONLY (log
    interception).

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into training scripts.

    Args:
        metrics_port: Port for HTTP metrics server.
        mode: Speculator mode string ("data_only", "train_only").
        num_epochs: Total training epochs (used for train_only).

    Returns:
        Tuple of (apply_fn, start_data_fn, handler_class) for testing purposes.
    """
    import http.server
    import json
    import logging
    import os
    import threading
    import time

    _hidden_states_dir: str | None = None
    _total_samples: int = 0
    _data_start_time: float | None = None

    _train_start_time: float | None = None
    _steps_per_epoch: int | None = None
    _max_step_in_epoch0 = 0
    _last_global_step = 0
    _last_epoch = 0
    _latest_metrics: dict = {}
    _metrics_lock = threading.Lock()
    _termination_message_written = False
    _current_phase: str | None = None
    _phase_floor_pct: int = 0
    _training_started = False

    class MetricsHandler(logging.Handler):
        """Captures speculators.metrics log records in memory."""

        def emit(self, record):
            nonlocal \
                _steps_per_epoch, \
                _max_step_in_epoch0, \
                _last_global_step, \
                _last_epoch, \
                _latest_metrics, \
                _training_started, \
                _train_start_time
            try:
                msg = record.msg
                if not isinstance(msg, dict):
                    return
                with _metrics_lock:
                    if not _training_started:
                        _training_started = True
                        _train_start_time = time.time()
                    _latest_metrics = msg
                    if "global_step" in msg:
                        _last_global_step = msg["global_step"]
                    if "epoch" in msg:
                        _last_epoch = msg["epoch"]
                    if (
                        _steps_per_epoch is None
                        and "train" in msg
                        and "epoch" in msg
                        and "global_step" in msg
                    ):
                        if msg["epoch"] == 0:
                            _max_step_in_epoch0 = max(_max_step_in_epoch0, msg["global_step"])
                        elif msg["epoch"] >= 1 and _max_step_in_epoch0 >= 0:
                            _steps_per_epoch = _max_step_in_epoch0 + 1
            except (KeyError, TypeError, ValueError) as e:
                print(f"[Kubeflow] Warning: Failed to parse metrics record: {e}", flush=True)

    class SpeculatorMetricsHTTPHandler(http.server.BaseHTTPRequestHandler):
        """HTTP handler that serves mode-aware progress to the controller."""

        def do_GET(self):
            try:
                transformed = self._get_progress()
            except Exception as e:
                print(f"[Kubeflow] Failed to create progress metrics payload: {e}", flush=True)
                self.send_error(500)
            else:
                self._maybe_write_termination_message(transformed)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(transformed, indent=2).encode())

        def _get_progress(self):
            if mode == "data_only":
                return self._data_progress()
            elif mode == "train_only":
                return self._training_progress()
            return self._empty_response()

        def _data_progress(self):
            if _hidden_states_dir is None or _total_samples <= 0:
                return self._empty_response()

            try:
                count = len(
                    [
                        f
                        for f in os.listdir(_hidden_states_dir)
                        if f.startswith("hs_") and f.endswith(".safetensors")
                    ]
                )
            except FileNotFoundError:
                count = 0

            progress_pct = max(int(count / _total_samples * 100), _phase_floor_pct)

            estimated_remaining = None
            if _data_start_time and count > 0:
                elapsed = time.time() - _data_start_time
                remaining = _total_samples - count
                if remaining <= 0:
                    estimated_remaining = 0
                else:
                    time_per_sample = elapsed / count
                    estimated_remaining = int(remaining * time_per_sample)

            return {
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": estimated_remaining,
                "currentStep": count,
                "totalSteps": _total_samples,
                "currentEpoch": None,
                "totalEpochs": None,
                "currentPhase": _current_phase,
                "trainMetrics": None,
                "evalMetrics": None,
            }

        def _training_progress(self):
            with _metrics_lock:
                metrics_snapshot = dict(_latest_metrics)

            if not metrics_snapshot:
                response = self._empty_response()
                response["progressPercentage"] = _phase_floor_pct
                return response

            global_step = metrics_snapshot.get("global_step", _last_global_step)
            epoch = metrics_snapshot.get("epoch", _last_epoch)
            train_metrics = metrics_snapshot.get("train", {})
            val_metrics = metrics_snapshot.get("val", {})

            total_steps = None
            progress_pct = 0
            estimated_remaining = None

            if _steps_per_epoch and _steps_per_epoch > 0:
                total_steps = _steps_per_epoch * num_epochs
                if total_steps > 0:
                    completed_steps = global_step + 1
                    progress_pct = max(int(completed_steps / total_steps * 100), _phase_floor_pct)

                    if _train_start_time and completed_steps > 0:
                        elapsed = time.time() - _train_start_time
                        remaining_steps = total_steps - completed_steps
                        if remaining_steps <= 0:
                            estimated_remaining = 0
                        else:
                            time_per_step = elapsed / completed_steps
                            estimated_remaining = int(remaining_steps * time_per_step)

            loss_val = train_metrics.get("loss")
            lr_val = metrics_snapshot.get("lr")

            return {
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": estimated_remaining,
                "currentStep": global_step,
                "totalSteps": total_steps,
                "currentEpoch": epoch + 1,
                "totalEpochs": num_epochs,
                "currentPhase": _current_phase,
                "trainMetrics": {
                    "loss": f"{loss_val:.4f}" if loss_val is not None else None,
                    "learning_rate": f"{lr_val:.6f}" if lr_val is not None else None,
                },
                "evalMetrics": {
                    k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                    for k, v in val_metrics.items()
                }
                if val_metrics
                else {},
            }

        def _empty_response(self):
            return {
                "progressPercentage": _phase_floor_pct if _phase_floor_pct > 0 else None,
                "estimatedRemainingSeconds": None,
                "currentStep": None,
                "totalSteps": None,
                "currentEpoch": None,
                "totalEpochs": None,
                "currentPhase": _current_phase,
                "trainMetrics": None,
                "evalMetrics": None,
            }

        def _maybe_write_termination_message(self, metrics):
            nonlocal _termination_message_written
            progress = metrics.get("progressPercentage")
            if progress is not None and progress >= 100:
                with _metrics_lock:
                    if _termination_message_written:
                        return
                    try:
                        with open("/dev/termination-log", "w") as f:
                            f.write(json.dumps(metrics))
                        _termination_message_written = True
                        print("[Kubeflow] Complete. Final metrics saved.", flush=True)
                    except (OSError, ValueError, TypeError) as e:
                        print(
                            f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                            f"Controller will fall back to HTTP polling.",
                            flush=True,
                        )

        def log_message(self, format, *args):
            pass

    def _start_data_progress_server(hidden_states_dir, total_samples):
        nonlocal _hidden_states_dir, _total_samples, _data_start_time
        _hidden_states_dir = hidden_states_dir
        _total_samples = total_samples
        _data_start_time = time.time()
        print(
            f"[Kubeflow] Data progress tracking active "
            f"({total_samples} samples in {hidden_states_dir})",
            flush=True,
        )

    def set_steps_per_epoch(steps):
        nonlocal _steps_per_epoch
        _steps_per_epoch = steps

    def _mark_data_complete():
        nonlocal _training_started, _train_start_time
        _training_started = True
        _train_start_time = time.time()

    def _set_phase(phase: str, floor_pct: int = 0):
        nonlocal _current_phase, _phase_floor_pct, _termination_message_written
        _current_phase = phase
        _phase_floor_pct = floor_pct
        if floor_pct >= 100:
            with _metrics_lock:
                if not _termination_message_written:
                    metrics = {
                        "progressPercentage": 100,
                        "estimatedRemainingSeconds": 0,
                        "currentPhase": phase,
                    }
                    try:
                        with open("/dev/termination-log", "w") as f:
                            f.write(json.dumps(metrics))
                        _termination_message_written = True
                        print("[Kubeflow] Complete. Final metrics saved.", flush=True)
                    except (OSError, ValueError, TypeError) as e:
                        print(
                            f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                            f"Controller will fall back to HTTP polling.",
                            flush=True,
                        )

    def apply_progression_tracking():
        if mode == "train_only":
            handler = MetricsHandler()
            metrics_logger = logging.getLogger("speculators.metrics")
            metrics_logger.setLevel(logging.INFO)
            metrics_logger.addHandler(handler)

        try:
            server = http.server.HTTPServer(("0.0.0.0", metrics_port), SpeculatorMetricsHTTPHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print(f"[Kubeflow] Metrics server started on port {metrics_port}", flush=True)
        except OSError as e:
            print(
                f"[Kubeflow] Warning: Failed to start metrics server on port "
                f"{metrics_port}: {e}. Will continue without metrics server.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[Kubeflow] Warning: Unexpected error starting metrics server: {e}. "
                f"Will continue without metrics server.",
                flush=True,
            )

        return set_steps_per_epoch

    return (
        apply_progression_tracking,
        _start_data_progress_server,
        SpeculatorMetricsHTTPHandler,
        _mark_data_complete,
        _set_phase,
    )


def get_speculator_instrumentation_wrapper(
    metrics_port: int,
    mode: str,
    num_epochs: int = 0,
) -> str:
    """Generate self-contained instrumentation wrapper via inspect.getsource.

    Args:
        metrics_port: Port for HTTP metrics server.
        mode: Speculator mode string ("data_only", "train_only").
        num_epochs: Total training epochs.

    Returns:
        Python code as string with {{user_training_code}} placeholder.
    """
    import inspect
    import textwrap

    instrumentation_code = inspect.getsource(_create_speculator_progression_instrumentation)
    instrumentation_code = textwrap.dedent(instrumentation_code)

    wrapper = f"""\
# =============================================================================
# Kubeflow SDK - Speculator Progression Tracking Instrumentation
# Generated by kubeflow.trainer.rhai.speculator
# =============================================================================

import os as _instr_os
_local_rank = int(_instr_os.environ.get("LOCAL_RANK", "0"))

print("[Kubeflow] Initializing speculator progression tracking", flush=True)

# Instrumentation function definition
{instrumentation_code}

if _local_rank == 0:
    # Initialize and apply instrumentation
    (
        _apply_progression_tracking,
        _start_data_progress_server,
        _,
        _mark_data_complete,
        _set_phase,
    ) = _create_speculator_progression_instrumentation(
        metrics_port={metrics_port},
        mode={mode!r},
        num_epochs={num_epochs},
    )
    _set_steps_per_epoch = _apply_progression_tracking()
    print("[Kubeflow] Speculator progression tracking enabled", flush=True)

# =============================================================================
# USER CODE
# =============================================================================

{{{{user_training_code}}}}"""

    return wrapper


def _build_install_snippet(
    packages_to_install: list[str] | None,
    pip_index_urls: list[str],
) -> str:
    """Build the shell snippet to install Python packages if requested."""
    if not packages_to_install:
        return ""
    return k8s_utils.get_script_for_python_packages(
        packages_to_install,
        pip_index_urls,
    )


def _get_command_from_runtime(
    runtime: types.Runtime,
    func_code: str,
    func_file: str,
    install_snippet: str,
) -> list[str]:
    """Build command using runtime's command template.

    Args:
        runtime: Runtime configuration with command template.
        func_code: The training function code to execute.
        func_file: The filename to write the code to.
        install_snippet: Package installation script to prepend.

    Returns:
        Command list ready for trainer_crd.command.
    """
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_snippet:
                exec_script = install_snippet + exec_script
            command.append(exec_script)
        else:
            command.append(c)
    return command


def apply_speculator_sidecar_overrides(
    trainer: SpeculativeDecodingTrainer,
    pod_template_overrides: list,
) -> list:
    """Configure the vLLM sidecar init container via pod template overrides.

    Sets environment variables, PVC volume mount, and GPU resources on the
    ``vllm-sidecar`` init container defined in the ClusterTrainingRuntime.

    Args:
        trainer: SpeculativeDecodingTrainer with model path, GPU settings, and output_dir.
        pod_template_overrides: Existing pod template overrides list (mutated in place).

    Returns:
        Updated pod_template_overrides list.
    """
    from kubeflow.trainer.rhai.constants import (
        CHECKPOINT_MOUNT_PATH,
        CHECKPOINT_VOLUME_NAME,
        VLLM_SIDECAR_CONTAINER_NAME,
    )
    from kubeflow.trainer.rhai.utils import parse_output_dir_uri

    if trainer.output_dir.startswith(PVC_URI_SCHEME):
        resolved_output_dir, _ = parse_output_dir_uri(trainer.output_dir)
    else:
        resolved_output_dir = trainer.output_dir
    hs_path = f"{resolved_output_dir}/hidden_states"

    node_override = None
    for override in pod_template_overrides:
        target_jobs = override.get("targetJobs", [])
        if any(job.get("name") == constants.NODE for job in target_jobs):
            node_override = override
            break

    if node_override is None:
        node_override = {"targetJobs": [{"name": constants.NODE}], "spec": {}}
        pod_template_overrides.append(node_override)

    if "spec" not in node_override:
        node_override["spec"] = {}
    spec_dict = node_override["spec"]

    if "initContainers" not in spec_dict:
        spec_dict["initContainers"] = []

    if trainer.verifier_model.startswith(PVC_URI_SCHEME):
        resolved_verifier, _ = parse_output_dir_uri(trainer.verifier_model)
    else:
        resolved_verifier = trainer.verifier_model

    cfg = trainer.config or SpeculatorConfig()

    layer_ids_str = ",".join(str(lid) for lid in cfg.target_layer_ids)

    sidecar_env = [
        {"name": "SPECULATOR_VERIFIER_MODEL", "value": resolved_verifier},
        {"name": "SPECULATOR_HS_PATH", "value": hs_path},
        {
            "name": "SPECULATOR_GPU_MEM_UTIL",
            "value": str(trainer.vllm_gpu_memory_utilization),
        },
        {"name": "SPECULATOR_VLLM_GPU_COUNT", "value": str(trainer.vllm_gpu_count)},
        {"name": "SPECULATOR_TARGET_LAYER_IDS", "value": layer_ids_str},
    ]

    if trainer.env and "HF_TOKEN" in trainer.env:
        sidecar_env.append({"name": "HF_TOKEN", "value": trainer.env["HF_TOKEN"]})

    sidecar_override = {
        "name": VLLM_SIDECAR_CONTAINER_NAME,
        "env": sidecar_env,
        "resources": {
            "limits": {"nvidia.com/gpu": str(trainer.vllm_gpu_count)},
        },
    }
    if trainer.output_dir.startswith(PVC_URI_SCHEME):
        sidecar_override["volumeMounts"] = [
            {
                "name": CHECKPOINT_VOLUME_NAME,
                "mountPath": CHECKPOINT_MOUNT_PATH,
                "readOnly": False,
            }
        ]
    spec_dict["initContainers"].append(sidecar_override)

    return pod_template_overrides


def get_trainer_cr_from_speculator_trainer(
    runtime: types.Runtime,
    trainer: SpeculativeDecodingTrainer,
    initializer: types.Initializer | None = None,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD for SpeculativeDecodingTrainer.

    Args:
        runtime: Runtime configuration.
        trainer: SpeculativeDecodingTrainer configuration.
        initializer: Optional initializer configuration.

    Returns:
        Trainer CRD spec.
    """
    if trainer.mode == SpeculatorMode.DATA_ONLY:
        runtime.trainer.set_command(constants.DEFAULT_COMMAND)
    else:
        runtime.trainer.set_command(constants.TORCH_COMMAND)

    trainer_crd = models.TrainerV1alpha1Trainer()

    if trainer.mode == SpeculatorMode.TRAIN_ONLY:
        trainer_crd.resources_per_node = k8s_utils.get_resources_per_node(
            {"nvidia.com/gpu": trainer.training_gpu_count}
        )

    install_snippet = _build_install_snippet(trainer.packages_to_install, trainer.pip_index_urls)

    func_code = _render_speculator_training_script(trainer)
    func_file = "speculator_train.py"

    if trainer.enable_progression_tracking:
        wrapper_code = get_speculator_instrumentation_wrapper(
            metrics_port=trainer.metrics_port,
            mode=trainer.mode.value,
            num_epochs=trainer.epochs,
        )
        func_code = wrapper_code.replace("{{user_training_code}}", func_code)

    trainer_crd.command = _get_command_from_runtime(
        runtime=runtime,
        func_code=func_code,
        func_file=func_file,
        install_snippet=install_snippet,
    )

    trainer_crd.env = (
        [models.IoK8sApiCoreV1EnvVar(name=k, value=v) for k, v in trainer.env.items()]
        if trainer.env
        else None
    )

    return trainer_crd
