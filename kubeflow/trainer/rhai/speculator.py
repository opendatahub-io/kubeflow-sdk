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

This module provides the SpeculativeDecodingTrainer dataclass for training speculative
decoding draft models (e.g., Eagle3) using the speculators library. Currently
supports TRAIN_ONLY mode, which trains from pre-extracted hidden states on a PVC.
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
        DATA_ONLY: Extract hidden states from verifier model via vLLM (future).
        OFFLINE: Train using a user-managed vLLM endpoint (future).
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
class SpeculativeDecodingTrainer:
    """RHAI trainer for custom draft model training via the speculators library.

    Args:
        verifier_name_or_path: HuggingFace model ID or path to the verifier model.
        speculator_type: Draft model architecture (default: EAGLE3).
        mode: Training mode (TRAIN_ONLY for this ticket).
        hidden_states_path: Pre-extracted hidden states on PVC (required for TRAIN_ONLY).
        draft_vocab_size: Draft model vocabulary size.
        epochs: Training epochs (default: 3).
        lr: Learning rate (default: 1e-4).
        total_seq_len: Maximum total sequence length for batch packing (default: 8192).
        hidden_states_dtype: PyTorch dtype for hidden states tensors (default: "bfloat16").
            Must match the verifier model's dtype. Supported values: "bfloat16", "float16",
            "float32".
        num_nodes: Number of nodes for distributed training.
        resources_per_node: Computing resources per node.
            Example: {"nvidia.com/gpu": 2, "memory": "64Gi", "cpu": "8"}
        packages_to_install: Python packages to install before training.
        pip_index_urls: PyPI index URLs for package installation.
        env: Environment variables to set in training pods.
        output_dir: Directory for saving the trained model. Supports PVC URIs
            (pvc://<name>/<path>). The SDK auto-mounts the volume and resolves the path
            for the training script.
        enable_progression_tracking: Enable progression tracking.
        metrics_port: HTTP server port for metrics endpoint.
        metrics_poll_interval_seconds: How often controller polls metrics endpoint.
    """

    verifier_name_or_path: str
    speculator_type: SpeculatorType = SpeculatorType.EAGLE3
    mode: SpeculatorMode = SpeculatorMode.TRAIN_ONLY
    hidden_states_path: str | None = None
    draft_vocab_size: int | None = None
    epochs: int = 3
    lr: float = 1e-4
    total_seq_len: int = 8192
    hidden_states_dtype: str = "bfloat16"
    num_nodes: int | None = None
    resources_per_node: dict | None = None
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: dict[str, str] | None = None
    output_dir: str | None = None

    enable_progression_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.mode != SpeculatorMode.TRAIN_ONLY:
            raise NotImplementedError(
                f"Mode '{self.mode.value}' is not yet supported. "
                f"Currently only '{SpeculatorMode.TRAIN_ONLY.value}' is supported."
            )

        if self.speculator_type != SpeculatorType.EAGLE3:
            raise NotImplementedError(
                f"Speculator type '{self.speculator_type.value}' is not yet supported. "
                f"Currently only '{SpeculatorType.EAGLE3.value}' is supported."
            )

        if self.mode == SpeculatorMode.TRAIN_ONLY and not self.hidden_states_path:
            raise ValueError(
                "hidden_states_path is required for TRAIN_ONLY mode. "
                "Provide the path to pre-extracted hidden states on PVC."
            )

        if self.mode == SpeculatorMode.TRAIN_ONLY and not self.output_dir:
            raise ValueError(
                "output_dir is required for TRAIN_ONLY mode. "
                "Provide a PVC URI (pvc://<name>/<path>) for saving the trained draft model."
            )

        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {self.epochs!r}.")

        if not isinstance(self.lr, (int, float)) or self.lr <= 0:
            raise ValueError(f"lr must be a positive number, got {self.lr!r}.")

        if not isinstance(self.total_seq_len, int) or self.total_seq_len < 1:
            raise ValueError(
                f"total_seq_len must be a positive integer, got {self.total_seq_len!r}."
            )

        if self.draft_vocab_size is not None and (
            not isinstance(self.draft_vocab_size, int) or self.draft_vocab_size < 1
        ):
            raise ValueError(
                f"draft_vocab_size must be a positive integer, got {self.draft_vocab_size!r}."
            )

        if self.hidden_states_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"hidden_states_dtype must be one of {_SUPPORTED_DTYPES}, "
                f"got '{self.hidden_states_dtype}'."
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


def _speculator_train_only(
    verifier_name_or_path: str,
    hidden_states_path: str,
    save_path: str,
    epochs: int,
    lr: float,
    total_seq_len: int,
    hidden_states_dtype: str,
    draft_vocab_size: int | None,
) -> None:
    """Training function injected into pods via inspect.getsource().

    This function is NOT called directly in the SDK. It is extracted as source
    code and injected into the training script that runs inside the container.
    """
    import contextlib
    import os

    from speculators.models.eagle3.core import Eagle3DraftModel
    from speculators.models.eagle3.data import shift_batch
    from speculators.train.data import ArrowDataset, create_collate_fn
    from speculators.train.distributed_batch_sampler import (
        MultipackDistributedBatchSamplerV2,
    )
    from speculators.train.noise_transforms import AddUniformNoise
    from speculators.train.trainer import Trainer, TrainerConfig
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
    if draft_vocab_size is None:
        draft_vocab_size = verifier_config.vocab_size

    model = Eagle3DraftModel.from_training_args(
        verifier_config,
        draft_vocab_size=draft_vocab_size,
        norm_before_residual=True,
        ttt_steps=3,
        verifier_name_or_path=verifier_name_or_path,
    )

    max_len = total_seq_len
    collate_fn = create_collate_fn(max_len, verifier_config.hidden_size)
    hs_dtype = getattr(torch, hidden_states_dtype)

    train_dataset = ArrowDataset(
        max_len=max_len,
        datapath=hidden_states_path,
        split_ratio=0.9,
        on_missing="skip",
        transform=AddUniformNoise(),
        hidden_states_dtype=hs_dtype,
    )
    val_dataset = ArrowDataset(
        max_len=max_len,
        datapath=hidden_states_path,
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
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn)

    with contextlib.suppress(NameError):
        _set_steps_per_epoch(len(train_loader))  # noqa: F821

    config = TrainerConfig(
        lr=lr,
        num_epochs=epochs,
        save_path=save_path,
        is_distributed=is_distributed,
        local_rank=local_rank,
        train_call_kwargs={"shift_fn": shift_batch},
        val_call_kwargs={"shift_fn": shift_batch},
    )

    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.run_training()


def _render_speculator_training_script(trainer: SpeculativeDecodingTrainer) -> str:
    """Generate a training script via inspect.getsource().

    Extracts ``_speculator_train_only`` as source code and appends a call
    with the trainer's configuration values injected via ``repr()``.

    Args:
        trainer: SpeculativeDecodingTrainer configuration.

    Returns:
        Python source code string for the training script.
    """
    import inspect
    import textwrap

    from kubeflow.trainer.rhai.utils import parse_output_dir_uri

    resolved_output_dir, _ = parse_output_dir_uri(trainer.output_dir)
    resolved_hidden_states, _ = parse_output_dir_uri(trainer.hidden_states_path)

    func_code = inspect.getsource(_speculator_train_only)
    func_code = textwrap.dedent(func_code)

    call_code = (
        f"\n_speculator_train_only(\n"
        f"    verifier_name_or_path={trainer.verifier_name_or_path!r},\n"
        f"    hidden_states_path={resolved_hidden_states!r},\n"
        f"    save_path={resolved_output_dir!r},\n"
        f"    epochs={trainer.epochs!r},\n"
        f"    lr={trainer.lr!r},\n"
        f"    total_seq_len={trainer.total_seq_len!r},\n"
        f"    hidden_states_dtype={trainer.hidden_states_dtype!r},\n"
        f"    draft_vocab_size={trainer.draft_vocab_size!r},\n"
        f")\n"
    )

    return func_code + call_code


def _create_speculator_progression_instrumentation(
    metrics_port: int,
    num_epochs: int,
    save_path: str,
) -> tuple:
    """Instrumentation code injected into training pods (extracted via inspect.getsource).

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into training scripts.

    Args:
        metrics_port: Port for HTTP metrics server.
        num_epochs: Total training epochs for progress calculation.
        save_path: Directory for checkpoint output (unused, kept for API compat).

    Returns:
        Tuple of (apply_fn, handler_class) for testing purposes.
    """
    import http.server
    import json
    import logging
    import threading
    import time

    _start_time: float | None = None
    _steps_per_epoch: int | None = None
    _max_step_in_epoch0 = 0
    _last_global_step = 0
    _last_epoch = 0
    _latest_metrics: dict = {}
    _metrics_lock = threading.Lock()
    _termination_message_written = False

    class MetricsHandler(logging.Handler):
        """Captures speculators.metrics log records in memory."""

        def emit(self, record):
            nonlocal \
                _steps_per_epoch, \
                _max_step_in_epoch0, \
                _last_global_step, \
                _last_epoch, \
                _latest_metrics
            try:
                msg = record.msg
                if not isinstance(msg, dict):
                    return
                with _metrics_lock:
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
        """HTTP handler that serves in-memory metrics to the controller."""

        def do_GET(self):
            try:
                with _metrics_lock:
                    metrics_snapshot = dict(_latest_metrics)
                transformed = self._transform(metrics_snapshot)
            except Exception as e:
                print(f"[Kubeflow] Failed to create progress metrics payload: {e}", flush=True)
                self.send_error(500)
            else:
                self._maybe_write_termination_message(transformed)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(transformed, indent=2).encode())

        def _transform(self, metrics):
            if not metrics:
                return {
                    "progressPercentage": None,
                    "estimatedRemainingSeconds": None,
                    "currentStep": None,
                    "totalSteps": None,
                    "currentEpoch": None,
                    "totalEpochs": None,
                    "trainMetrics": None,
                    "evalMetrics": None,
                }

            global_step = metrics.get("global_step", _last_global_step)
            epoch = metrics.get("epoch", _last_epoch)
            train_metrics = metrics.get("train", {})
            val_metrics = metrics.get("val", {})

            total_steps = None
            progress_pct = 0
            estimated_remaining = None

            if _steps_per_epoch and _steps_per_epoch > 0:
                total_steps = _steps_per_epoch * num_epochs
                if total_steps > 0:
                    completed_steps = global_step + 1
                    progress = completed_steps / total_steps * 100
                    progress_pct = min(100, int(round(progress)))

                    if _start_time and completed_steps > 0:
                        elapsed = time.time() - _start_time
                        remaining_steps = total_steps - completed_steps
                        if progress_pct >= 100 or remaining_steps <= 0:
                            estimated_remaining = 0
                        else:
                            time_per_step = elapsed / completed_steps
                            estimated_remaining = int(remaining_steps * time_per_step)

            loss_val = train_metrics.get("loss")
            lr_val = metrics.get("lr")

            return {
                "progressPercentage": progress_pct,
                "estimatedRemainingSeconds": estimated_remaining,
                "currentStep": global_step,
                "totalSteps": total_steps,
                "currentEpoch": epoch + 1,
                "totalEpochs": num_epochs,
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

        def _maybe_write_termination_message(self, metrics):
            nonlocal _termination_message_written
            if _termination_message_written:
                return
            progress = metrics.get("progressPercentage")
            if progress is not None and progress >= 100:
                try:
                    with open("/dev/termination-log", "w") as f:
                        f.write(json.dumps(metrics))
                    _termination_message_written = True
                    print("[Kubeflow] Training complete. Final metrics saved.", flush=True)
                except (OSError, ValueError, TypeError) as e:
                    print(
                        f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                        f"Controller will fall back to HTTP polling.",
                        flush=True,
                    )

        def log_message(self, format, *args):
            pass

    def set_steps_per_epoch(steps):
        nonlocal _steps_per_epoch
        _steps_per_epoch = steps

    def apply_progression_tracking():
        nonlocal _start_time
        _start_time = time.time()

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
                f"{metrics_port}: {e}. Training will continue without metrics server.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[Kubeflow] Warning: Unexpected error starting metrics server: {e}. "
                f"Training will continue without metrics server.",
                flush=True,
            )

        return set_steps_per_epoch

    return (apply_progression_tracking, SpeculatorMetricsHTTPHandler, set_steps_per_epoch)


def get_speculator_instrumentation_wrapper(
    metrics_port: int,
    num_epochs: int,
    save_path: str,
) -> str:
    """Generate self-contained instrumentation wrapper via inspect.getsource.

    Args:
        metrics_port: Port for HTTP metrics server.
        num_epochs: Total training epochs.
        save_path: Directory for checkpoint output (passed through for API compat).

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

print("[Kubeflow] Initializing speculator progression tracking", flush=True)

# Instrumentation function definition
{instrumentation_code}

# Initialize and apply instrumentation
(
    apply_progression_tracking,
    _,
    _,
) = _create_speculator_progression_instrumentation(
    metrics_port={metrics_port},
    num_epochs={num_epochs},
    save_path={save_path!r},
)
_set_steps_per_epoch = apply_progression_tracking()
print("[Kubeflow] Speculator progression tracking enabled", flush=True)

# =============================================================================
# USER TRAINING CODE
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
    runtime.trainer.set_command(constants.TORCH_COMMAND)

    trainer_crd = models.TrainerV1alpha1Trainer()

    if trainer.num_nodes is not None:
        trainer_crd.num_nodes = trainer.num_nodes

    if trainer.resources_per_node:
        trainer_crd.resources_per_node = k8s_utils.get_resources_per_node(
            trainer.resources_per_node
        )

    install_snippet = _build_install_snippet(trainer.packages_to_install, trainer.pip_index_urls)

    func_code = _render_speculator_training_script(trainer)
    func_file = "speculator_train.py"

    if trainer.enable_progression_tracking:
        from kubeflow.trainer.rhai.utils import parse_output_dir_uri

        resolved_path, _ = parse_output_dir_uri(trainer.output_dir)
        wrapper_code = get_speculator_instrumentation_wrapper(
            metrics_port=trainer.metrics_port,
            num_epochs=trainer.epochs,
            save_path=resolved_path,
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
