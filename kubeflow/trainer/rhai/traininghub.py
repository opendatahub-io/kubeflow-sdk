from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect
import os
import textwrap
from typing import Optional

from kubeflow_trainer_api import models

import kubeflow.trainer.utils.utils as k8s_utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class TrainingHubAlgorithms(Enum):
    """Algorithm for TrainingHub Trainer."""

    SFT = "sft"
    OSFT = "osft"


@dataclass
class TrainingHubTrainer:
    """TrainingHub RHAI trainer configuration.

    Notes:
        - volumes and volume_mounts are intentionally not supported per requirements.
    """

    func: Optional[Callable] = None
    func_args: Optional[dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: [constants.DEFAULT_PIP_INDEX_URL]
    )
    env: Optional[dict[str, str]] = None
    algorithm: Optional[TrainingHubAlgorithms] = None

    # Progress tracking parameters
    enable_progress_tracking: bool = True
    metrics_port: int = 28080
    metrics_poll_interval_seconds: int = 30


def _derive_topology_from_func_args(func_args: Optional[dict]) -> tuple[int, int]:
    """Return (nnodes, nproc_per_node) based on provided func_args with safe defaults."""
    nnodes = 1
    nproc_per_node = 1
    if isinstance(func_args, dict):
        if isinstance(func_args.get("nnodes"), int):
            nnodes = func_args["nnodes"]
        if isinstance(func_args.get("nproc_per_node"), int):
            nproc_per_node = func_args["nproc_per_node"]
    return nnodes, nproc_per_node


def _build_install_snippet(
    runtime: types.Runtime,
    packages_to_install: Optional[list[str]],
    pip_index_urls: list[str],
) -> str:
    """Build the shell snippet to install Python packages if requested."""
    # Check if runtime has trainer and command attribute
    is_mpi = False
    if runtime and runtime.trainer:
        try:
            is_mpi = len(runtime.trainer.command) > 0 and runtime.trainer.command[0] == "mpirun"
        except (AttributeError, TypeError):
            # If command attribute doesn't exist or isn't accessible, assume not MPI
            is_mpi = False
    
    if packages_to_install:
        return k8s_utils.get_script_for_python_packages(
            packages_to_install,
            pip_index_urls[0] if pip_index_urls else "https://pypi.org/simple",
            is_mpi,
        )
    return ""


def _render_algorithm_wrapper(algorithm_name: str, func_args: Optional[dict]) -> str:
    """Render a small Python script that calls training_hub.<algorithm>(**func_args)."""
    base_script = textwrap.dedent("""
    def training_func(func_args):
        import os
        from training_hub import {algo}

        _dp = (func_args or {{}}).get('data_path')
        if _dp:
            print("[PY] Data file found: {{}}".format(_dp))
        else:
            print("[PY] Data file NOT found: {{}}".format(_dp))

        args = dict(func_args or {{}})
        print("[PY] Launching {algo_upper} training...")
        try:
            result = {algo}(**args)
            print("[PY] {algo_upper} training complete. Result=", result)
        except ValueError as e:
            print("Configuration error:", e)
        except Exception as e:
            import traceback
            print("[PY] Training failed with error:", e)
            traceback.print_exc()

    """).format(algo=algorithm_name, algo_upper=algorithm_name.upper())

    if func_args is None:
        call_line = "training_func({})\n"
    elif isinstance(func_args, dict):
        params_lines: list[str] = ["training_func({\n"]
        for key, value in func_args.items():
            params_lines.append(f"    {repr(key)}: {repr(value)},\n")
        params_lines.append("})\n")
        call_line = "".join(params_lines)
    else:
        call_line = f"training_func({func_args})\n"

    return base_script + call_line


def _render_user_func_code(func: Callable, func_args: Optional[dict]) -> tuple[str, str]:
    """Return (func_code, func_file_basename) embedding the user function and call."""
    if not callable(func):
        raise ValueError(f"Training function must be callable, got function type: {type(func)}")

    func_code = inspect.getsource(func)
    func_code = textwrap.dedent(func_code)

    if func_args is None:
        call_block = f"{func.__name__}()"
    elif isinstance(func_args, dict):
        params_lines: list[str] = [f"{func.__name__}({{"]
        for key, value in func_args.items():
            params_lines.append(f"    {repr(key)}: {repr(value)},")
        params_lines.append("})")
        call_block = "\n".join(params_lines)
    else:
        call_block = f"{func.__name__}({func_args})"

    func_code = f"{func_code}\n{call_block}\n"
    func_file = os.path.basename(inspect.getfile(func))
    return func_code, func_file


def _compose_exec_script(func_code: str, func_file: str) -> str:
    """Compose the final exec script body using the common template."""
    return constants.EXEC_FUNC_SCRIPT.replace("__ENTRYPOINT__", "python").format(
        func_code=func_code,
        func_file=func_file,
    )


def _get_training_hub_progress_instrumentation(
    algorithm: str,
    ckpt_output_dir: str,
    port: int = 28080,
) -> str:
    """Generate HTTP server code for file-based progress tracking.
    
    This function generates Python code that:
    1. Reads JSONL metrics files written by Training Hub backends (Mini-Trainer or InstructLab Training)
    2. Transforms the metrics to a rich schema compatible with the UI Backend
    3. Serves the metrics via HTTP on the specified port
    
    The generated code runs in the training pod and doesn't require any modifications
    to the Training Hub library itself.
    
    Args:
        algorithm: The Training Hub algorithm ("sft" or "osft")
        ckpt_output_dir: Directory where metrics JSONL files are written
        port: HTTP server port (default: 28080)
    
    Returns:
        Python code string to be injected into the training script
    """
    return f'''
# ==============================================================================
# Kubeflow Training Hub Progress Tracking - File-Based HTTP Server
# ==============================================================================

import http.server
import json
import glob
import os
import threading

class TrainingHubMetricsHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that reads JSONL metrics from Training Hub backends."""
    
    algorithm = "{algorithm}"
    ckpt_output_dir = "{ckpt_output_dir}"
    
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
            
        elif self.path in ("/", "/metrics"):
            try:
                # Read latest metrics from JSONL file
                metrics = self._read_latest_metrics()
                # Transform to rich UI Backend schema
                transformed = self._transform_schema(metrics)
                # Serve JSON
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(transformed, indent=2).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_response = {{"error": str(e), "status": "error"}}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def _read_latest_metrics(self):
        """Read last line of JSONL file (most recent metrics from rank 0)."""
        # Determine file pattern based on algorithm
        if self.algorithm == "osft":
            # Mini-Trainer: Read from rank 0 file explicitly
            metrics_file = f"{{self.ckpt_output_dir}}/training_metrics_0.jsonl"
        else:  # sft
            # InstructLab: Find rank 0 file (global0.jsonl)
            pattern = f"{{self.ckpt_output_dir}}/training_params_and_metrics_global*.jsonl"
            files = glob.glob(pattern)
            
            if not files:
                return {{}}
            
            # Prefer rank 0 file
            rank_0_files = [f for f in files if 'global0.jsonl' in f]
            metrics_file = rank_0_files[0] if rank_0_files else files[0]
        
        # Read configuration from first line and metrics from last line
        try:
            if not os.path.exists(metrics_file):
                return {{}}
            
            config = {{}}
            last_line = None
            
            with open(metrics_file, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        if i == 0:
                            # First line contains training configuration
                            try:
                                config = json.loads(line)
                            except:
                                pass
                        last_line = line
            
            if last_line:
                metrics = json.loads(last_line)
                # Merge config into metrics for access to num_epochs
                if config:
                    metrics['_config'] = config
                return metrics
        except FileNotFoundError:
            return {{}}
        except json.JSONDecodeError:
            return {{}}
        
        return {{}}
    
    def _transform_schema(self, metrics):
        """Transform backend schema → Controller-compatible progress format."""
        if not metrics:
            return {{
                "progressPercentage": 0,
                "estimatedRemainingSeconds": None,
                "currentStep": 0,
                "totalSteps": 0,
                "currentEpoch": 1,
                "totalEpochs": 1,
                "trainMetrics": {{
                    "loss": None,
                    "learning_rate": None,
                    "grad_norm": None,
                }},
                "evalMetrics": {{}},
                "elapsedSeconds": None,
            }}
        
        # Detect backend based on metrics keys
        if "tokens_per_second" in metrics or "samples_per_second" in metrics:
            # Mini-Trainer (OSFT)
            return self._transform_mini_trainer(metrics)
        else:
            # InstructLab Training (SFT)
            return self._transform_instructlab(metrics)
    
    def _transform_mini_trainer(self, metrics):
        """Transform Mini-Trainer (OSFT) schema to rich progress format with controller compatibility."""
        step = metrics.get("step", 0)
        epoch = metrics.get("epoch", 0)
        steps_per_epoch = metrics.get("steps_per_epoch", 0)
        
        # Calculate total steps across all epochs
        total_epochs = metrics.get("total_epochs", 1)
        step_total = steps_per_epoch * total_epochs if steps_per_epoch > 0 else step
        current_step_absolute = (epoch * steps_per_epoch) + step if steps_per_epoch > 0 else step
        
        # Calculate progress percentage (cap at 100%)
        percent = min(100, (current_step_absolute / step_total * 100)) if step_total > 0 else 0
        
        # Estimate remaining time
        time_per_batch = metrics.get("time_per_batch", 0)
        remaining_steps = step_total - current_step_absolute
        
        # If training is complete (100%), set remaining time to 0
        if percent >= 100 or remaining_steps <= 0:
            estimated_remaining_sec = 0
        else:
            estimated_remaining_sec = int(remaining_steps * time_per_batch) if time_per_batch > 0 else None
        
        loss_val = metrics.get("loss", 0)
        lr_val = metrics.get("lr")
        grad_norm_val = metrics.get("grad_norm", 0)
        throughput_val = metrics.get("samples_per_second", 0)
        val_loss_val = metrics.get("val_loss")
        
        return {{
            # Controller-compatible format  
            "progressPercentage": int(round(percent)),
            "estimatedRemainingSeconds": estimated_remaining_sec,
            "currentStep": current_step_absolute,
            "totalSteps": step_total,
            "currentEpoch": epoch + 1,
            "totalEpochs": total_epochs,
            "trainMetrics": {{
                "loss": f"{{loss_val:.4f}}" if loss_val else None,
                "learning_rate": f"{{lr_val:.6f}}" if lr_val else None,
                "grad_norm": f"{{grad_norm_val:.4f}}" if grad_norm_val else None,
            }},
            "evalMetrics": {{
                "eval_loss": f"{{val_loss_val:.4f}}" if val_loss_val else None,
            }},
            "elapsedSeconds": None,
        }}
    
    def _transform_instructlab(self, metrics):
        """Transform InstructLab Training (SFT) schema to controller-compatible format."""
        step = metrics.get("step", 0)
        epoch = metrics.get("epoch", 0)
        num_epoch_steps = metrics.get("num_epoch_steps", 0)
        total_samples = metrics.get("total_samples", 0)
        
        # Extract real num_epochs from config (first line of JSONL)
        config = metrics.get("_config", {{}})
        configured_num_epochs = config.get("num_epochs")
        
        # Calculate total steps
        current_epoch = epoch + 1  # Current epoch number (0-indexed, so +1)
        current_step_absolute = step  # Current step (already properly indexed)
        
        # Use samples_seen to detect when we've moved to a new epoch
        samples_seen = metrics.get("samples_seen", 0)
        
        # Determine total epochs
        if configured_num_epochs:
            # Use the real configured value from training config!
            estimated_total_epochs = configured_num_epochs
        elif total_samples > 0 and samples_seen > 0:
            # Fallback: estimate based on samples_seen if config not available
            epochs_ratio = samples_seen / total_samples
            if epochs_ratio > 1.0:
                import math
                estimated_total_epochs = max(math.ceil(epochs_ratio), current_epoch)
            elif current_step_absolute > 0 and num_epoch_steps > 0:
                if current_step_absolute < num_epoch_steps * 0.9:
                    estimated_total_epochs = max(2, current_epoch)
                else:
                    estimated_total_epochs = current_epoch
            else:
                estimated_total_epochs = current_epoch
        else:
            # Final fallback
            estimated_total_epochs = max(2, current_epoch) if current_step_absolute > 0 else current_epoch
        
        # Calculate total expected steps based on current knowledge
        if num_epoch_steps > 0:
            step_total = num_epoch_steps * estimated_total_epochs
        else:
            # Fallback: use current step + some buffer
            step_total = max(step, step + 10)
        
        # Calculate progress percentage (cap at 100%)
        percent = min(100, (current_step_absolute / step_total * 100)) if step_total > 0 else 0
        
        # Estimate remaining time based on throughput
        throughput = metrics.get("overall_throughput", 0)  # samples/second
        remaining_steps = step_total - current_step_absolute
        
        # If training is complete (100%), set remaining time to 0
        if percent >= 100 or remaining_steps <= 0:
            estimated_remaining_sec = 0
        else:
            estimated_remaining_sec = int(remaining_steps / throughput) if throughput > 0 and remaining_steps > 0 else None
        
        # Extract training metrics from JSONL
        loss_val = metrics.get("avg_loss", 0)
        lr_val = metrics.get("lr")
        grad_norm_val = metrics.get("gradnorm")
        throughput_val = metrics.get("overall_throughput")
        
        return {{
            # Controller-compatible format
            "progressPercentage": int(round(percent)),
            "estimatedRemainingSeconds": estimated_remaining_sec,
            "currentStep": current_step_absolute,
            "totalSteps": step_total,
            "currentEpoch": current_epoch,
            "totalEpochs": estimated_total_epochs,
            "trainMetrics": {{
                "loss": f"{{loss_val:.4f}}" if loss_val else None,
                "learning_rate": f"{{lr_val:.6f}}" if lr_val is not None and lr_val > 0 else None,
                "grad_norm": f"{{grad_norm_val:.4f}}" if grad_norm_val is not None else None,
                "throughput": f"{{throughput_val:.2f}}" if throughput_val else None,
            }},
            "evalMetrics": {{}},
            "elapsedSeconds": None,
        }}
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""
        pass

def start_metrics_server(port={port}):
    """Start HTTP metrics server in background thread."""
    import socket
    
    # Enable SO_REUSEADDR to allow port reuse
    server = http.server.HTTPServer(("", port), TrainingHubMetricsHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    print(f"[Kubeflow] Metrics server started on port {{port}} for {{TrainingHubMetricsHandler.algorithm}}", flush=True)
    
    # Run server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    return server

# Start metrics server with error handling
try:
    _metrics_server = start_metrics_server()
    print("[Kubeflow] Progress tracking initialized for Training Hub", flush=True)
except OSError as e:
    if e.errno == 98:  # Address already in use
        print(f"[Kubeflow] Warning: Port {port} already in use, progress tracking may not work correctly", flush=True)
        _metrics_server = None
    else:
        raise

# ==============================================================================
# End of Progress Tracking Code
# ==============================================================================

'''


def get_trainer_crd_from_training_hub_trainer(
    runtime: types.Runtime,
    trainer: TrainingHubTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD for TrainingHub trainer."""
    trainer_crd = models.TrainerV1alpha1Trainer()

    # Derive numNodes and resourcesPerNode from func_args (defaults: 1)
    nnodes, nproc_per_node = _derive_topology_from_func_args(trainer.func_args)

    trainer_crd.num_nodes = nnodes
    trainer_crd.resources_per_node = k8s_utils.get_resources_per_node({"gpu": str(nproc_per_node)})

    install_snippet = _build_install_snippet(
        runtime, trainer.packages_to_install, trainer.pip_index_urls
    )

    # Primary case: no user function; generate wrapper that imports and calls algorithm(**func_args)
    if trainer.func is None:
        if not trainer.algorithm:
            raise ValueError("TrainingHubTrainer requires 'algorithm' when 'func' is not provided")

        algorithm_name = trainer.algorithm.value
        raw_code = _render_algorithm_wrapper(algorithm_name, trainer.func_args)
        
        # Inject progress tracking code if enabled
        if trainer.enable_progress_tracking and trainer.func_args and "ckpt_output_dir" in trainer.func_args:
            progress_code = _get_training_hub_progress_instrumentation(
                algorithm=algorithm_name,
                ckpt_output_dir=trainer.func_args["ckpt_output_dir"],
                port=trainer.metrics_port,
            )
            raw_code = progress_code + "\n" + raw_code
        
        exec_script = _compose_exec_script(raw_code, "training_script.py")
        full_script = install_snippet + exec_script

        trainer_crd.command = ["bash", "-c"]
        trainer_crd.args = [full_script]
    else:
        # Secondary case: user provided function; embed their function and call with kwargs
        func_code, func_file = _render_user_func_code(trainer.func, trainer.func_args)
        
        # Inject progress tracking code if enabled (for custom functions)
        # Try to extract ckpt_output_dir from the function code or use default
        if trainer.enable_progress_tracking and trainer.algorithm:
            # For custom functions, we inject progress tracking with a default checkpoint dir
            # User should ensure their function writes to this directory
            ckpt_dir = "/tmp/checkpoints"
            if trainer.func_args and "ckpt_output_dir" in trainer.func_args:
                ckpt_dir = trainer.func_args["ckpt_output_dir"]
            
            progress_code = _get_training_hub_progress_instrumentation(
                algorithm=trainer.algorithm.value,
                ckpt_output_dir=ckpt_dir,
                port=trainer.metrics_port,
            )
            func_code = progress_code + "\n" + func_code
        
        exec_script = _compose_exec_script(func_code, func_file)
        full_script = install_snippet + exec_script

        trainer_crd.command = ["bash", "-c"]
        trainer_crd.args = [full_script]

    # Add environment variables to the Trainer if provided by user
    trainer_crd.env = (
        [models.IoK8sApiCoreV1EnvVar(name=k, value=v) for k, v in trainer.env.items()]
        if trainer.env
        else None
    )

    return trainer_crd


def get_progress_tracking_annotations(trainer: TrainingHubTrainer) -> dict[str, str]:
    """Generate progress tracking annotations for TrainJob metadata.
    
    These annotations enable the trainer controller to poll the HTTP metrics endpoint
    and update the TrainJob status with real-time progress.
    
    Args:
        trainer: TrainingHubTrainer instance with progress tracking configuration
        
    Returns:
        Dictionary of annotations to add to TrainJob metadata
    """
    if not trainer.enable_progress_tracking:
        return {}
    
    return {
        # Enable progression tracking (controller will poll HTTP endpoint)
        "trainer.opendatahub.io/progression-tracking": "true",
        
        # Set metrics port (where HTTP server listens)
        "trainer.opendatahub.io/metrics-port": str(trainer.metrics_port),
        
        # Set metrics poll interval (how often controller polls)
        "trainer.opendatahub.io/metrics-poll-interval": f"{trainer.metrics_poll_interval_seconds}s",
        
        # Set framework annotation
        "trainer.opendatahub.io/framework": "traininghub",
    }
