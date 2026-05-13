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

"""Training Hub progression tracking instrumentation."""


def create_traininghub_progression_instrumentation(
    algorithm_metadata: dict,
    ckpt_output_dir: str,
    metrics_port: int,
) -> tuple:
    """Instrumentation code injected into training pods (extracted via inspect.getsource).

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into training scripts. This approach
    provides syntax highlighting, testability, and type checking while avoiding
    string templates.

    The algorithm metadata is pre-resolved from the centralized registry and embedded
    when extracted via inspect.getsource() to keep the function self-contained.

    Args:
        algorithm_metadata: Pre-resolved algorithm metadata dict containing:
            - name: Algorithm name (e.g., "sft", "osft")
            - metrics_file_patterns: List of glob patterns for metrics files
            - metrics_file_rank0: Filename for rank 0 metrics
            - config_file: Optional config file name (for OSFT)
        ckpt_output_dir: Directory where metrics files are written
        metrics_port: Port for HTTP metrics server

    Returns:
        Tuple of (apply_fn, handler_class) for testing purposes
    """
    import glob
    import http.server
    import json
    import math
    import os
    import subprocess
    import threading

    # Extract algorithm metadata (pre-resolved from centralized registry)
    algorithm = algorithm_metadata["name"]
    metrics_file_pattern = algorithm_metadata["metrics_file_pattern"]
    metrics_file_rank0 = algorithm_metadata["metrics_file_rank0"]

    # Track if termination message has been written (to avoid duplicates)
    _termination_message_written = False

    class TrainingHubMetricsHandler(http.server.BaseHTTPRequestHandler):
        """HTTP handler that reads JSONL metrics from Training Hub backends."""

        def do_GET(self):
            """Handle GET requests to expose metrics as JSON."""
            try:
                # Read latest metrics
                metrics = self._read_latest_metrics()
                # Transform to controller-compatible schema
                transformed = self._transform_schema(metrics)
            except Exception as e:
                print(
                    f"[Kubeflow] Failed to create progress metrics payload: {e}",
                    flush=True,
                )
                self.send_error(500)
            else:
                # Write termination message when training completes (100%)
                self._maybe_write_termination_message(transformed)
                # Serve JSON
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(transformed, indent=2).encode())

        def _maybe_write_termination_message(self, metrics):
            """Write final metrics to termination message for reliable progress reporting.

            This ensures the controller can get the final training state even if
            HTTP polling misses it. Only writes once when training reaches 100%.
            """
            nonlocal _termination_message_written
            if _termination_message_written:
                return

            progress = metrics.get("progressPercentage")
            if progress is not None and progress >= 100:
                # Write final metrics to termination message for reliable progress reporting
                try:
                    final_metrics = json.dumps(metrics)
                    with open("/dev/termination-log", "w") as f:
                        f.write(final_metrics)
                    _termination_message_written = True
                    print(
                        "[Kubeflow] Final metrics written to termination message",
                        flush=True,
                    )
                except (OSError, ValueError, TypeError) as e:
                    print(
                        f"[Kubeflow] Warning: Failed to write termination message: {e}. "
                        f"Controller will fall back to HTTP polling.",
                        flush=True,
                    )

        def _read_latest_metrics(self):
            """Read last line of JSONL file (most recent metrics from rank 0)."""
            # Return empty metrics if algorithm doesn't produce metrics files
            if metrics_file_pattern is None:
                return {}

            if algorithm == "sft":
                return self._read_sft_metrics()
            elif algorithm == "osft":
                return self._read_osft_metrics()
            elif algorithm == "lora_sft":
                return self._read_lora_sft_metrics()
            else:
                # Algorithm not yet supported for metrics reading
                return {}

        def _read_osft_metrics(self):
            """Read OSFT metrics from metrics file."""
            metrics_file = f"{ckpt_output_dir}/{metrics_file_rank0}"

            try:
                if not os.path.exists(metrics_file):
                    return {}

                # Read config from training_params.json (OSFT-specific)
                config = {}
                config_file_path = f"{ckpt_output_dir}/training_params.json"
                if os.path.exists(config_file_path):
                    try:
                        with open(config_file_path) as f:
                            config = json.load(f)
                    except Exception:
                        print(
                            "[Kubeflow] Warning: Failed to read OSFT config",
                            flush=True,
                        )

                # Read last line of metrics using tail
                try:
                    result = subprocess.run(
                        ["tail", "-n", "1", metrics_file],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    last_line = result.stdout.strip()
                except subprocess.CalledProcessError:
                    return {}

                if last_line:
                    metrics = json.loads(last_line)
                    if config:
                        metrics["_config"] = config
                    return metrics

            except FileNotFoundError:
                return {}
            except json.JSONDecodeError:
                print("[Kubeflow] Warning: Failed to parse OSFT metrics JSON", flush=True)
                return {}
            except Exception:
                print("[Kubeflow] Error reading OSFT metrics", flush=True)
                return {}

            return {}

        def _read_sft_metrics(self):
            """Read SFT metrics from metrics files."""
            # Find rank 0 metrics file
            pattern = f"{ckpt_output_dir}/{metrics_file_pattern}"
            files = glob.glob(pattern)

            if not files:
                return {}

            # Prefer rank 0 file
            rank_0_files = [f for f in files if metrics_file_rank0 in f]
            metrics_file = rank_0_files[0] if rank_0_files else files[0]

            try:
                if not os.path.exists(metrics_file):
                    return {}

                # Read first line for config
                config = {}
                try:
                    with open(metrics_file) as f:
                        first_line = f.readline().strip()
                        if first_line:
                            config = json.loads(first_line)
                except Exception:
                    print(
                        "[Kubeflow] Warning: Failed to read SFT config",
                        flush=True,
                    )

                # Read last line for latest metrics using tail
                try:
                    result = subprocess.run(
                        ["tail", "-n", "1", metrics_file],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    last_line = result.stdout.strip()
                except subprocess.CalledProcessError:
                    return {}

                if last_line:
                    metrics = json.loads(last_line)
                    if config:
                        metrics["_config"] = config
                    return metrics

            except FileNotFoundError:
                return {}
            except json.JSONDecodeError:
                print("[Kubeflow] Warning: Failed to parse SFT metrics JSON", flush=True)
                return {}
            except Exception:
                print("[Kubeflow] Error reading SFT metrics", flush=True)
                return {}

            return {}

        def _read_lora_sft_metrics(self):
            """Read LoRA SFT metrics from training_metrics.jsonl.

            LoRA writes a single metrics file (no per-rank wildcard).
            Each line is a JSON object with step, epoch, loss, learning_rate.
            """
            metrics_file = f"{ckpt_output_dir}/{metrics_file_rank0}"

            try:
                if not os.path.exists(metrics_file):
                    return {}

                # Read last line of metrics using tail
                try:
                    result = subprocess.run(
                        ["tail", "-n", "1", metrics_file],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    last_line = result.stdout.strip()
                except subprocess.CalledProcessError:
                    return {}

                if last_line:
                    return json.loads(last_line)

            except json.JSONDecodeError:
                print("[Kubeflow] Warning: Failed to parse LoRA metrics JSON", flush=True)
                return {}
            except Exception:
                print("[Kubeflow] Error reading LoRA metrics", flush=True)
                return {}

            return {}

        def _transform_schema(self, metrics):
            """Transform backend schema to controller-compatible progress format."""
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

            # Use algorithm parameter to determine transformation
            if algorithm == "sft":
                return self._transform_sft(metrics)
            elif algorithm == "osft":
                return self._transform_osft(metrics)
            elif algorithm == "lora_sft":
                return self._transform_lora_sft(metrics)
            else:
                return {}

        def _transform_osft(self, metrics):
            """Transform OSFT schema to controller-compatible format."""
            step = metrics.get("step", 0)
            epoch = metrics.get("epoch", 0)
            steps_per_epoch = metrics.get("steps_per_epoch", 0)

            config = metrics.get("_config", {})
            configured_max_epochs = config.get("max_epochs")

            total_epochs = configured_max_epochs or metrics.get("total_epochs", 1)

            step_total = steps_per_epoch * total_epochs if steps_per_epoch > 0 else 0

            current_step_absolute = step
            progress = (current_step_absolute / step_total * 100) if step_total > 0 else 0
            percent_int = int(round(progress))
            if current_step_absolute < step_total:
                percent_int = min(99, percent_int)
            else:
                percent_int = min(100, percent_int)

            time_per_batch = metrics.get("time_per_batch", 0)
            remaining_steps = step_total - current_step_absolute

            if percent_int >= 100 or remaining_steps <= 0:
                estimated_remaining_sec = 0
            else:
                estimated_remaining_sec = (
                    int(remaining_steps * time_per_batch) if time_per_batch > 0 else None
                )

            loss_val = metrics.get("loss", 0)
            lr_val = metrics.get("lr")
            grad_norm_val = metrics.get("grad_norm", 0)
            samples_per_second = metrics.get("samples_per_second")
            val_loss_val = metrics.get("val_loss")

            return {
                "progressPercentage": percent_int,
                "estimatedRemainingSeconds": estimated_remaining_sec,
                "currentStep": current_step_absolute,
                "totalSteps": step_total,
                "currentEpoch": epoch + 1,
                "totalEpochs": total_epochs,
                "trainMetrics": {
                    "loss": f"{loss_val:.4f}" if loss_val is not None else None,
                    "learning_rate": f"{lr_val:.6f}" if lr_val is not None else None,
                    "grad_norm": f"{grad_norm_val:.4f}" if grad_norm_val is not None else None,
                    "throughput": (
                        f"{samples_per_second:.2f}" if samples_per_second is not None else None
                    ),
                },
                "evalMetrics": {
                    "eval_loss": f"{val_loss_val:.4f}" if val_loss_val is not None else None,
                },
            }

        def _transform_sft(self, metrics):
            """Transform SFT schema to controller-compatible format."""
            step = metrics.get("step", 0)
            epoch = metrics.get("epoch", 0)
            num_epoch_steps = metrics.get("num_epoch_steps") or metrics.get("num_batches", 0)
            total_samples = metrics.get("total_samples", 0)

            config = metrics.get("_config", {})
            configured_num_epochs = config.get("num_epochs")

            if not num_epoch_steps:
                num_epoch_steps = config.get("num_batches", 0)

            current_epoch = epoch + 1
            current_step_absolute = step
            samples_seen = metrics.get("samples_seen", 0)

            if configured_num_epochs:
                estimated_total_epochs = configured_num_epochs
            elif num_epoch_steps > 0 and samples_seen > 0:
                estimated_progress_through_epochs = (
                    samples_seen / (num_epoch_steps * total_samples / num_epoch_steps)
                    if total_samples > 0
                    else 0
                )
                if estimated_progress_through_epochs > current_epoch:
                    estimated_total_epochs = max(2, int(estimated_progress_through_epochs) + 1)
                else:
                    estimated_total_epochs = current_epoch
            else:
                estimated_total_epochs = (
                    max(2, current_epoch) if current_step_absolute > 0 else current_epoch
                )

            if num_epoch_steps > 0:
                step_total = num_epoch_steps * estimated_total_epochs
            else:
                step_total = max(step, step + 10)

            progress = (current_step_absolute / step_total * 100) if step_total > 0 else 0
            percent_int = int(round(progress))
            if current_step_absolute < step_total:
                percent_int = min(99, percent_int)
            else:
                percent_int = min(100, percent_int)

            throughput = metrics.get("overall_throughput", 0)
            remaining_steps = step_total - current_step_absolute

            if percent_int >= 100 or remaining_steps <= 0:
                estimated_remaining_sec = 0
            else:
                estimated_remaining_sec = (
                    int(remaining_steps / throughput)
                    if throughput > 0 and remaining_steps > 0
                    else None
                )

            loss_val = metrics.get("avg_loss", 0)
            lr_val = metrics.get("lr")
            grad_norm_val = metrics.get("gradnorm")
            throughput_val = metrics.get("overall_throughput")

            return {
                "progressPercentage": percent_int,
                "estimatedRemainingSeconds": estimated_remaining_sec,
                "currentStep": current_step_absolute,
                "totalSteps": step_total,
                "currentEpoch": current_epoch,
                "totalEpochs": estimated_total_epochs,
                "trainMetrics": {
                    "loss": f"{loss_val:.4f}" if loss_val is not None else None,
                    "learning_rate": f"{lr_val:.6f}" if lr_val is not None else None,
                    "grad_norm": f"{grad_norm_val:.4f}" if grad_norm_val is not None else None,
                    "throughput": f"{throughput_val:.2f}" if throughput_val is not None else None,
                },
                "evalMetrics": {},
            }

        def _transform_lora_sft(self, metrics):
            """Transform LoRA SFT schema to controller-compatible format.

            LoRA metrics format (from training_metrics.jsonl):
                {"step": 1, "epoch": 0.015625, "loss": 4.2727, "learning_rate": 2e-6}

            max_steps is optionally present when the training_hub callback
            includes it (via state.max_steps). When absent, progressPercentage
            is reported as None (unknown) rather than 0.
            """
            step = metrics.get("step", 0)
            epoch = metrics.get("epoch", 0)
            max_steps = metrics.get("max_steps")

            # Calculate progress based on steps, clamping to 99 until final step.
            # When max_steps is unavailable, report None (unknown) instead of 0.
            if max_steps and max_steps > 0:
                progress = step / max_steps * 100
                percent_int = int(round(progress))
                percent_int = min(99, percent_int) if step < max_steps else min(100, percent_int)
            else:
                percent_int = None

            loss_val = metrics.get("loss")
            lr_val = metrics.get("learning_rate")
            grad_norm_val = metrics.get("grad_norm")

            return {
                "progressPercentage": percent_int,
                "estimatedRemainingSeconds": None,
                "currentStep": step,
                "totalSteps": max_steps if max_steps else None,
                "currentEpoch": max(1, math.ceil(epoch)),
                "totalEpochs": None,
                "trainMetrics": {
                    "loss": f"{loss_val:.4f}" if loss_val is not None else None,
                    "learning_rate": f"{lr_val:.6f}" if lr_val is not None else None,
                    "grad_norm": f"{grad_norm_val:.4f}" if grad_norm_val is not None else None,
                },
                "evalMetrics": {},
            }

        def log_message(self, format, *args):
            """Suppress default HTTP server logging."""
            pass

    def apply_progression_tracking():
        """Start HTTP server for metrics in background thread."""
        # Clean stale metrics files from previous runs
        # Since backends restart training from step 0 without checkpoint resumption,
        # old metrics files should be deleted to avoid showing stale/incorrect progress
        # Only the primary pod (rank 0) performs cleanup

        # Determine if this is the primary pod using standard environment variables
        # Precedence: JOB_COMPLETION_INDEX -> PET_NODE_RANK -> False (conservative)
        job_index = os.environ.get("JOB_COMPLETION_INDEX")
        if job_index is not None:
            is_primary_pod = job_index == "0"
        else:
            pet_rank = os.environ.get("PET_NODE_RANK")
            # Conservative default: if neither signal present, don't assume primary
            is_primary_pod = pet_rank == "0" if pet_rank is not None else False

        if is_primary_pod:
            # Skip cleanup if algorithm doesn't produce metrics files
            if metrics_file_pattern is None:
                print(
                    "[Kubeflow] Algorithm produces no metrics files, skipping cleanup", flush=True
                )
            else:
                try:
                    print("[Kubeflow] Primary pod cleaning stale metrics files", flush=True)

                    # Use metrics patterns from algorithm metadata
                    patterns = [metrics_file_pattern]

                    # Delete matching files
                    files_removed = 0
                    for pattern in patterns:
                        full_pattern = os.path.join(ckpt_output_dir, pattern)
                        for file_path in sorted(glob.glob(full_pattern)):
                            try:
                                os.remove(file_path)
                                files_removed += 1
                                filename = os.path.basename(file_path)
                                print(
                                    f"[Kubeflow] Removed stale metrics file: {filename}",
                                    flush=True,
                                )
                            except OSError as e:
                                filename = os.path.basename(file_path)
                                print(
                                    f"[Kubeflow] Warning: Could not remove {filename}: {e}",
                                    flush=True,
                                )

                    if files_removed > 0:
                        file_word = "files" if files_removed != 1 else "file"
                        print(
                            f"[Kubeflow] Cleaned {files_removed} stale metrics {file_word}",
                            flush=True,
                        )
                    else:
                        print("[Kubeflow] No stale metrics files found", flush=True)

                except OSError as e:
                    print(f"[Kubeflow] Warning: Metrics cleanup failed: {e}", flush=True)
        else:
            print("[Kubeflow] Non-primary pod skipping metrics cleanup", flush=True)

        # Start HTTP server for metrics
        try:
            server = http.server.HTTPServer(("0.0.0.0", metrics_port), TrainingHubMetricsHandler)

            # Run server in background thread
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()

            if metrics_file_pattern is None:
                msg = (
                    f"[Kubeflow] Metrics server started on port {metrics_port} for {algorithm} "
                    "(no metrics files - progress tracking unavailable)"
                )
                print(msg, flush=True)
            else:
                print(
                    f"[Kubeflow] Metrics server started on port {metrics_port} for {algorithm}",
                    flush=True,
                )

            return server
        except OSError as e:
            print(
                f"[Kubeflow] Warning: Failed to start metrics server on port "
                f"{metrics_port}: {e}. Training will continue without metrics server.",
                flush=True,
            )
            return None
        except Exception as e:
            print(
                f"[Kubeflow] Warning: Unexpected error starting metrics server: {e}. "
                f"Training will continue without metrics server.",
                flush=True,
            )
            return None

    return (apply_progression_tracking, TrainingHubMetricsHandler)
