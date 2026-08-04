"""Code generation utilities for TrainingHubTrainer script construction."""

from collections.abc import Callable
import inspect
import os
import textwrap

from kubeflow.trainer.types import types


def _render_dict_literal(func_args: dict, indent: str) -> str:
    """Render a dict literal with repr-safe values only."""
    import ast

    lines: list[str] = []
    for key, value in func_args.items():
        if not isinstance(key, str):
            raise ValueError(f"func_args key {key!r} of type {type(key).__name__} must be str")
        # Render repr exactly once and validate that rendering — a stateful __repr__
        # must not be able to return a safe literal during validation and an
        # executable expression during rendering (CWE-94).
        rendered_value = repr(value)
        try:
            if ast.literal_eval(rendered_value) != value:
                raise ValueError(
                    f"func_args[{key!r}] value type {type(value).__name__} is not repr-safe"
                )
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"func_args[{key!r}] value type {type(value).__name__} is not repr-safe"
            ) from e
        lines.append(f"{indent}{key!r}: {rendered_value},")
    return "\n".join(lines)


def _render_algorithm_wrapper(algorithm_metadata: dict, func_args: dict | None) -> str:
    """Render a small Python script that calls training_hub.<algorithm>(**func_args).

    Includes termination message writing after training completes (on_train_end equivalent)
    to ensure controller captures final metrics even if HTTP server becomes unreachable.

    Args:
        algorithm_metadata: Pre-resolved algorithm metadata dict from
            get_algorithm_pod_metadata() containing:
            - name: Algorithm name
            - metrics_file_rank0: Filename for rank 0 metrics
        func_args: Arguments to pass to the training function
    """
    # Extract values from pre-validated metadata
    algorithm_name = algorithm_metadata["name"]
    if not isinstance(algorithm_name, str) or not algorithm_name.isidentifier():
        raise ValueError(f"Invalid algorithm name for code generation: {algorithm_name!r}")
    metrics_file_rank0 = algorithm_metadata["metrics_file_rank0"]

    base_script = textwrap.dedent("""
    def _write_termination_message(ckpt_output_dir, algorithm, metrics_file_rank0):
        \"\"\"Write final metrics to /dev/termination-log for reliable capture.

        Kubernetes reads /dev/termination-log after container exit, providing
        a reliable fallback mechanism for metrics capture that doesn't depend
        on pod lifecycle timing or network availability.
        \"\"\"
        import json
        import os

        # Skip termination message for algorithms without metrics files
        if metrics_file_rank0 is None:
            print(
                "[Kubeflow] Algorithm produces no metrics files - skipping termination message",
                flush=True,
            )
            return

        try:
            # If we reach here, metrics ARE expected for this algorithm
            metrics_file = os.path.join(ckpt_output_dir, metrics_file_rank0)

            # Check if expected metrics file exists
            if not os.path.exists(metrics_file):
                print(
                    f"[Kubeflow] WARNING: Expected metrics file not found: {{metrics_file_rank0}}",
                    flush=True,
                )
                print(
                    "[Kubeflow] Training may have failed to write metrics or terminated early",
                    flush=True,
                )
                return

            # Read final metrics from rank 0 file (keep only the last line to
            # avoid loading the whole JSONL history into memory)
            metrics = None
            with open(metrics_file, 'r') as f:
                last_line = None
                for line in f:
                    last_line = line
                if last_line is not None:
                    metrics = json.loads(last_line)
                else:
                    print(
                        f"[Kubeflow] WARNING: Metrics file is empty: {{metrics_file_rank0}}",
                        flush=True,
                    )
                    return

            if metrics:
                # Build final progress JSON (matches controller's AnnotationStatus struct)
                # Try different metric names based on algorithm
                loss_value = (
                    metrics.get("loss")
                    or metrics.get("train_loss")
                    or metrics.get("avg_loss")
                    or 0
                )
                final_progress = {{
                    "progressPercentage": 100,
                    "trainMetrics": {{
                        "loss": str(loss_value),
                    }},
                    "evalMetrics": {{}},
                }}

                with open("/dev/termination-log", 'w') as f:
                    json.dump(final_progress, f)
                print("[Kubeflow] Termination message written with final metrics", flush=True)
            else:
                print(
                    "[Kubeflow] WARNING: Metrics file read but could not parse final metrics",
                    flush=True,
                )


        except PermissionError:
            print("[Kubeflow] Cannot write termination message (not in container)", flush=True)
        except Exception as e:
            print(f"[Kubeflow] Warning: Failed to write termination message: {{e}}", flush=True)

    def training_func(func_args):
        import os
        from training_hub import {algo}

        _dp = (func_args or {{}}).get('data_path')
        if _dp:
            print("[PY] Data file found: {{}}".format(_dp), flush=True)
        else:
            print("[PY] Data file NOT found: {{}}".format(_dp), flush=True)

        args = dict(func_args or {{}})
        ckpt_output_dir = args.get('ckpt_output_dir', '/tmp/checkpoints')
        algorithm = '{algo}'
        metrics_file_rank0 = {metrics_file_rank0!r}

        print("[PY] Launching {algo_upper} training...", flush=True)
        try:
            result = {algo}(**args)
            print("[PY] {algo_upper} training complete. Result=", result, flush=True)

            # Write termination message (on_train_end equivalent)
            # Ensures controller captures final metrics even if HTTP server unreachable
            _write_termination_message(ckpt_output_dir, algorithm, metrics_file_rank0)

        except ValueError as e:
            print("Configuration error:", e, flush=True)
            # Propagate configuration errors so the pod fails
            raise
        except Exception as e:
            import traceback
            print("[PY] Training failed with error:", e, flush=True)
            traceback.print_exc()
            # Propagate errors so the pod fails
            raise

    """).format(
        algo=algorithm_name,
        algo_upper=algorithm_name.upper(),
        metrics_file_rank0=metrics_file_rank0,
    )

    # Wrap the call in if __name__ == '__main__' to prevent re-execution
    # when algorithms use multiprocessing.spawn (e.g. ART's lora_grpo).
    # Harmless for algorithms that don't use spawn (SFT, OSFT, LoRA SFT).
    if func_args is None:
        call_line = "if __name__ == '__main__':\n    training_func({})\n"
    elif isinstance(func_args, dict):
        params_body = _render_dict_literal(func_args, "        ")
        call_line = (
            "if __name__ == '__main__':\n    training_func({\n"
            + (params_body + "\n" if params_body else "")
            + "    })\n"
        )
    else:
        raise ValueError("func_args must be a dict or None")

    return base_script + call_line


def _render_user_func_code(func: Callable, func_args: dict | None) -> tuple[str, str]:
    """Return (func_code, func_file_basename) embedding the user function and call."""
    if not callable(func):
        raise ValueError(f"Training function must be callable, got function type: {type(func)}")

    func_code = inspect.getsource(func)
    func_code = textwrap.dedent(func_code)

    if func_args is None:
        call_block = f"{func.__name__}()"
    elif isinstance(func_args, dict):
        params_body = _render_dict_literal(func_args, "    ")
        params_lines: list[str] = [f"{func.__name__}(**{{"]
        if params_body:
            params_lines.append(params_body)
        params_lines.append("})")
        call_block = "\n".join(params_lines)
    else:
        raise ValueError("func_args must be a dict or None")

    func_code = f"{func_code}\n{call_block}\n"
    func_file = os.path.basename(inspect.getfile(func))
    return func_code, func_file


def _get_command_from_runtime(
    runtime: types.Runtime,
    func_code: str,
    func_file: str,
    install_snippet: str,
) -> list[str]:
    """Build command using runtime's command template (matches CustomTrainer pattern).

    Args:
        runtime: Runtime configuration with command template
        func_code: The training function code to execute
        func_file: The filename to write the code to
        install_snippet: Package installation script to prepend

    Returns:
        Command list ready for trainer_crd.command/args

    Note:
        This matches CustomTrainer's approach of using runtime.trainer.command directly,
        ensuring consistent behavior across all trainer types.
    """
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            # Format the runtime's command template with our code
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_snippet:
                exec_script = install_snippet + exec_script
            command.append(exec_script)
        else:
            command.append(c)
    return command
