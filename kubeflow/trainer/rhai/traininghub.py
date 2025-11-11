from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect
import os
import textwrap
from typing import Optional

from kubeflow_trainer_api import models

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
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
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: Optional[dict[str, str]] = None
    algorithm: Optional[TrainingHubAlgorithms] = None


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
    is_mpi = len(runtime.trainer.command) > 0 and runtime.trainer.command[0] == "mpirun"
    if packages_to_install:
        return k8s_utils.get_script_for_python_packages(
            packages_to_install,
            pip_index_urls,
            is_mpi,
        )
    return ""


def _render_algorithm_wrapper(algorithm_name: str, func_args: Optional[dict]) -> str:
    """Render a small Python script that calls training_hub.<algorithm>(**func_args)."""
    lines: list[str] = [
        "def training_func(func_args):",
        "    import os",
        f"    from training_hub import {algorithm_name}",
        "",
        "    _dp = (func_args or {}).get('data_path')",
        "    if _dp:",
        "        if os.path.isfile(_dp):",
        '            print(f"[PY] Data file found: {_dp}")',
        "        else:",
        '            print(f"[PY] Data file NOT found: {_dp}")',
        "",
        '    master_addr = os.environ.get("PET_MASTER_ADDR", "127.0.0.1")',
        '    master_port = os.environ.get("PET_MASTER_PORT", "29500")',
        '    node_rank = int(os.environ.get("PET_NODE_RANK", "0"))',
        '    rdzv_endpoint = f"{master_addr}:{master_port}"',
        "",
        "    args = dict(func_args or {})",
        "    args['node_rank'] = node_rank",
        "    args['rdzv_endpoint'] = rdzv_endpoint",
        f'    print("[PY] Launching {algorithm_name.upper()} training...")',
        "    try:",
        f"        result = {algorithm_name}(**args)",
        f'        print("[PY] {algorithm_name.upper()} training complete. Result=", result)',
        "    except ValueError as e:",
        '        print(f"Configuration error: {e}")',
        "    except Exception as e:",
        "        import traceback",
        '        print("[PY] Training failed with error:", e)',
        "        traceback.print_exc()",
        "",
        "    print('[PY] Training finished successfully.')",
        "",
    ]

    if func_args is None:
        call_line = "training_func({})"
    elif isinstance(func_args, dict):
        params_lines: list[str] = ["training_func({"]
        for key, value in func_args.items():
            params_lines.append(f"    {repr(key)}: {repr(value)},")
        params_lines.append("})")
        call_line = "\n".join(params_lines)
    else:
        call_line = f"training_func({func_args})"

    lines.append(call_line)
    return "\n".join(lines) + "\n"


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
        exec_script = _compose_exec_script(raw_code, "training_script.py")
        full_script = install_snippet + exec_script

        trainer_crd.command = ["bash", "-c"]
        trainer_crd.args = [full_script]
    else:
        # Secondary case: user provided function; embed their function and call with kwargs
        func_code, func_file = _render_user_func_code(trainer.func, trainer.func_args)
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
