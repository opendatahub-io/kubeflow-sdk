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

"""Serialize Training Hub callbacks for injection into training pod scripts."""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from typing import Any

_HUB_CALLBACKS_VAR = "_KUBEFLOW_HUB_CALLBACKS"
_TRAINING_HUB_API_NAMES = ("sft", "osft", "lora_sft", "lora_grpo")
_TRAINING_HUB_TYPE_NAMES = frozenset({"TrainingHubCallback", "TrainingHubContext"})
_SUPPORTED_HOOKS = frozenset(
    {
        "on_train_begin",
        "on_epoch_begin",
        "on_step_begin",
        "on_log",
        "on_evaluate",
        "on_save",
        "on_step_end",
        "on_epoch_end",
        "on_train_end",
    }
)


def _is_training_hub_callback_subclass(cls: type) -> bool:
    """Return whether cls subclasses TrainingHubCallback."""
    try:
        from training_hub import TrainingHubCallback
    except ImportError:
        return any(
            getattr(base, "__name__", None) == "TrainingHubCallback" for base in cls.__mro__[1:]
        )
    return issubclass(cls, TrainingHubCallback)


def _module_level_names(module_tree: ast.Module) -> set[str]:
    """Collect names defined at module scope."""
    names: set[str] = set()
    for node in module_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _names_defined_in_scope(scope: ast.AST) -> set[str]:
    """Collect names defined within a single function/method scope (non-recursive)."""
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _names_defined_in_class(class_tree: ast.ClassDef) -> set[str]:
    """Collect names defined at class level (not inside methods)."""
    names: set[str] = set()
    for node in class_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _unresolved_references_in_class(
    class_tree: ast.ClassDef, module_names: set[str], allowed: set[str]
) -> set[str]:
    """Find module-level references not resolved in their own method scope."""
    class_level = _names_defined_in_class(class_tree)
    unresolved: set[str] = set()

    for node in class_tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        method_locals = _names_defined_in_scope(node)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                name = child.id
                if name in module_names and name not in (
                    method_locals | class_level | allowed
                ):
                    unresolved.add(name)

    for node in class_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                name = child.id
                if name in module_names and name not in (class_level | allowed):
                    unresolved.add(name)

    return unresolved


def _validate_callback_is_self_contained(cls: type) -> None:
    """Reject callbacks that rely on module-level symbols outside the class."""
    module = inspect.getmodule(cls)
    if module is None:
        return

    try:
        module_source = inspect.getsource(module)
        class_source = inspect.getsource(cls)
    except (OSError, TypeError):
        return

    module_tree = ast.parse(module_source)
    parsed_class = ast.parse(class_source).body[0]
    if not isinstance(parsed_class, ast.ClassDef):
        return

    module_names = _module_level_names(module_tree)
    allowed = set(dir(builtins)) | _TRAINING_HUB_TYPE_NAMES | {"self", "cls", cls.__name__}
    external = _unresolved_references_in_class(parsed_class, module_names, allowed)
    if external:
        raise ValueError(
            f"Callback {cls.__name__!r} references module-level symbols "
            f"{sorted(external)!r} that cannot be serialized into the training pod. "
            "Move helpers inside the callback class or define them inline."
        )


def _validate_no_unknown_hooks(cls: type) -> None:
    """Reject callbacks that define hooks not in the unified contract."""
    user_hooks: set[str] = set()
    for base in cls.__mro__:
        if base is object or base.__name__ == "TrainingHubCallback":
            continue
        for name, value in vars(base).items():
            if name.startswith("on_") and callable(value):
                user_hooks.add(name)
    unknown = user_hooks - _SUPPORTED_HOOKS
    if unknown:
        raise ValueError(
            f"Callback {cls.__name__!r} defines unsupported hooks {sorted(unknown)!r}. "
            f"Supported hooks: {sorted(_SUPPORTED_HOOKS)}."
        )


def _validate_callback_has_noarg_init(cls: type) -> None:
    """Reject callbacks that require constructor arguments in the pod."""
    signature = inspect.signature(cls.__init__)
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f"Callback {cls.__name__!r} must use a no-argument constructor; "
                f"required parameter {parameter.name!r} is not supported for pod injection."
            )


def validate_callbacks(callbacks: list[Any] | None) -> None:
    """Validate callbacks before CRD generation.

    Args:
        callbacks: User-provided callback classes.

    Raises:
        TypeError: If callbacks is not a list or contains invalid entries.
        ValueError: If a callback class cannot be serialized with inspect.getsource.
    """
    if callbacks is None:
        return

    if not isinstance(callbacks, list):
        raise TypeError(f"callbacks must be a list, got {type(callbacks).__name__}")

    seen_class_names: set[str] = set()
    for callback in callbacks:
        if isinstance(callback, str):
            raise TypeError(
                "Each callback must be a TrainingHubCallback subclass (class object), "
                f"got {type(callback).__name__}"
            )
        if not isinstance(callback, type):
            raise TypeError(
                "Each callback must be a class, not an instance. "
                "Pass callback classes, e.g. callbacks=[MyCallback]."
            )
        cls = callback
        if not _is_training_hub_callback_subclass(cls):
            name = getattr(cls, "__name__", repr(callback))
            raise TypeError(
                f"Callback {name!r} must subclass TrainingHubCallback, got {cls.__name__!r}."
            )
        if cls.__name__ in seen_class_names:
            raise ValueError(
                f"Duplicate callback class name {cls.__name__!r}. "
                "Each callback class must have a unique name."
            )
        seen_class_names.add(cls.__name__)
        try:
            inspect.getsource(cls)
        except (OSError, TypeError) as exc:
            name = getattr(cls, "__name__", repr(callback))
            raise ValueError(
                f"Callback {name!r} must be defined in a regular Python source file "
                "so it can be serialized into the training pod. "
                "Define the callback at module level, not inside a function or REPL."
            ) from exc
        _validate_callback_is_self_contained(cls)
        _validate_no_unknown_hooks(cls)
        _validate_callback_has_noarg_init(cls)


def build_training_hub_callback_injection_code(callbacks: list[Any]) -> str:
    """Generate pod script preamble that defines and registers Training Hub callbacks.

    Serializes each callback class with inspect.getsource, instantiates them in the pod,
    and patches training_hub entrypoints so callbacks reach sft/osft/lora_sft/lora_grpo.

    Args:
        callbacks: Callback classes supplied on TrainingHubTrainer.

    Returns:
        Python source to prepend before generated training code.
    """
    if not callbacks:
        return ""

    callback_sources: list[str] = []
    class_names: list[str] = []
    for callback in callbacks:
        source = textwrap.dedent(inspect.getsource(callback))
        callback_sources.append(source)
        class_names.append(callback.__name__)

    all_callback_code = "\n\n".join(callback_sources)
    instantiation_lines = ", ".join(f"{name}()" for name in class_names)

    return f"""# =============================================================================
# Kubeflow SDK - Training Hub Callback Injection
# Generated by kubeflow.trainer.rhai.traininghub
# =============================================================================

import functools
from training_hub import TrainingHubCallback, TrainingHubContext
import training_hub as _kubeflow_training_hub

# --- User callback definitions ---
{all_callback_code}

{_HUB_CALLBACKS_VAR} = [{instantiation_lines}]
print(f"[Kubeflow] Prepared {{len({_HUB_CALLBACKS_VAR})}} Training Hub callback(s)", flush=True)


def _kubeflow_wrap_training_hub_api(_orig_fn, _extra_callbacks):
    @functools.wraps(_orig_fn)
    def _wrapped(*args, **kwargs):
        merged = list(kwargs.get("callbacks") or [])
        merged.extend(_extra_callbacks)
        kwargs["callbacks"] = merged
        return _orig_fn(*args, **kwargs)

    return _wrapped


if not getattr(_kubeflow_training_hub, "_kubeflow_callbacks_applied", False):
    setattr(_kubeflow_training_hub, "_kubeflow_callbacks_applied", True)
    for _api_name in {_TRAINING_HUB_API_NAMES!r}:
        _orig_api = getattr(_kubeflow_training_hub, _api_name, None)
        if _orig_api is not None:
            setattr(
                _kubeflow_training_hub,
                _api_name,
                _kubeflow_wrap_training_hub_api(_orig_api, {_HUB_CALLBACKS_VAR}),
            )

print("[Kubeflow] Training Hub callback injection configured", flush=True)
"""
