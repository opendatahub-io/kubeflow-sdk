# TransformersTrainer Instrumentation and Monkey Patching

See also: [rhai-layer.md](rhai-layer.md) for why RHAI code lives in a separate package.

## Context

`TransformersTrainer` (`kubeflow/trainer/rhai/transformers.py`) wraps a user training
function for execution on a Kubernetes `TrainJob`. Two cross-cutting features inject
code into the training script that runs **inside the cluster pod**:

1. **Progression tracking** — metrics callback wrapping user function
2. **Checkpoint instrumentation** — monkey-patches `transformers.Trainer` at runtime in
   the worker process

This document covers checkpoint instrumentation and its monkey-patch preconditions.

## How instrumentation reaches the pod

### CR build time (SDK, client machine or CI)

`get_trainer_cr_from_transformers_trainer()`:

1. Serializes `trainer.func` via `inspect.getsource`
2. Optionally wraps with progression instrumentation (`enable_progression_tracking`)
3. Prepends/appends checkpoint code from `_build_checkpoint_code()` when JIT or periodic
   checkpointing is enabled
4. Embeds wrapped code into `TrainerV1alpha1Trainer.command` using runtime command
   templates (`{func_code}`, `{func_file}` placeholders)

Checkpoint injection uses `get_jit_checkpoint_injection_code()` which embeds a copy of
`_create_checkpoint_instrumentation` (via `inspect.getsource`) and a serialized
`checkpoint_config` dict into the pod script header.

### Pod runtime (training worker)

At the start of the injected header (generated code):

```python
_, _, apply_checkpointing, upload_final_model_to_cloud = _create_checkpoint_instrumentation(checkpoint_config)
apply_checkpointing()
```

`apply_checkpointing()` (defined inside `_create_checkpoint_instrumentation`):

1. Imports `transformers.Trainer` as `_TransformersTrainer`
2. Replaces `_TransformersTrainer.__init__` with `_patched_trainer_init`
3. The patch runs when the user's code constructs a HuggingFace `Trainer`

## What `_patched_trainer_init` does

When a HuggingFace `Trainer` is constructed:

1. Reads `enable_jit` from `checkpoint_config`
2. Resolves `TrainingArguments` from `kwargs["args"]` or `args[1]`
3. **Validates / mutates `TrainingArguments`:**
   - Raises if `save_only_model=True`
   - Applies `output_dir`, `save_strategy`, `save_steps`, `save_total_limit` from config
   - Raises if `save_on_each_node=True` with S3 `cloud_remote_storage_uri`
4. If `enable_jit`: appends module-level `JITCheckpointCallback` to `kwargs["callbacks"]`
5. Calls original `__init__`
6. If `enable_jit`: sets `_jit_checkpoint_callback._trainer_ref = self`
7. Replaces `self.train` with `_patched_train` that:
   - Runs distributed `dist.barrier()` when `dist.is_initialized()`
   - Auto-sets `resume_from_checkpoint` to latest valid checkpoint in `output_dir`
   - Delegates to original `train()`

## Preconditions (must hold for safe behavior)

| # | Precondition | Source | If violated |
|---|--------------|--------|-------------|
| 1 | Checkpoint injection enabled (`enable_jit_checkpoint` or `periodic_checkpoint_config`) | `_build_checkpoint_code()` | No patch code emitted |
| 2 | User code constructs `transformers.Trainer` after header runs | Injection order in `func_code` | Patch never applied |
| 3 | `TrainingArguments` passed to `Trainer` | `_patched_trainer_init` | Config not applied; JIT callback may not inject |
| 4 | `save_only_model=False` when Kubeflow checkpointing active | explicit `ValueError` | Training fails at init |
| 5 | Not `save_on_each_node=True` with S3 remote URI | explicit `ValueError` | Training fails at init |
| 6 | `enable_jit=True` in injected config for callback injection | `checkpoint_config` | Periodic-only path may apply args without JIT callback |
| 7 | Distributed training: `torch.distributed` initialized before `_patched_train` barrier | `_patched_train` | `RuntimeError` with barrier failure message |
| 8 | Cloud upload: `LOCAL_RANK` env `0` for final upload path | `upload_final_model_to_cloud` | Non-zero ranks skip upload |
| 9 | Primary pod semantics for rank detection elsewhere | `rhai_utils.is_primary_pod()` | Uses `JOB_COMPLETION_INDEX` or `PET_NODE_RANK` |

## Invariants

- Monkey patch is **process-global**: `_TransformersTrainer.__init__` is replaced for all
  `Trainer` instances in that Python process after `apply_checkpointing()` runs.
- `JITCheckpointCallback` is a **module-level singleton** in the injected closure
  (`_jit_checkpoint_callback`); one instance shared across trainers in the process.
- Do not call patched callback methods (`on_save`, upload worker) outside a live HF
  `Trainer` lifecycle — `_trainer_ref` may be unset (warning logged).
- Checkpoint incomplete markers: only global rank 0 deletes incomplete checkpoint dirs
  (`_find_latest_checkpoint`).
- After JIT upload + graceful shutdown path, instrumentation may `sys.exit(0)` to prevent
  duplicate `save_model()` during SIGTERM handling.

## Injection triggers

`_build_checkpoint_code()` returns empty strings when **both**:

- `trainer.enable_jit_checkpoint` is false, and
- `trainer.periodic_checkpoint_config` is unset

If JIT enabled without periodic config, defaults to `PeriodicCheckpointConfig()`.

## Decision

Checkpoint and progression features use **runtime monkey-patching of
`transformers.Trainer.__init__`** (and `Trainer.train` for checkpointing) rather than
requiring users to pass `JITCheckpointCallback` explicitly or subclass `Trainer`.

`TransformersTrainer` serializes the user's training function into the pod script via
`inspect.getsource`. Before user code runs, injected headers call `apply_checkpointing()`
and (when enabled) `apply_progression_tracking()`, which replace `Trainer.__init__` in that
worker process. User notebooks and examples keep standard HuggingFace `Trainer(...)` /
`trl.SFTTrainer(...)` call sites unchanged.

| Approach | Chosen? | Why / why not |
|----------|---------|---------------|
| Monkey-patch `Trainer.__init__` at pod startup | **Yes** | Transparent UX; works with serialized user source; mutates `TrainingArguments` and injects callbacks before user `Trainer` is constructed |
| User passes `JITCheckpointCallback` explicitly | No | Breaks "no code changes" goal; easy to omit in examples; callback class not in user's serialized imports |
| Subclass `KubeflowTrainer` | No | User and third-party code (TRL, libraries) must adopt subclass; conflicts with `CustomTrainer`-style function serialization |
| Patch only at SDK / CR build time | No | Checkpointing and SIGTERM handling must run **inside the pod process** where `Trainer` is constructed |

Progression tracking uses the same pattern (patch `Trainer.__init__` to add
`KubeflowProgressCallback` after the original init). Checkpoint patching is richer: it runs
**before** `__init__` to validate and apply `TrainingArguments`, inject the JIT callback, then
wraps `train()` for distributed barrier sync and auto-resume.

## Rationale

### Product goal: platform features without editing user training code

RHOAI `TransformersTrainer` targets HuggingFace / TRL users who already have a training
function built around `transformers.Trainer`. The intended UX (documented in
[Trainer v2 CRD Ecosystem §3.4 / §6.4](https://redhat.atlassian.net/wiki/spaces/RHODS/pages/418452915))
is:

- **Progression tracking** — real-time metrics for the Dashboard / controller
- **JIT checkpointing** — save on SIGTERM (preemption, scale-down) and resume without restarting from scratch
- **Cloud storage** — PVC or S3 `output_dir` with optional async upload

All of this should work **without modifying the user's training code**. Requiring an explicit
`JITCheckpointCallback(...)` in every function defeats that goal and creates drift between
examples, docs, and customer notebooks.

### Why patching fits the serialization model

The SDK does not import the user's module on the client machine. It embeds:

1. Source of `_create_checkpoint_instrumentation` / `_create_progression_instrumentation`
   (via `inspect.getsource`)
2. A serialized `checkpoint_config` dict
3. The user's function body

At pod startup the header runs `apply_checkpointing()` **before** the user's `Trainer(...)`
line executes. Patching is the hook that connects platform config (`TransformersTrainer.output_dir`,
`enable_jit_checkpoint`, periodic save settings) to the HuggingFace object the user constructs
anyway.

An explicit callback would require users to import RHAI types inside the serialized function and
remember to append the callback — fragile for copy-pasted examples and incompatible with the
"wrap my existing function" onboarding story.

### Why not subclass `Trainer`

Subclassing would force:

```python
trainer = KubeflowTrainer(...)  # user must change this line
```

Many RHOAI users wrap **TRL** trainers (`SFTTrainer`, etc.) or library code that internally
constructs `transformers.Trainer`. A subclass does not compose with those call sites without
forking upstream examples. Monkey-patching the class actually constructed (`Trainer`) keeps
third-party patterns working as long as they inherit from HuggingFace `Trainer`.

### What the patch must do (beyond callback injection)

The patch is not only about adding `JITCheckpointCallback`. `_patched_trainer_init` also:

- Validates incompatible `TrainingArguments` (`save_only_model`, `save_on_each_node` + S3)
- Applies platform `output_dir`, `save_strategy`, `save_steps`, `save_total_limit` from
  `TransformersTrainer` config
- Wraps `train()` for distributed `dist.barrier()` before training and **auto-resume** from the
  latest complete checkpoint

Those behaviors need to run at `Trainer` construction and first `train()` call — a standalone
callback passed manually would not cover `TrainingArguments` mutation or `train()` wrapping
without additional user wiring.

### Trade-offs accepted

| Benefit | Cost |
|---------|------|
| Zero changes to user `Trainer(...)` call sites | **Process-global** mutation — all `Trainer` instances in the pod after `apply_checkpointing()` |
| Works with serialized functions and TRL patterns | **Init order** — user must construct `Trainer` after injected header runs (see preconditions) |
| Platform controls checkpoint policy centrally | **`transformers` upgrade risk** — depends on `Trainer.__init__` signature and callback injection via `kwargs["callbacks"]` |
| Single injected closure shares upload worker / JIT manager | **Module-level singleton** `_jit_checkpoint_callback` — one callback instance per process |

Mitigations in code today:

- Injection is opt-in via `enable_jit_checkpoint` / `periodic_checkpoint_config` / `output_dir`
- Explicit `ValueError`s for known incompatible `TrainingArguments` combinations
- Broad test coverage in `transformers_test.py` for injection, callback behavior, and patch guards

### Rejected alternatives

- **Document "add this callback yourself"** — places burden on data scientists; inconsistent with RHAI midstream value prop
- **Fork HuggingFace examples to use a Kubeflow trainer class** — high merge cost; poor upstream compatibility (see [rhai-layer.md](rhai-layer.md))
- **AST-rewrite user source to insert callback** — more brittle than a single class-level patch; harder to maintain across Python/HF versions

## Trade-offs (summary)

| Approach | Pros | Cons |
|----------|------|------|
| Runtime monkey-patch (**chosen**) | No user call-site changes; applies platform `TrainingArguments`; auto-resume + JIT on SIGTERM | Global side effects; HF version sensitivity; strict init order |
| Explicit callback in user code | No patch; explicit dependency | Users must remember it; breaks serialization/onboarding UX |
| Subclass `Trainer` | Explicit type | Does not compose with TRL/library code; requires editing every example |

## Related code

- `kubeflow/trainer/rhai/transformers.py` — `get_trainer_cr_from_transformers_trainer`,
  `_build_checkpoint_code`, `_create_checkpoint_instrumentation`, `apply_checkpointing`,
  `_create_progression_instrumentation`, `apply_progression_tracking`
- `kubeflow/trainer/rhai/transformers_test.py` — injection and patch behavior tests
- `kubeflow/trainer/rhai/utils.py` — `is_primary_pod()`, storage setup
- `kubeflow/trainer/rhai/constants.py` — checkpoint markers, URI schemes
- [rhai-layer.md](rhai-layer.md) — why RHAI instrumentation lives in a separate package
