# RHAI Layer Design

## Context

The Kubeflow SDK upstream provides generic trainer types and backends (Kubernetes,
Container, LocalProcess). Red Hat OpenShift AI (RHOAI) needs HuggingFace
Transformers integration, Training Hub support, and OpenShift-specific behavior
without forking the upstream trainer core.

## Decision

We chose a separate `kubeflow/trainer/rhai/` layer on top of upstream trainer
types rather than modifying upstream modules in place.

## Rationale

- **Upstream compatibility**: Keeps merge cost low when rebasing on
  `opendatahub-io/kubeflow-sdk` / Kubeflow Trainer changes.
- **Optional dependency surface**: RHAI-specific code (Transformers, Training Hub)
  stays isolated; users who only need generic trainers do not import RHAI modules.
- **Clear ownership**: RHOAI extensions live in one directory with matching tests
  (`transformers_test.py`, `traininghub_test.py`, `utils_test.py`).

## Invariants

- Public RHAI types exported via `kubeflow/trainer/rhai/__init__.py` must remain
  stable; new params use keyword-only arguments.
- RHAI trainers must not bypass backend abstraction — execution still flows through
  `TrainerClient` and the selected backend.
- Monkey patching in `TransformersTrainer` callbacks assumes the training function
  runs inside a HuggingFace Trainer context with instrumentation enabled; do not
  invoke patched callbacks outside that lifecycle. See
  [transformers-trainer-instrumentation.md](transformers-trainer-instrumentation.md)
  for preconditions.

## Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Separate `rhai/` layer (chosen) | Clean upstream boundary, optional import | Some duplication of trainer wiring |
| Inline in upstream types | Fewer files | Harder rebases, blurs OSS vs RHOAI scope |

## Related code

- Reference implementation: `kubeflow/trainer/rhai/transformers.py`
- Tests: `kubeflow/trainer/rhai/transformers_test.py`
- Backend integration: [backend-abstraction.md](backend-abstraction.md),
  [train-job-lifecycle.md](train-job-lifecycle.md)
- Checkpoint monkey-patch detail: [transformers-trainer-instrumentation.md](transformers-trainer-instrumentation.md)
