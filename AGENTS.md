# AI Agents Guide for OpenDataHub Kubeflow SDK

The Kubeflow SDK is a set of unified Pythonic APIs that let you run any AI workload at any scale –
without the need to learn Kubernetes. It provides simple and consistent APIs across the Kubeflow
ecosystem, enabling users to focus on building AI applications rather than managing complex
infrastructure.

This file is the short entry point for AI agents working in this repository. Keep it lean,
and follow linked documents for topic-specific detail.

## Who This Is For

- **AI agents**: Automate repository tasks with minimal context
- **Contributors**: Humans using AI assistants or working directly
- **Maintainers**: Ensure assistants follow project conventions and CI rules

## Start here

| Read this | When |
|-----------|------|
| [references/agent-behaviors.md](references/agent-behaviors.md) | Before any code change — agent constraints, scope limits, and context-awareness rules |
| [references/core-principles.md](references/core-principles.md) | Writing or reviewing Python — public APIs, style, tests, security, or docstrings |
| [references/development-workflow.md](references/development-workflow.md) | Full change checklist — pre-edit steps, validation sequence, and commit/PR detail beyond AGENTS.md |
| [references/common-changes.md](references/common-changes.md) | Implementing a common change — trainer types, algorithms, backend options, tests, or RHAI extensions |
| [docs/design/rhai-layer.md](docs/design/rhai-layer.md) | Why the RHAI layer exists and its invariants |
| [docs/design/backend-abstraction.md](docs/design/backend-abstraction.md) | How Kubernetes / Container / LocalProcess backends work |
| [docs/design/train-job-lifecycle.md](docs/design/train-job-lifecycle.md) | How `.train()` becomes a TrainJob CR on cluster |
| [docs/design/transformers-trainer-instrumentation.md](docs/design/transformers-trainer-instrumentation.md) | Monkey-patch preconditions for `TransformersTrainer` |

For code changes, read `agent-behaviors.md` first, then `core-principles.md` when editing Python, `common-changes.md` when following an existing pattern, and `development-workflow.md` before proposing a commit or PR. When changing architecture, review and update the relevant doc in `docs/design/`.

## Repository Map

| Path | Use for |
|------|---------|
| `.claude/` | Claude Code hooks |
| `.github/` | CI/CD workflows |
| `docs/` | Project documentation |
| `docs/design/` | Design rationale and architectural intent |
| `examples/` | Usage examples |
| `kubeflow/common/` | Shared utilities and types |
| `kubeflow/trainer/api/` | `TrainerClient` — main trainer entry point |
| `kubeflow/trainer/backends/kubernetes/` | Cluster execution backend |
| `kubeflow/trainer/backends/container/` | Local container backend (Docker/Podman) |
| `kubeflow/trainer/backends/localprocess/` | Subprocess backend for quick prototyping |
| `kubeflow/trainer/options/` | Backend config (`KubernetesOptions`, etc.) |
| `kubeflow/trainer/types/` | Trainer schemas (`TrainJob`, `CustomTrainer`, …) |
| `kubeflow/trainer/algorithms.py` | Algorithm registry |
| `kubeflow/trainer/rhai/` | RHOAI extensions (e.g. `TransformersTrainer`) |
| `kubeflow/spark/api/` | `SparkClient` |
| `kubeflow/optimizer/api/` | `OptimizerClient` |
| `kubeflow/optimizer/types/` | Optimization schemas (`OptimizationJob`, `Search`) |
| `kubeflow/hub/api/` | `ModelRegistryClient` |

## Environment & Tooling

- **Package manager**: `uv` (creates `.venv` automatically via targets)
- **Lint/format**: `ruff` (isort integrated)
- **Tests**: `pytest` with coverage
- **Build**: Hatchling (optional `uv build`)
- **Pre-commit**: Config provided and enforced in CI
- **Hooks**: Claude Code hooks in `.claude/settings.json` auto-format Python and block dangerous commands

## Commands

<!-- BEGIN: AGENT_COMMANDS -->

**Setup**:

```bash
make install-dev              # Install uv, create .venv, sync deps
```

**Verify (CI parity)**:

```bash
make verify                   # Runs ruff check --show-fixes and ruff format --check
```

**Testing**:

```bash
make test-python              # All unit tests + coverage (HTML by default)
make test-python report=xml   # XML coverage report
uv run pytest -q kubeflow/trainer/utils/utils_test.py                    # One file
uv run pytest -q kubeflow/trainer/utils/utils_test.py::test_name -k "pattern"  # One test
uv run coverage run -m pytest <path> && uv run coverage report          # Ad-hoc coverage
```

**Local lint/format**:

```bash
uv run ruff check --fix .     # Fix lint issues
uv run ruff format kubeflow   # Format code
```

**Single file Verification**:

```bash
uv run ruff check path/to/file.py
uv run mypy path/to/file.py
```

**Type checking**:

```bash
uv run mypy kubeflow          # Run type checker
uv run ty check kubeflow      # Alternative type checker (dev dependency)
```

**Pre-commit**:

```bash
uv run pre-commit install                    # Install hooks
uv run pre-commit run --all-files           # Run all hooks
```

<!-- END: AGENT_COMMANDS -->

## Key Conventions
- Preserve public API signatures; use keyword-only args for new params
- Type hints on all functions; line length 100; first-party import is `kubeflow`
- Unit tests in `*_test.py`; no network calls in unit tests

## Pattern References

Copy-modify from these examples; see [references/common-changes.md](references/common-changes.md) for file lists.

- New trainer type: follow the pattern in `kubeflow/trainer/rhai/transformers.py`
- New algorithm: see `kubeflow/trainer/algorithms.py` for registry layout
- New backend option: follow the pattern in `kubeflow/trainer/options/kubernetes.py`
- New test case: see `kubeflow/trainer/backends/kubernetes/backend_test.py` for parametrized tests
- RHOAI extensions: see `kubeflow/trainer/rhai/` for module layout

## Pull Requests
- Use Conventional Commits for PR titles; see [CONTRIBUTING.md](CONTRIBUTING.md#pull-request-title-conventions)
- Before proposing: run `make verify` and targeted tests for changed code
- Keep diffs minimal and scoped to the task; include rationale ("why") in the PR description
- Do not commit secrets or modify git config

Details: [references/development-workflow.md](references/development-workflow.md)