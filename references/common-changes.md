# Common Change Patterns

Copy structure from these references. See [core-principles.md](core-principles.md) for code quality and test style.

### Adding a new trainer type
- **Reference:** `kubeflow/trainer/rhai/transformers.py` (`TransformersTrainer`)
- **Also see:** `kubeflow/trainer/rhai/transformers_test.py`, `kubeflow/trainer/rhai/traininghub.py` (`TrainingHubTrainer`)
- **Files:** implementation module under `kubeflow/trainer/rhai/`, `kubeflow/trainer/rhai/__init__.py` exports, co-located `*_test.py`; export via `kubeflow/trainer/__init__.py` if public API

### Adding a new algorithm to the registry
- **Reference:** `kubeflow/trainer/algorithms.py` (`ALGORITHMS`, `AlgorithmSpec`)
- **Also see:** `kubeflow/trainer/algorithms_test.py`
- **Files:** `algorithms.py` (add `AlgorithmSpec` entry), `algorithms_test.py`

### Adding a new backend option
- **Reference:** `kubeflow/trainer/options/kubernetes.py` (e.g. `ContainerOverride`, option dataclasses)
- **Also see:** `kubeflow/trainer/options/common.py`, `kubeflow/trainer/options/kubernetes_test.py`
- **Files:** option module under `kubeflow/trainer/options/`, `kubeflow/trainer/options/__init__.py` exports, `*_test.py`, consuming backend under `kubeflow/trainer/backends/`

### Adding a new test case
- **Reference:** `kubeflow/trainer/backends/kubernetes/backend_test.py` (parametrized `TestCase` pattern)
- **Also see:** `kubeflow/trainer/test/common.py` (`TestCase`, `SUCCESS`/`FAILED` fixtures)
- **Files:** co-located `*_test.py` next to the module under test

### Modifying the RHOAI extensions
- **Reference:** `kubeflow/trainer/rhai/` (`transformers.py`, `traininghub.py`, `utils.py`)
- **Also see:** `kubeflow/trainer/rhai/transformers_test.py`, `kubeflow/trainer/rhai/traininghub_test.py`, `kubeflow/trainer/rhai/utils_test.py`
- **Files:** module under `kubeflow/trainer/rhai/`, matching `*_test.py`, `kubeflow/trainer/rhai/__init__.py` exports
