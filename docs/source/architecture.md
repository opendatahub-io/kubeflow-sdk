# Architecture Overview

This document describes the high-level architecture of the Kubeflow SDK, its subsystems, key classes, and data flow.

## Subsystems

The SDK is organized into five subsystems under the `kubeflow/` namespace package:

```
kubeflow/
├── common/      Shared utilities (logging, types) used by all subsystems
├── trainer/     ML training job management
├── optimizer/   Hyperparameter optimization
├── hub/         Model registry client
└── spark/       Spark Connect session management
```

### Subsystem Relationships

```
┌────────────────────────────────────────────────────────────────┐
│                         User Code                              │
└──────┬──────────────┬───────────────┬──────────────┬──────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
┌─────────────┐ ┌───────────┐ ┌─────────────┐ ┌───────────┐
│TrainerClient│ │Optimizer- │ │ModelRegistry│ │SparkClient│
│             │ │Client     │ │Client       │ │           │
└──────┬──────┘ └─────┬─────┘ └──────┬──────┘ └─────┬─────┘
       │              │               │              │
       ▼              ▼               │              ▼
┌─────────────────────────────┐      │    ┌───────────────────┐
│        Backend Layer        │      │    │  Backend Layer     │
│ (Kubernetes, Container,     │      │    │  (Kubernetes)      │
│  LocalProcess)              │      │    │                    │
└──────┬──────────────────────┘      │    └────────┬──────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
┌────────────────────────────────────────────────────────────────┐
│              kubeflow.common (logging, types)                   │
└────────────────────────────────────────────────────────────────┘
```

The **Optimizer** reuses Trainer's `TrainJobTemplate` type to define trial workloads, coupling optimization runs to training job definitions.

---

## kubeflow.common

Shared infrastructure consumed by all subsystems.

| Module | Responsibility |
|--------|---------------|
| `common/types.py` | `KubernetesBackendConfig` — Pydantic model for Kubernetes connection settings (namespace, context, config file) |
| `common/logging.py` | Structured logger factory (`get_logger`) used across all clients |

---

## kubeflow.trainer

Manages ML training job lifecycle across multiple execution backends.

### Entry Point

`TrainerClient` — instantiated with a backend config, dispatches to the appropriate backend.

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `TrainerClient` | `trainer/api/trainer_client.py` | Main user-facing API: `train()`, `get_job()`, `list_jobs()`, `delete_job()` |
| `KubernetesBackend` | `trainer/backends/kubernetes/backend.py` | Submits TrainJob CRs to Kubernetes |
| `ContainerBackend` | `trainer/backends/container/backend.py` | Runs training in Docker/Podman containers locally |
| `LocalProcessBackend` | `trainer/backends/localprocess/backend.py` | Runs training as a subprocess for quick iteration |
| `BuiltinTrainer` | `trainer/types/types.py` | Configuration for built-in trainers (HuggingFace, TorchTune) |
| `CustomTrainer` | `trainer/types/types.py` | User-defined trainer with custom containers |
| `TrainJobTemplate` | `trainer/types/types.py` | Reusable job template (also used by Optimizer) |

### Data Flow

```
User calls TrainerClient.train(trainer=..., ...)
    │
    ├─ Resolves backend from backend_config type
    │
    ├─ Backend.train() builds job spec from trainer config
    │   ├─ BuiltinTrainer → resolves runtime image + model/dataset initializers
    │   └─ CustomTrainer  → uses user-provided container spec
    │
    └─ Backend submits job:
        ├─ Kubernetes → creates TrainJob CR via k8s API
        ├─ Container  → runs docker/podman container
        └─ LocalProcess → spawns subprocess
```

### Package Layout

```
trainer/
├── api/              TrainerClient (user interface)
├── backends/
│   ├── kubernetes/   K8s backend (TrainJob CR lifecycle)
│   ├── container/    Docker/Podman backend + adapters
│   └── localprocess/ Subprocess backend
├── constants/        Default paths (MODEL_PATH, DATASET_PATH)
├── options/          Backend-specific options (Name, Labels, PodTemplateOverrides)
├── rhai/             Red Hat AI trainer implementations (Transformers, TrainingHub)
└── types/            Pydantic models for jobs, trainers, initializers
```

---

## kubeflow.optimizer

Automates hyperparameter search by creating optimization experiments that spawn trainer trials.

### Entry Point

`OptimizerClient` — configured with a `KubernetesBackendConfig`.

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `OptimizerClient` | `optimizer/api/optimizer_client.py` | `optimize()`, `get_job()`, `list_jobs()`, `delete_job()` |
| `KubernetesBackend` | `optimizer/backends/kubernetes/backend.py` | Manages Experiment CRs on Kubernetes |
| `OptimizationJob` | `optimizer/types/optimization_types.py` | Job definition with objective, search space, trial template |
| `GridSearch` / `RandomSearch` | `optimizer/types/algorithm_types.py` | Search algorithm configurations |
| `Search` | `optimizer/types/search_types.py` | Search space parameter definitions |

### Data Flow

```
User calls OptimizerClient.optimize(optimization_job=...)
    │
    ├─ Builds Experiment CR from OptimizationJob spec
    │   ├─ Objective (metric to optimize)
    │   ├─ Search algorithm (Grid/Random)
    │   ├─ Search space (parameter ranges)
    │   └─ TrainJobTemplate (trial workload definition — reused from trainer)
    │
    └─ Kubernetes backend creates/watches Experiment CR
        └─ Controller spawns trial TrainJobs, collects metrics
```

---

## kubeflow.hub

Thin client over the Kubeflow Model Registry for versioning and retrieving trained models.

### Entry Point

`ModelRegistryClient` — connects to a model registry server via REST.

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `ModelRegistryClient` | `hub/api/model_registry_client.py` | `register_model()`, `get_model()`, `list_models()`, `list_versions()` |

### Data Flow

```
User calls ModelRegistryClient.register_model(name=..., uri=...)
    │
    └─ REST calls to model-registry server
        ├─ Creates/updates RegisteredModel
        ├─ Creates ModelVersion
        └─ Creates ModelArtifact with storage URI
```

---

## kubeflow.spark

Manages Apache Spark Connect sessions on Kubernetes for distributed data processing.

### Entry Point

`SparkClient` — configured with `KubernetesBackendConfig`.

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `SparkClient` | `spark/api/spark_client.py` | `connect()`, `disconnect()`, `list()` |
| `KubernetesBackend` | `spark/backends/kubernetes/backend.py` | Manages SparkApplication CRs |
| `Driver` / `Executor` | `spark/types/types.py` | Resource configuration for Spark pods |
| `SparkConnectInfo` | `spark/types/types.py` | Connection state information |

### Data Flow

```
User calls SparkClient.connect(...)
    │
    ├─ Backend creates SparkApplication CR on Kubernetes
    │
    ├─ Waits for Spark Connect endpoint to become ready
    │
    └─ Returns PySpark SparkSession connected to the cluster
```

---

## Shared Patterns

1. **Backend abstraction**: Each subsystem's client delegates to a backend that implements submission, polling, and deletion. This enables multiple execution environments from a single API.

2. **Pydantic models**: All configuration and type definitions use Pydantic v2 `BaseModel` for validation, serialization, and schema generation.

3. **Structured logging**: All clients use `kubeflow.common.logging.get_logger()` for consistent log output.

4. **Backend config dispatch**: Client constructors accept a union of backend config types and route to the matching backend implementation at runtime.

5. **Iterator-based streaming**: Job status and logs are returned as `Iterator` types for memory-efficient streaming of long-running operations.
