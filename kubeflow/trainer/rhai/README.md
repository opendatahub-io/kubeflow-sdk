# TransformersTrainer

Run HuggingFace Transformers and TRL training on under officially supported RHAITrainer.

## Quick Start

```python
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.rhai import TransformersTrainer

def train():
    from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
    
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./output", num_train_epochs=3),
        train_dataset=dataset,
    )
    trainer.train()

# Submit training job to Kubernetes
client = TrainerClient()
job_id = client.train(
    trainer=TransformersTrainer(
        func=train,
        num_nodes=2,
        resources_per_node={"gpu": 1, "cpu": 4, "memory": "16Gi"},
        packages_to_install=["transformers", "datasets", "torch"],
    )
)

# Wait for completion
client.wait_for_job_completion(job_id)
```

## Features

- ✅ **Zero code changes** - Standard Transformers/TRL code works as-is
- ✅ **Multi-GPU/Multi-node** - Distributed training support built-in
- ✅ **TRL support** - SFTTrainer, DPOTrainer, PPOTrainer, etc.
- ✅ **Package management** - Auto-install dependencies in training pods
- ✅ **Resource control** - Specify GPU, CPU, memory per node

## TRL Support

Works with TRL trainers (SFTTrainer, DPOTrainer, PPOTrainer) without code changes:

```python
def train_with_trl():
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir="./output", num_train_epochs=3),
        train_dataset=dataset,
    )
    trainer.train()

client.train(
    trainer=TransformersTrainer(
        func=train_with_trl,
        packages_to_install=["trl", "transformers", "torch"],
    )
)
```

## Configuration Options

```python
TransformersTrainer(
    func=train,                                  # Required: your training function
    func_args={"model_name": "gpt2"},           # Optional: function arguments
    packages_to_install=["transformers"],        # Auto-install dependencies
    pip_index_urls=["https://pypi.org/simple"], # Custom PyPI URLs
    num_nodes=2,                                 # Multi-node training
    resources_per_node={                         # Per-node resources
        "gpu": 1,
        "cpu": 4,
        "memory": "16Gi"
    },
    env={"CUDA_VISIBLE_DEVICES": "0"},          # Environment variables
)
```

---

## Feature (Optional): Real-Time Progress Tracking

TransformersTrainer includes **optional** progression tracking to monitor training in real-time.

### Enable Progress Tracking

```python
trainer = TransformersTrainer(
    func=train,
    enable_progression_tracking=True,  # Default: enabled
)

# Get progress
progress = client.get_job_progress(job_id)
if progress:
    print(f"Progress: {progress['progressPercentage']}%")
    print(f"Step: {progress['currentStep']}/{progress['totalSteps']}")
    print(f"Loss: {progress['trainMetrics'].get('loss')}")
```

### Progress Data Structure

```python
{
    "progressPercentage": 45,
    "estimatedRemainingSeconds": 217,
    "currentStep": 450,
    "totalSteps": 1000,
    "currentEpoch": 1,
    "totalEpochs": 3,
    "trainMetrics": {
        "loss": 0.35,
        "learning_rate": 5e-5,
        "throughput_samples_sec": 4.2
    },
    "evalMetrics": {
        "eval_loss": 0.42,
        "eval_accuracy": 0.89
    }
}
```

### Track Custom Metrics

```python
trainer = TransformersTrainer(
    func=train,
    custom_metrics={
        "perplexity": "perplexity",        # Track as-is
        "my_f1": "eval_custom_f1",         # Rename + categorize as eval
    }
)
```

**Categorization:** Prefix target name with `eval_` → `evalMetrics`, otherwise → `trainMetrics` (officially supported pattern by TransfomersTrainer)

### Monitoring Example

```python
import time

while True:
    progress = client.get_job_progress(job_id)
    if progress and progress['progressPercentage'] == 100:
        print("Training complete!")
        break
    time.sleep(30)
```

### How It Works

1. SDK injects instrumentation wrapper into training pod
2. HTTP metrics server starts on rank-0 (port 28080)
3. Controller polls `/metrics` every 30s
4. Data stored in TrainJob annotation: `trainer.opendatahub.io/trainerStatus`

### Disable If Not Needed

```python
trainer = TransformersTrainer(
    func=train,
    enable_progression_tracking=False,  # No overhead
)
```