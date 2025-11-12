# TransformersTrainer Quickstart

Get real-time training progress for HuggingFace Transformers and TRL jobs **without changing your training code**.

**What you get:**
- 📊 Live progress tracking (steps, loss, learning rate, ETA)
- 🔄 Automatic HTTP metrics server
- 📈 Custom metrics support
- 🚀 Works with TRL trainers (SFTTrainer, DPOTrainer, etc.)
- 🎯 Zero code changes to your training function

## 🚀 Quick Start (3 Steps)

```python
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.rhai import TransformersTrainer, get_job_progress

# 1. Define your training function (standard Transformers code)
def train():
    from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
    from datasets import load_dataset
    
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    dataset = load_dataset("imdb")
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./output", num_train_epochs=3),
        train_dataset=dataset["train"].select(range(1000)),
    )
    trainer.train()

# 2. Submit with TransformersTrainer
client = TrainerClient()
job_name = client.train(
    trainer=TransformersTrainer(
        func=train,
        num_nodes=2,
        resources_per_node={"gpu": 1},
        packages_to_install=["transformers", "datasets", "torch"],
    ),
    runtime=client.get_runtime("pytorch-distributed"),
)

# 3. Monitor progress
progress = get_job_progress(job_name)
print(f"Progress: {progress['progress']['percent']:.1f}%")
print(f"Status: {progress['status_message']}")
```

## 📊 What You Get

**Progress Data** available via `get_job_progress()`:
```python
{
  "status": "training",
  "status_message": "Training in progress: 45.3% complete, 2h 44m remaining",
  "progress": {
    "step_current": 340,
    "step_total": 750,
    "percent": 45.3,
    "epoch": 1
  },
  "time": {
    "elapsed": "6m 4s",
    "remaining": "2h 44m"
  },
  "metrics": {
    "loss": 0.4523,
    "learning_rate": 5.2e-5,
    "throughput_samples_sec": 4.08
  }
}
```

**How it works:**
1. SDK automatically starts HTTP metrics server in your training pod (port 28080)
2. Kubeflow Training Operator polls metrics every 30s
3. Progress data stored in TrainJob annotations
4. Access via `get_job_progress()` or kubectl

## ⚙️ Configuration Options

```python
trainer = TransformersTrainer(
    func=train,
    
    # Track custom metrics (map log keys to metric names)
    custom_metrics={
        "eval_accuracy": "accuracy",
        "eval_f1": "f1_score",
    },
    
    # Change metrics server port (default: 28080)
    metrics_port=8080,
    
    # Disable progression tracking if not needed
    enable_progression_tracking=False,
)
```

## 📈 Monitoring Progress Loop

```python
import time
from kubeflow.trainer.rhai import get_job_progress

while True:
    progress = get_job_progress(job_name, namespace="default")
    
    if progress:
        print(f"📊 {progress['status_message']}")
        print(f"   Step {progress['progress']['step_current']}/{progress['progress']['step_total']}")
        print(f"   Loss: {progress['metrics'].get('loss', 'N/A')}")
        print(f"   Time: {progress['time']['elapsed']} elapsed, {progress['time']['remaining']} remaining")
        
        if progress['status'] in ['completed', 'failed']:
            break
    
    time.sleep(30)
```

## 🎯 TRL Support

Works with **TRL trainers** (SFTTrainer, DPOTrainer, PPOTrainer, etc.) - no code changes needed!

```python
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.rhai import TransformersTrainer

def train_with_trl():
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    trainer = SFTTrainer(  # Automatically instrumented!
        model=model,
        args=SFTConfig(output_dir="./output", num_train_epochs=3),
        train_dataset=load_dataset("tatsu-lab/alpaca", split="train[:1000]"),
        tokenizer=tokenizer,
    )
    trainer.train()

# Submit just like any other TransformersTrainer
client.train(
    trainer=TransformersTrainer(
        func=train_with_trl,
        packages_to_install=["trl", "transformers", "datasets"],
    ),
    runtime=client.get_runtime("pytorch-distributed"),
)
```

**Supported TRL trainers:** SFTTrainer, DPOTrainer, PPOTrainer, RewardTrainer, and more

## 💡 Key Points

- ✅ **Zero code changes** - Standard Transformers/TRL code works as-is
- ✅ **Automatic instrumentation** - SDK injects progress callbacks at runtime
- ✅ **Distributed training** - Works with multi-node, multi-GPU setups
- ✅ **Non-intrusive** - HTTP server runs in background daemon thread
- ✅ **Kubernetes-native** - Progress stored in TrainJob annotations