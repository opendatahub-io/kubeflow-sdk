#!/usr/bin/env python3
"""
Training Hub SFT Example: Qwen 2.5 with Progress Tracking (Algorithm Wrapper Mode)

This script uses the algorithm wrapper mode where func=None and we just pass func_args.
This should avoid multiprocessing issues since we're not embedding a complex function.
"""

import sys
sys.path.insert(0, '/Users/knema/Project/kubeflow/kubeflow-sdk')

from kubernetes import client, config
from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai.traininghub import (
    TrainingHubTrainer,
    TrainingHubAlgorithms,
    get_trainer_crd_from_training_hub_trainer,
    get_progress_tracking_annotations,
)
from kubeflow.trainer.types import types
from kubeflow_trainer_api import models


def create_pvc_if_not_exists(core_api, namespace, pvc_name, storage_size="20Gi"):
    """Create PVC if it doesn't exist."""
    try:
        core_api.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
        print(f"   ✅ PVC '{pvc_name}' already exists")
        return False
    except client.exceptions.ApiException as e:
        if e.status == 404:
            pvc = client.V1PersistentVolumeClaim(
                api_version="v1",
                kind="PersistentVolumeClaim",
                metadata=client.V1ObjectMeta(name=pvc_name, namespace=namespace),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=["ReadWriteOnce"],
                    resources=client.V1ResourceRequirements(
                        requests={"storage": storage_size}
                    ),
                ),
            )
            core_api.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc)
            print(f"   ✅ PVC '{pvc_name}' created ({storage_size})")
            return True
        else:
            raise


def main():
    """Submit Training Hub Qwen SFT job using algorithm wrapper mode."""
    
    print("=" * 80)
    print("🚀 Training Hub SFT: Qwen (Algorithm Wrapper Mode)")
    print("=" * 80)
    
    # Load Kubernetes config
    try:
        config.load_kube_config()
    except:
        config.load_incluster_config()
    
    custom_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()
    namespace = "knema-ftest"
    runtime_name = "torch-distributed"
    pvc_name = "traininghub-qwen-wrapper-storage"
    
    # Create PVC for /workspace
    print(f"\n💾 Setting up storage...")
    create_pvc_if_not_exists(core_api, namespace, pvc_name, storage_size="20Gi")
    
    print(f"\n📦 Using ClusterTrainingRuntime: {runtime_name}")
    
    # Create a minimal runtime object for SDK utilities
    runtime = types.Runtime(
        name=runtime_name,
        trainer=types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
            num_nodes=1,
            device="gpu",
            device_count="1",
        )
    )
    
    # Configure TrainingHubTrainer in algorithm wrapper mode (func=None)
    print("\n⚙️  Configuring TrainingHubTrainer (Algorithm Wrapper Mode)...")
    
    # Use a smaller model but more data for longer training with visible metrics
    trainer = TrainingHubTrainer(
        func=None,  # Algorithm wrapper mode - SDK generates wrapper
        algorithm=TrainingHubAlgorithms.SFT,
        func_args={
            # Model and data - using tiny model but more training steps
            "model_path": "HuggingFaceTB/SmolLM-135M-Instruct",  # Small model
            "data_path": "/workspace/data/qwen_training.jsonl",
            "ckpt_output_dir": "/workspace/checkpoints",
            
            # Training parameters - longer training for visible progress
            "num_epochs": 3,  # More epochs
            "effective_batch_size": 2,
            "learning_rate": 1e-4,
            "max_seq_len": 512,  # Longer sequences for slower training
            "max_batch_len": 4000,
            
            # Data processing
            "data_output_dir": "/workspace/data/processed",
            "warmup_steps": 10,
            "save_samples": 30,  # Save checkpoints more frequently
            
            # Checkpointing
            "checkpoint_at_epoch": False,  # Don't checkpoint at epoch
            
            # Hardware
            "disable_flash_attn": True,
            
            # Logging - log more frequently
            "log_level": "INFO",
            "logging_steps": 1,  # Log every step
        },
        packages_to_install=["training-hub", "typing-extensions"],
        env={
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/workspace/cache",
            "PYTHONHASHSEED": "0",
            "OPENSSL_CONF": "/dev/null",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "HF_HUB_OFFLINE": "0",  # Allow downloads
            "TRANSFORMERS_OFFLINE": "0",
        },
        
        # Progress tracking enabled
        enable_progress_tracking=True,
        metrics_port=28080,
        metrics_poll_interval_seconds=30,
    )
    
    print("✅ Trainer Configuration:")
    print(f"   Mode: Algorithm Wrapper (func=None)")
    print(f"   Algorithm: {trainer.algorithm.value}")
    print(f"   Model: HuggingFaceTB/SmolLM-135M-Instruct (135M params)")
    print(f"   Epochs: 3 (for longer training)")
    print(f"   Sequence Length: 512 (slower per step)")
    print(f"   Progress Tracking: Enabled (port {trainer.metrics_port})")
    
    # Build TrainJob CRD
    print("\n🔧 Building TrainJob CRD...")
    
    trainer_crd = get_trainer_crd_from_training_hub_trainer(
        runtime=runtime,
        trainer=trainer,
    )
    
    # Add data creation script to the command
    # We'll prepend a data generation script before the training
    original_args = trainer_crd.args[0]
    
    data_creation_script = '''
# Create sample training data
python3 << 'EODATA'
import json
import os

print("[Data] Creating training dataset...")

topics = [
    ("Python", "Explain Python list comprehensions", "List comprehensions provide a concise way to create lists"),
    ("ML", "What is gradient descent?", "Gradient descent is an optimization algorithm for minimizing loss"),
    ("Docker", "How do containers work?", "Containers package applications with dependencies"),
    ("Kubernetes", "Explain Kubernetes pods", "A Pod is the smallest deployable unit in Kubernetes"),
    ("Git", "What is git rebase?", "Git rebase integrates changes by rewriting commit history"),
]

samples = []
for i in range(30):  # 150 total samples (30 * 5 topics) for longer training
    for category, question, answer in topics:
        samples.append({
            "messages": [
                {"role": "system", "content": f"You are an expert in {category}."},
                {"role": "user", "content": f"{question} (iteration {i+1})"},
                {"role": "assistant", "content": f"{answer}. Example {i+1}."}
            ]
        })

os.makedirs("/workspace/data", exist_ok=True)
data_file = "/workspace/data/qwen_training.jsonl"
with open(data_file, 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\\n')

print(f"[Data] ✅ Created {len(samples)} training samples")
EODATA

# Clean old checkpoints
rm -rf /workspace/checkpoints
mkdir -p /workspace/checkpoints

'''
    
    # Prepend data creation to the original script
    trainer_crd.args[0] = data_creation_script + original_args
    
    # Get progress tracking annotations
    annotations = get_progress_tracking_annotations(trainer)
    
    # Create TrainJob
    job_name = "traininghub-qwen-wrapper"
    
    train_job = models.TrainerV1alpha1TrainJob(
        api_version=f"{constants.GROUP}/{constants.VERSION}",
        kind="TrainJob",
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={
                "app.kubernetes.io/name": job_name,
                "app.kubernetes.io/component": "training",
                "experiment": "qwen-sft-wrapper-mode",
            },
            annotations=annotations,
        ),
        spec=models.TrainerV1alpha1TrainJobSpec(
            runtime_ref=models.TrainerV1alpha1RuntimeRef(
                api_group=constants.GROUP,
                kind="ClusterTrainingRuntime",
                name=runtime_name,
            ),
            managed_by="trainer.kubeflow.org/trainjob-controller",
            trainer=trainer_crd,
        ),
    )
    
    print(f"\n🚀 Submitting TrainJob: {job_name}")
    
    try:
        custom_api.create_namespaced_custom_object(
            group=constants.GROUP,
            version=constants.VERSION,
            namespace=namespace,
            plural=constants.TRAINJOB_PLURAL,
            body=train_job.to_dict(),
        )
        
        print(f"✅ TrainJob '{job_name}' created successfully!")
        print("\n📊 Monitor progress:")
        print(f"  kubectl get trainjob {job_name} -n {namespace} -o jsonpath='{{.metadata.annotations.trainer\\.opendatahub\\.io/trainerStatus}}' | jq .")
        print("\n📋 View logs:")
        print(f"  kubectl logs -n {namespace} -l jobset.sigs.k8s.io/jobset-name={job_name} --tail=50 -f")
        
    except Exception as e:
        print(f"❌ Error creating TrainJob: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

