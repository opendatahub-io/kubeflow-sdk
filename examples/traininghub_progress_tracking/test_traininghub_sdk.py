#!/usr/bin/env python3
"""
Training Hub SDK Test - Simplified Version

This script uses the SDK to submit a Training Hub job with progress tracking.
"""

from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.rhai.traininghub import (
    TrainingHubTrainer,
    TrainingHubAlgorithms,
    get_progress_tracking_annotations,
)
from kubeflow.trainer.options import Name, Annotations
from kubeflow.common.types import KubernetesBackendConfig


def training_function(func_args=None):
    """
    Training function for Training Hub SFT with progress tracking.
    This function will be embedded and run in the training pod.
    """
    import sys
    import json
    import time
    import os
    import shutil
    
    # Force multiprocessing to use 'spawn' context (avoids fork issues with FIPS)
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # FIPS/MD5 workaround - patch multiprocessing BEFORE training_hub import
    import multiprocessing.connection
    
    def _deliver_challenge_sha256(connection, authkey):
        import hmac
        message = os.urandom(20)
        connection.send_bytes(message)
        digest = hmac.new(authkey, message, 'sha256').digest()
        response = connection.recv_bytes(256)
        if len(response) != len(digest) or response != digest:
            raise multiprocessing.connection.AuthenticationError('digest received was wrong')
    
    def _answer_challenge_sha256(connection, authkey):
        import hmac
        message = connection.recv_bytes(256)
        digest = hmac.new(authkey, message, 'sha256').digest()
        connection.send_bytes(digest)
    
    multiprocessing.connection.deliver_challenge = _deliver_challenge_sha256
    multiprocessing.connection.answer_challenge = _answer_challenge_sha256
    
    # Python 3.11 compatibility patch for typing.override
    if sys.version_info < (3, 12):
        import typing
        from typing_extensions import override
        typing.override = override
    
    from training_hub import sft
    
    print("=" * 80, flush=True)
    print("[TrainingHub SDK] Starting Training Hub SDK Test", flush=True)
    print("=" * 80, flush=True)
    
    # Get checkpoint directory from func_args
    ckpt_output_dir = func_args.get("ckpt_output_dir", "/tmp/checkpoints") if func_args else "/tmp/checkpoints"
    
    # Create training data
    print("[TrainingHub SDK] Creating sample training data...", flush=True)
    os.makedirs("/tmp/data", exist_ok=True)
    
    # Clean checkpoint directory to avoid resuming from old runs
    if os.path.exists(ckpt_output_dir):
        print(f"[TrainingHub SDK] Cleaning old checkpoints from {ckpt_output_dir}", flush=True)
        shutil.rmtree(ckpt_output_dir)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    print(f"[TrainingHub SDK] Checkpoint directory ready: {ckpt_output_dir}", flush=True)
    
    sample_data = [
        {"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "Artificial Intelligence"}]},
        {"messages": [{"role": "user", "content": "What is ML?"}, {"role": "assistant", "content": "Machine Learning"}]},
        {"messages": [{"role": "user", "content": "What is DL?"}, {"role": "assistant", "content": "Deep Learning"}]},
        {"messages": [{"role": "user", "content": "What is NLP?"}, {"role": "assistant", "content": "Natural Language Processing"}]},
        {"messages": [{"role": "user", "content": "What is CV?"}, {"role": "assistant", "content": "Computer Vision"}]},
    ]
    
    # Duplicate for more training steps
    sample_data = sample_data * 4  # 20 total samples
    
    data_file = "/tmp/data/training.jsonl"
    with open(data_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"[TrainingHub SDK] Created {len(sample_data)} training samples", flush=True)
    print(f"[TrainingHub SDK] Data file: {data_file}", flush=True)
    print(f"[TrainingHub SDK] Checkpoint dir: {ckpt_output_dir}", flush=True)
    
    # Run Training Hub SFT
    print("[TrainingHub SDK] Starting SFT training...", flush=True)
    print("", flush=True)
    
    result = sft(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        data_path=data_file,
        ckpt_output_dir=ckpt_output_dir,
        data_output_dir="/tmp/processed",
        num_epochs=2,
        effective_batch_size=2,
        learning_rate=5e-5,
        max_seq_len=256,
        max_tokens_per_gpu=1024,
        warmup_steps=2,
        save_samples=5,
        nnodes=1,
        nproc_per_node=1,
        disable_flash_attn=True,
    )
    
    print("", flush=True)
    print("=" * 80, flush=True)
    print("[TrainingHub SDK] Training completed!", flush=True)
    print("=" * 80, flush=True)
    print(f"Result: {result}", flush=True)
    
    # Keep server running for debugging
    print("", flush=True)
    print("=" * 80, flush=True)
    print("[DEBUG] Pod will stay alive for 10 minutes for debugging", flush=True)
    print("=" * 80, flush=True)
    
    # Wait 10 minutes for debugging
    for remaining in range(600, 0, -30):
        mins = remaining // 60
        secs = remaining % 60
        print(f"[DEBUG] Time remaining: {mins}m {secs}s", flush=True)
        time.sleep(30)


def main():
    """Submit Training Hub job using SDK with progress tracking."""
    print("=" * 80)
    print("Training Hub SDK Test - Progress Tracking")
    print("=" * 80)
    
    # Configure client for knema-ftest namespace
    config = KubernetesBackendConfig(namespace="knema-ftest")
    client = TrainerClient(backend_config=config)
    
    # Create trainer with custom function and progress tracking
    trainer = TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.SFT,
        func=training_function,
        func_args={
            "ckpt_output_dir": "/tmp/checkpoints",
        },
        packages_to_install=["training-hub", "typing-extensions"],  # Required packages
        env={
            "PYTHONHASHSEED": "0",
            "OPENSSL_CONF": "/dev/null",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        },
        enable_progress_tracking=True,
        metrics_port=28080,
        metrics_poll_interval_seconds=30,
    )
    
    print("\n✅ Configuration:")
    print(f"   Namespace: {config.namespace}")
    print(f"   Algorithm: {trainer.algorithm.value}")
    print(f"   Function: training_function with inline data + debug wait")
    print(f"   Progress Tracking: ENABLED")
    print(f"   Metrics Port: {trainer.metrics_port}")
    print(f"   Poll Interval: {trainer.metrics_poll_interval_seconds}s")
    
    job_name = "traininghub-sdk-wrapper"
    print(f"\n🚀 Submitting TrainJob: {job_name}")
    
    try:
        # Get runtime
        runtime = client.backend.get_runtime("torch-cuda-251")
        print(f"   Runtime: torch-cuda-251")
        
        # Get progress tracking annotations
        annotations = get_progress_tracking_annotations(trainer)
        print(f"\n✅ Progress Tracking Annotations:")
        for key, value in annotations.items():
            print(f"   {key}: {value}")
        
        # Submit training job
        client.train(
            runtime=runtime,
            trainer=trainer,
            options=[
                Name(job_name),
                Annotations(annotations),
            ],
        )
        
        print(f"\n✅ Training job '{job_name}' submitted successfully!")
        
        print("\n" + "=" * 80)
        print("MONITORING COMMANDS")
        print("=" * 80)
        
        print("\n1. Check pod status:")
        print(f"   kubectl get pods -n kap-test | grep {job_name}")
        
        print("\n2. Monitor logs:")
        print(f"   POD=$(kubectl get pods -n kap-test -o name | grep {job_name} | cut -d/ -f2)")
        print(f"   kubectl logs -n kap-test $POD -f")
        
        print("\n3. Check progress metrics (after training starts):")
        print(f"   kubectl get trainjob {job_name} -n kap-test -o jsonpath='{{.metadata.annotations.trainer\\.opendatahub\\.io/trainerStatus}}' | jq")
        
        print("\n4. Monitor real-time progress:")
        print(f"   watch 'kubectl get trainjob {job_name} -n kap-test -o jsonpath=\"{{.metadata.annotations.trainer\\\\.opendatahub\\\\.io/trainerStatus}}\" | jq .progressPercentage,.currentStep,.trainMetrics.loss'")
        
        print("\n5. Check HTTP metrics directly (during debug wait):")
        print(f"   POD=$(kubectl get pods -n kap-test -o name | grep {job_name} | cut -d/ -f2)")
        print(f"   kubectl exec -n kap-test $POD -- curl http://localhost:28080/metrics | jq")
        
        print("\n6. Check checkpoint files:")
        print(f"   kubectl exec -n kap-test $POD -- ls -lah /tmp/checkpoints/")
        print(f"   kubectl exec -n kap-test $POD -- tail -1 /tmp/checkpoints/training_params_and_metrics_global0.jsonl | jq")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error submitting training job: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

