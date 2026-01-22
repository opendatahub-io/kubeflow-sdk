import logging
from typing import Optional

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai import (
    RHAITrainer,
    traininghub,
    transformers,
)
from kubeflow.trainer.rhai.constants import (
    CHECKPOINT_EPHEMERAL_MOUNT_PATH,
    CHECKPOINT_EPHEMERAL_STORAGE_CLASS,
    CHECKPOINT_EPHEMERAL_VOLUME_NAME,
    CHECKPOINT_EPHEMERAL_VOLUME_SIZE,
    CHECKPOINT_MOUNT_PATH,
    CHECKPOINT_VOLUME_NAME,
    PVC_URI_SCHEME,
    S3_CREDENTIAL_KEYS,
    S3_URI_SCHEME,
)
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


def get_trainer_cr_from_rhai_trainer(
    runtime: types.Runtime,
    trainer: RHAITrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    if isinstance(trainer, traininghub.TrainingHubTrainer):
        return traininghub.get_trainer_cr_from_training_hub_trainer(
            runtime,
            trainer,
            initializer,
        )

    elif isinstance(trainer, transformers.TransformersTrainer):
        return transformers.get_trainer_cr_from_transformers_trainer(
            runtime,
            trainer,
            initializer,
        )

    else:
        raise ValueError(f"Unknown trainer {trainer}.")


def merge_progression_annotations(
    trainer: RHAITrainer,
    metadata_annotations: Optional[dict[str, str]] = None,
) -> Optional[dict[str, str]]:
    """Merge progression tracking annotations for RHAI trainers with existing metadata annotations.

    Args:
        trainer: RHAI trainer instance (TransformersTrainer or TrainingHubTrainer).
        metadata_annotations: Existing metadata annotations dict to merge with, if any.

    Returns:
        Merged annotations dict with progression tracking added (if enabled),
        or original metadata_annotations if progression tracking is disabled.
    """
    if (
        not hasattr(trainer, "enable_progression_tracking")
        or not trainer.enable_progression_tracking
        or not hasattr(trainer, "metrics_port")
        or not hasattr(trainer, "metrics_poll_interval_seconds")
    ):
        return metadata_annotations

    from kubeflow.trainer.rhai.constants import (
        ANNOTATION_METRICS_POLL_INTERVAL,
        ANNOTATION_METRICS_PORT,
        ANNOTATION_PROGRESSION_TRACKING,
    )

    progression_annotations = {
        ANNOTATION_PROGRESSION_TRACKING: "true",
        ANNOTATION_METRICS_PORT: str(trainer.metrics_port),
        ANNOTATION_METRICS_POLL_INTERVAL: str(trainer.metrics_poll_interval_seconds),
    }

    if metadata_annotations is None:
        return progression_annotations
    # Merge metadata_annotations last to allow users to override progression annotations
    return {**progression_annotations, **metadata_annotations}


def parse_output_dir_uri(output_dir: Optional[str]) -> tuple[Optional[str], Optional[dict]]:
    """
    Parse output_dir URI and return resolved path and volume mount specs.

    For PVC URIs (pvc://), returns a resolved local path and PVC volume specs.
    For S3 URIs (s3://), returns the local staging path and ephemeral volume specs.

    Args:
        output_dir: Output directory URI (pvc://, s3://, or local path).

    Returns:
        Tuple of (resolved_path, volume_specs) where volume_specs contains
        'volume' and 'volumeMount' dicts for Kubernetes, or None if no mounting needed.
    """
    if not output_dir:
        return None, None

    if output_dir.startswith(PVC_URI_SCHEME):
        # Parse URI and split into PVC name and path
        uri_path = output_dir[len(PVC_URI_SCHEME) :]
        parts = uri_path.split("/", 1)
        pvc_name = parts[0]

        if not pvc_name:
            raise ValueError(
                f"Invalid PVC URI: '{output_dir}'. "
                f"PVC name cannot be empty. Expected format: '{PVC_URI_SCHEME}<pvc-name>/<path>'"
            )
        checkpoint_path = parts[1] if len(parts) > 1 else ""

        # SDK mounts PVC at a standard location
        mount_path = CHECKPOINT_MOUNT_PATH
        resolved_path = f"{mount_path}/{checkpoint_path}" if checkpoint_path else mount_path

        # Build Kubernetes volume and volumeMount specs
        volume_spec = {
            "name": CHECKPOINT_VOLUME_NAME,
            "persistentVolumeClaim": {"claimName": pvc_name},
        }
        volume_mount_spec = {
            "name": CHECKPOINT_VOLUME_NAME,
            "mountPath": mount_path,
            "readOnly": False,
        }

        return resolved_path, {"volume": volume_spec, "volumeMount": volume_mount_spec}

    if output_dir.startswith(S3_URI_SCHEME):
        # Build ephemeral volume spec for S3 checkpoint staging
        # This volume is used as temporary local storage before uploading to S3
        volume_spec = {
            "name": CHECKPOINT_EPHEMERAL_VOLUME_NAME,
            "ephemeral": {
                "volumeClaimTemplate": {
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": CHECKPOINT_EPHEMERAL_STORAGE_CLASS,
                        "resources": {
                            "requests": {
                                "storage": models.IoK8sApimachineryPkgApiResourceQuantity(
                                    CHECKPOINT_EPHEMERAL_VOLUME_SIZE
                                ),
                            }
                        },
                    }
                }
            },
        }
        volume_mount_spec = {
            "name": CHECKPOINT_EPHEMERAL_VOLUME_NAME,
            "mountPath": CHECKPOINT_EPHEMERAL_MOUNT_PATH,
        }

        # Return local staging path (training writes here, then uploads to S3)
        return CHECKPOINT_EPHEMERAL_MOUNT_PATH, {"volume": volume_spec, "volumeMount": volume_mount_spec}

    return output_dir, None


def apply_output_dir_uri_to_pod_overrides(
    output_dir: str,
    pod_template_overrides: Optional[list],
) -> tuple[str, list]:
    """
    Process output_dir URI and apply volume mounting to pod template overrides.

    Handles both PVC URIs (mounts PVC) and S3 URIs (mounts ephemeral staging volume).

    Args:
        output_dir: Output directory URI (pvc://, s3://, or local path).
        pod_template_overrides: Existing pod template overrides list.

    Returns:
        Tuple of (resolved_output_dir, updated_pod_template_overrides).
    """
    resolved_output_dir, volume_mount_specs = parse_output_dir_uri(output_dir)

    # If no volume mounting needed, return as-is
    if volume_mount_specs is None:
        return resolved_output_dir, pod_template_overrides or []

    # Initialize pod_template_overrides as list if needed
    if pod_template_overrides is None:
        pod_template_overrides = []

    # Find existing override for node target job, or create new one
    node_override = None
    for override in pod_template_overrides:
        target_jobs = override.get("targetJobs", [])
        if any(job.get("name") == constants.NODE for job in target_jobs):
            node_override = override
            break

    if node_override is None:
        # Create new override targeting the node job
        node_override = {"targetJobs": [{"name": constants.NODE}], "spec": {}}
        pod_template_overrides.append(node_override)

    # Ensure spec dict exists
    if "spec" not in node_override:
        node_override["spec"] = {}

    spec_dict = node_override["spec"]

    # Add volume to spec (only if not already present)
    if "volumes" not in spec_dict:
        spec_dict["volumes"] = []

    # Check if volume with the same name already exists
    volume_name = volume_mount_specs["volume"]["name"]
    if any(vol.get("name") == volume_name for vol in spec_dict["volumes"]):
        raise ValueError(
            f"Volume name conflict: A volume with name '{volume_name}' already exists in "
            f"pod_template_overrides. This name is reserved by Kubeflow SDK for "
            f"checkpoint storage. Please rename your existing volume to a different name."
        )
    spec_dict["volumes"].append(volume_mount_specs["volume"])

    # Add volumeMount to the trainer container
    if "containers" not in spec_dict:
        spec_dict["containers"] = []

    # Find the trainer container in containers list
    trainer_container_dict = None
    for container_dict in spec_dict["containers"]:
        if container_dict.get("name") == constants.NODE:
            trainer_container_dict = container_dict
            break

    if trainer_container_dict is None:
        # Create new container override for trainer
        trainer_container_dict = {"name": constants.NODE, "volumeMounts": []}
        spec_dict["containers"].append(trainer_container_dict)

    # Add volumeMount to trainer container (only if not already present)
    if "volumeMounts" not in trainer_container_dict:
        trainer_container_dict["volumeMounts"] = []

    # Check if volumeMount with the same name already exists
    volume_mount_name = volume_mount_specs["volumeMount"]["name"]
    if any(vm.get("name") == volume_mount_name for vm in trainer_container_dict["volumeMounts"]):
        raise ValueError(
            f"VolumeMount name conflict: A volumeMount with name '{volume_mount_name}' already "
            f"exists in pod_template_overrides. This name is reserved by Kubeflow SDK for "
            f"checkpoint storage. Please rename your existing volumeMount to a different name."
        )
    trainer_container_dict["volumeMounts"].append(volume_mount_specs["volumeMount"])

    return resolved_output_dir, pod_template_overrides


def get_s3_credential_env_vars(
    secret_name: str,
) -> list[models.IoK8sApiCoreV1EnvVar]:
    """Get environment variables for S3 credentials from a Kubernetes secret.

    Uses valueFrom with secretKeyRef to load specific keys from the secret as environment
    variables. This exposes keys like AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
    Keys that don't exist in the secret are marked as optional.

    Args:
        secret_name: The name of the K8s secret containing storage credentials.

    Returns:
        List of EnvVar objects with secretKeyRef for each S3 credential key.
    """
    env_vars = []
    for key in S3_CREDENTIAL_KEYS:
        env_var = models.IoK8sApiCoreV1EnvVar(
            name=key,
            value_from=models.IoK8sApiCoreV1EnvVarSource(
                secret_key_ref=models.IoK8sApiCoreV1SecretKeySelector(
                    name=secret_name,
                    key=key,
                    optional=True,  # Don't fail if key doesn't exist in secret
                )
            ),
        )
        env_vars.append(env_var)

    return env_vars


