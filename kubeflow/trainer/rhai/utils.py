import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

from kubeflow_trainer_api import models
from kubernetes import client
from kubernetes.client.rest import ApiException

from kubeflow.trainer.constants import constants
from kubeflow.trainer.rhai import (
    RHAITrainer,
    traininghub,
    transformers,
)
from kubeflow.trainer.rhai.constants import (
    CHECKPOINT_MOUNT_PATH,
    CHECKPOINT_VOLUME_NAME,
    PVC_URI_SCHEME,
    S3_URI_SCHEME,
)
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


def is_primary_pod() -> bool:
    """Return True if this pod is the primary training pod.

    Detection precedence:
    1) JOB_COMPLETION_INDEX (downward API for label batch.kubernetes.io/job-completion-index)
       - Primary pod is index "0"
    2) PET_NODE_RANK (set by PyTorch Elastic runtime)
       - Primary pod is rank "0"

    If neither signal is present, returns False (conservative default).

    Returns:
        True if this is the primary pod, False otherwise.

    Examples:
        >>> os.environ["JOB_COMPLETION_INDEX"] = "0"
        >>> is_primary_pod()
        True

        >>> os.environ["JOB_COMPLETION_INDEX"] = "1"
        >>> is_primary_pod()
        False

        >>> del os.environ["JOB_COMPLETION_INDEX"]
        >>> os.environ["PET_NODE_RANK"] = "0"
        >>> is_primary_pod()
        True
    """
    job_index = os.environ.get("JOB_COMPLETION_INDEX")
    if job_index is not None:
        return job_index == "0"

    pet_rank = os.environ.get("PET_NODE_RANK")
    if pet_rank is not None:
        return pet_rank == "0"

    logger.debug(
        "is_primary_pod: neither JOB_COMPLETION_INDEX nor PET_NODE_RANK is set; returning False"
    )
    return False


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


def normalize_and_validate_output_dir(output_dir: Optional[str]) -> Optional[str]:
    """Normalize and validate storage URI for output_dir.

    Supports:
    - PVC URIs: pvc://<pvc-name>/<optional-path>
    - S3 URIs: s3://<bucket>/<optional-prefix>
    - Local filesystem paths (no scheme)

    Auto-normalizes:
    - Trailing slashes: 's3://bucket/prefix/' → 's3://bucket/prefix'
    - Duplicate slashes: 's3://bucket//prefix' → 's3://bucket/prefix'

    Args:
        output_dir: Storage URI or local path

    Returns:
        Normalized URI, or None if input is None

    Raises:
        ValueError: If URI is invalid or uses unsupported scheme
    """
    if not output_dir:
        return output_dir

    parsed = urlparse(output_dir)

    # If no scheme, treat as local filesystem path (return as-is)
    if not parsed.scheme:
        return output_dir

    # Validate and normalize URI schemes
    scheme = parsed.scheme.lower()

    # Extract scheme names from constants (remove "://")
    pvc_scheme = PVC_URI_SCHEME.replace("://", "").lower()
    s3_scheme = S3_URI_SCHEME.replace("://", "").lower()

    if scheme not in (pvc_scheme, s3_scheme):
        raise ValueError(
            f"Unsupported storage URI scheme '{scheme}://'. "
            f"Only '{PVC_URI_SCHEME}' and '{S3_URI_SCHEME}' URIs are supported. "
            f"Supported formats: '{PVC_URI_SCHEME}<pvc-name>/<path>', "
            f"'{S3_URI_SCHEME}<bucket>/<prefix>', or local filesystem paths."
        )

    # Validate netloc (bucket/PVC name) is not empty
    if not parsed.netloc:
        resource_type = "bucket" if scheme == s3_scheme else "PVC name"
        raise ValueError(
            f"Invalid {scheme.upper()} URI: '{output_dir}'. "
            f"Missing {resource_type}. "
            f"Expected format: '{scheme}://<{resource_type}>/<path>'"
        )

    # Normalize path: remove duplicate slashes and trailing slash
    path = parsed.path or ""
    if path:
        # Replace multiple consecutive slashes with single slash
        path = re.sub(r"/+", "/", path)
        # Remove trailing slash
        path = path.rstrip("/")

    # Reconstruct normalized URI
    normalized = f"{scheme}://{parsed.netloc}{path}"
    return normalized


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
        # Build emptyDir volume spec for S3 checkpoint staging
        # This volume is used as temporary local storage before uploading to S3
        volume = {
            "name": CHECKPOINT_VOLUME_NAME,
            "emptyDir": {},
        }
        volume_mount_spec = {
            "name": CHECKPOINT_VOLUME_NAME,
            "mountPath": CHECKPOINT_MOUNT_PATH,
            "readOnly": False,
        }

        # Return local staging path (training writes here, then uploads to S3)
        return CHECKPOINT_MOUNT_PATH, {"volume": volume, "volumeMount": volume_mount_spec}

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


def get_cloud_storage_credential_env_vars(
    core_api: "client.CoreV1Api",
    data_connection_name: str,
    namespace: str,
) -> list[models.IoK8sApiCoreV1EnvVar]:
    """Get environment variables for all keys in a Data Connection secret.

    Dynamically reads all keys from the Data Connection secret and creates
    environment variables with secretKeyRef for each key. This allows supporting
    any cloud storage credentials (S3, Azure, etc.) without hardcoding specific
    key names.

    Args:
        core_api: Kubernetes CoreV1Api client.
        data_connection_name: The name of the Data Connection (K8s secret).
        namespace: Namespace containing the secret.

    Returns:
        List of EnvVar objects with secretKeyRef for each key in the secret.

    Raises:
        ValueError: If the secret does not exist or user lacks permission to read it.
    """
    try:
        secret = core_api.read_namespaced_secret(name=data_connection_name, namespace=namespace)
    except ApiException as e:
        if e.status == 404:
            raise ValueError(
                f"Unable to add credentials for Data Connection '{data_connection_name}': "
                f"secret '{data_connection_name}' not found in namespace '{namespace}'. "
                "Please verify the Data Connection exists in your project."
            ) from e
        if e.status == 403:
            raise ValueError(
                f"Unable to add credentials for Data Connection '{data_connection_name}': "
                f"permission denied reading secret '{data_connection_name}' in namespace "
                f"'{namespace}'. Please ensure your service account has permission to "
                "read secrets, or contact your cluster administrator."
            ) from e
        raise

    env_vars = []
    # secret.data contains all the keys in the secret
    if secret.data:
        for key in secret.data:
            env_var = models.IoK8sApiCoreV1EnvVar(
                name=key,
                value_from=models.IoK8sApiCoreV1EnvVarSource(
                    secret_key_ref=models.IoK8sApiCoreV1SecretKeySelector(
                        name=data_connection_name,
                        key=key,
                    )
                ),
            )
            env_vars.append(env_var)

    return env_vars


def inject_cloud_storage_credentials(
    trainer: RHAITrainer,
    trainer_cr: "models.TrainerV1alpha1Trainer",
    core_api: "client.CoreV1Api",
    namespace: str,
) -> "models.TrainerV1alpha1Trainer":
    """Inject cloud storage credentials into trainer CR if using cloud output_dir.

    Validates the data connection secret exists and appends cloud storage credential
    environment variables to the trainer CR. Supports S3, Azure, etc.

    Args:
        trainer: RHAI trainer instance with data_connection_name attribute.
        trainer_cr: Trainer custom resource to inject env vars into.
        core_api: Kubernetes CoreV1Api client.
        namespace: Namespace to validate secret in.

    Returns:
        The trainer CR (with S3 credentials injected if applicable).
    """
    # Check if trainer is using S3 output_dir with data connection
    if (
        not hasattr(trainer, "output_dir")
        or not trainer.output_dir
        or not trainer.output_dir.startswith(S3_URI_SCHEME)
        or not hasattr(trainer, "data_connection_name")
        or not trainer.data_connection_name
    ):
        return trainer_cr

    # Get all keys from the data connection secret as env vars
    cloud_env_vars = get_cloud_storage_credential_env_vars(
        core_api, trainer.data_connection_name, namespace
    )
    if trainer_cr.env is None:
        trainer_cr.env = []

    # Check for duplicate env var names (K8s rejects duplicate names)
    existing_names = {env.name for env in trainer_cr.env}
    for cloud_env in cloud_env_vars:
        if cloud_env.name in existing_names:
            raise ValueError(
                f"Environment variable '{cloud_env.name}' from data connection secret "
                f"'{trainer.data_connection_name}' conflicts with an existing env var. "
                "Please remove the duplicate from your trainer configuration."
            )
        trainer_cr.env.append(cloud_env)

    return trainer_cr


def setup_rhai_trainer_storage(
    trainer: RHAITrainer,
    trainer_cr: "models.TrainerV1alpha1Trainer",
    pod_template_overrides: Optional[list],
    core_api: "client.CoreV1Api",
    namespace: str,
) -> tuple[Optional[str], "models.TrainerV1alpha1Trainer", list]:
    """Setup RHAI trainer storage: volume mounts and S3 credentials.

    This is a consolidated helper that:
    1. Parses output_dir URI and applies volume mounting to pod template overrides
    2. Injects S3 credentials into trainer CR if using S3 output_dir

    Args:
        trainer: RHAI trainer instance.
        trainer_cr: Trainer custom resource to inject env vars into.
        pod_template_overrides: Existing pod template overrides list.
        core_api: Kubernetes CoreV1Api client.
        namespace: Namespace for secret validation.

    Returns:
        Tuple of (resolved_output_dir, updated_trainer_cr, updated_pod_template_overrides).
    """
    resolved_output_dir = None

    # Apply output_dir URI parsing and volume mounting
    if hasattr(trainer, "output_dir") and trainer.output_dir:
        resolved_output_dir, pod_template_overrides = apply_output_dir_uri_to_pod_overrides(
            trainer.output_dir, pod_template_overrides
        )
    else:
        pod_template_overrides = pod_template_overrides or []

    # Inject cloud storage credentials if applicable
    trainer_cr = inject_cloud_storage_credentials(trainer, trainer_cr, core_api, namespace)

    return resolved_output_dir, trainer_cr, pod_template_overrides
