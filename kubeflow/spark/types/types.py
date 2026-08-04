# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Types for Kubeflow Spark SDK."""

<<<<<<< HEAD
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
=======
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)
>>>>>>> upstream/main


class SparkConnectState(str, Enum):
    """State of a SparkConnect session."""

    PROVISIONING = "Provisioning"
    READY = "Ready"
    RUNNING = "Running"  # Operator may set this when server is up; treated as ready
    NOT_READY = "NotReady"
    FAILED = "Failed"


@dataclass
class SparkConnectInfo:
    """Information about a SparkConnect session.

    Args:
        name: Name of the SparkConnect session.
        namespace: Kubernetes namespace. Included in SparkConnectInfo for standalone usage
            and passing info between components without requiring SparkClient context.
        state: Current state of the session.
<<<<<<< HEAD
        pod_name: Name of the server pod.
=======
        driver_pod_name: Name of the driver pod.
>>>>>>> upstream/main
        pod_ip: IP address of the server pod.
        service_name: Name of the Kubernetes service.
        creation_timestamp: Timestamp when the session was created.
    """

    name: str
    namespace: str
    state: SparkConnectState
<<<<<<< HEAD
    pod_name: str | None = None
=======
    driver_pod_name: str | None = None
>>>>>>> upstream/main
    pod_ip: str | None = None
    service_name: str | None = None
    creation_timestamp: datetime | None = None


@dataclass
class Driver:
<<<<<<< HEAD
    """Driver configuration for Spark Connect session (KEP-107 lines 165-170).
=======
    """Driver configuration for Spark Connect session.
>>>>>>> upstream/main

    The Driver configuration allows fine-grained control over the Spark driver pod.
    All fields are optional, with sensible defaults applied by the backend.

    Args:
        image: Custom container image for the driver.
        resources: Resource requirements as dict (e.g., {"cpu": "2", "memory": "4Gi"}).
<<<<<<< HEAD
            Supports arbitrary Kubernetes resources including GPUs (nvidia.com/gpu).
=======
>>>>>>> upstream/main
        java_options: JVM options for the driver (e.g., "-Xmx4g -XX:+UseG1GC").
        service_account: Kubernetes service account name for RBAC.

    Example:
        driver = Driver(
            resources={"cpu": "4", "memory": "8Gi"},
            service_account="spark-driver-prod"
        )

    Note:
        The resources dict is extensible - any valid Kubernetes resource name is supported.
<<<<<<< HEAD
        This design allows GPU support and future resource types without API changes.
=======
        This design allows future resource types without API changes.
>>>>>>> upstream/main
    """

    image: str | None = None
    resources: dict[str, str] | None = None
    java_options: str | None = None
    service_account: str | None = None


@dataclass
class Executor:
<<<<<<< HEAD
    """Executor configuration for Spark Connect session (KEP-107 lines 172-177).
=======
    """Executor configuration for Spark Connect session.
>>>>>>> upstream/main

    The Executor configuration controls the worker pods that execute Spark tasks.
    All fields are optional, with sensible defaults applied by the backend.

    Args:
        num_instances: Number of executor instances (pods).
        resources_per_executor: Resource requirements per executor as dict
<<<<<<< HEAD
            (e.g., {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"}).
            Supports arbitrary Kubernetes resources for future extensibility.
=======
            (e.g., {"cpu": "4", "memory": "8Gi"}).
>>>>>>> upstream/main
        java_options: JVM options for executors (e.g., "-Xmx28g -XX:+UseG1GC").

    Example:
        executor = Executor(
            num_instances=20,
<<<<<<< HEAD
            resources_per_executor={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "2"}
=======
            resources_per_executor={"cpu": "8", "memory": "32Gi"}
>>>>>>> upstream/main
        )

    Note:
        The resources_per_executor dict is extensible - any valid Kubernetes resource
<<<<<<< HEAD
        name is supported. This design allows GPU support, custom devices, and future
        resource types without API changes.
=======
        name is supported. This design allows future resource types without API changes.
>>>>>>> upstream/main
    """

    num_instances: int | None = None
    resources_per_executor: dict[str, str] | None = None
    java_options: str | None = None
<<<<<<< HEAD
=======


class SparkJobStatus(str, Enum):
    """State of a Spark batch job."""

    CREATED = "Created"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

    @classmethod
    def from_operator_state(
        cls,
        raw_state: str | None,
    ) -> "SparkJobStatus":
        """Map a SparkApplication state to a SparkJobStatus.

        Args:
            raw_state: SparkApplication ``applicationState.state`` value.

        Returns:
            Corresponding SparkJobStatus.

        Note:
            Unknown SparkApplication states default to FAILED so newly
            introduced operator states are handled conservatively.
        """
        normalized_state = (raw_state or "").upper()

        status = _SDK_STATE_BY_OPERATOR_STATE.get(normalized_state)

        if status is None:
            logger.warning("Unknown SparkApplication state '%s'. Defaulting to FAILED.", raw_state)
            return cls.FAILED

        return status


_SDK_STATE_BY_OPERATOR_STATE: dict[str, SparkJobStatus] = {
    state: sdk_status
    for sdk_status, operator_states in {
        SparkJobStatus.CREATED: (
            "",
            "SUBMITTED",
        ),
        SparkJobStatus.RUNNING: (
            "RUNNING",
            "SUCCEEDING",
            "SUSPENDING",
            "SUSPENDED",
            "RESUMING",
        ),
        SparkJobStatus.COMPLETED: ("COMPLETED",),
        SparkJobStatus.FAILED: (
            "FAILED",
            "SUBMISSION_FAILED",
            "FAILING",
            "PENDING_RERUN",
            "INVALIDATING",
            "UNKNOWN",
        ),
    }.items()
    for state in operator_states
}


@dataclass
class SparkJob:
    """Information about a Spark batch job.

    Args:
        name: Name of the SparkApplication.
        namespace: Kubernetes namespace containing the SparkApplication.
            Included in SparkJob for standalone usage and passing job information
            between components without requiring SparkClient context.
        status: Current state of the Spark batch job.
        creation_timestamp: Timestamp when the SparkApplication was created.
        num_executors: Number of configured Spark executor instances.
        driver_pod_name: Name of the Spark driver pod, if available.
    """

    name: str
    namespace: str
    status: SparkJobStatus | None = None
    creation_timestamp: datetime | None = None
    num_executors: int | None = None
    driver_pod_name: str | None = None


@dataclass
class FileJob:
    """Spark application referenced by a local or remote file source.

    Args:
        file_source: Path or URI of the Spark application.
            Supports local paths available to the Spark cluster as well as
            remote URIs such as s3a://, gs://, hdfs:// and https://.
        args: Optional command-line arguments passed to the application.
    """

    file_source: str
    args: list[str] | None = None


@dataclass
class FuncJob:
    """Function-based Spark application.

    The provided function must be self-contained. Any required imports
    should be placed inside the function body. Module-level globals,
    closures, and decorated functions are not supported.

    Args:
        func: Python function executed as a Spark batch job.
        func_args: Optional keyword arguments passed to the function.
    """

    func: Callable
    func_args: dict[str, Any] | None = None
>>>>>>> upstream/main
