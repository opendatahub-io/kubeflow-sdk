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

"""Unit tests for Kubernetes Spark backend utilities."""

<<<<<<< HEAD
=======
from datetime import datetime
import multiprocessing
from unittest.mock import Mock, patch

>>>>>>> upstream/main
from kubeflow_spark_api import models
import pytest

from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.backends.kubernetes.utils import (
    _memory_kubernetes_to_spark,
<<<<<<< HEAD
    build_service_url,
    build_spark_connect_cr,
    generate_session_name,
    get_spark_connect_info_from_cr,
    validate_spark_connect_url,
)
from kubeflow.spark.types.types import Driver, Executor, SparkConnectInfo, SparkConnectState


class TestMemoryKubernetesToSpark:
    """Tests for _memory_kubernetes_to_spark."""

    @pytest.mark.parametrize(
        "k8s_memory,expected_spark",
        [
            ("4Gi", "4g"),
            ("512Mi", "512m"),
            ("8Gi", "8g"),
            ("1Ti", "1t"),
            ("4g", "4g"),
            ("512m", "512m"),
            ("2G", "2g"),
        ],
    )
    def test_conversion(self, k8s_memory: str, expected_spark: str) -> None:
        assert _memory_kubernetes_to_spark(k8s_memory) == expected_spark


class TestGenerateSessionName:
    """Tests for generate_session_name function."""

    def test_generates_unique_name(self):
        """U11: Generate unique session name with prefix."""
        name = generate_session_name()
        assert name.startswith("spark-connect-")
        assert len(name) > len("spark-connect-")

    def test_generates_different_names(self):
        """Generated names should be unique."""
        names = {generate_session_name() for _ in range(10)}
        assert len(names) == 10


class TestValidateSparkConnectUrl:
    """Tests for validate_spark_connect_url function."""

    def test_valid_url(self):
        """U12: Valid Spark Connect URL passes."""
        assert validate_spark_connect_url("sc://localhost:15002") is True
        assert validate_spark_connect_url("sc://spark-server:15002") is True

    def test_invalid_scheme(self):
        """U13: Invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            validate_spark_connect_url("http://localhost:15002")

    def test_missing_port(self):
        """U14: Missing port raises ValueError."""
        with pytest.raises(ValueError, match="Port is required"):
            validate_spark_connect_url("sc://localhost")


class TestBuildServiceUrl:
    """Tests for build_service_url function."""

    def test_build_from_session_info(self):
        """U15: Build service URL from SparkConnectInfo."""
        info = SparkConnectInfo(
            name="my-session",
            namespace="spark",
            state=SparkConnectState.READY,
            service_name="my-session-svc",
        )
        url = build_service_url(info)
        assert url == "sc://my-session-svc.spark.svc.cluster.local:15002"

    def test_build_without_service_name(self):
        """Build URL when service_name is None."""
        info = SparkConnectInfo(
            name="my-session",
            namespace="default",
            state=SparkConnectState.READY,
        )
        url = build_service_url(info)
        assert "my-session-svc" in url


class TestBuildSparkConnectCr:
    """Tests for build_spark_connect_cr function."""

    def test_minimal_cr(self):
        """U01: Build SparkConnect CR with minimal config."""
        spark_connect = build_spark_connect_cr(name="test-session", namespace="default")

=======
    _resolve_driver_resources,
    _resolve_executor_resources,
    _validate_cpu_value,
    build_service_url,
    build_spark_connect_cr,
    generate_job_name,
    generate_session_name,
    get_command_using_spark_func,
    get_func_job_init_container,
    get_spark_application_cr_from_file_job,
    get_spark_application_cr_from_func_job,
    get_spark_application_info_from_cr,
    get_spark_connect_info_from_cr,
    get_spark_job_driver_spec,
    get_spark_job_executor_spec,
    read_pod_logs,
    validate_spark_connect_url,
)
from kubeflow.spark.test.common import FAILED, SUCCESS, TestCase
from kubeflow.spark.types.types import (
    Driver,
    Executor,
    SparkConnectInfo,
    SparkConnectState,
    SparkJobStatus,
)

# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def minimal_spec():
    """Creates minimal SparkConnect spec."""
    return models.SparkV1alpha1SparkConnectSpec(
        sparkVersion=constants.DEFAULT_SPARK_VERSION,
        server=models.SparkV1alpha1ServerSpec(),
        executor=models.SparkV1alpha1ExecutorSpec(),
    )


@pytest.fixture
def spark_application_spec():
    """Create minimal SparkApplication spec."""
    return models.SparkV1beta2SparkApplicationSpec(
        spark_version=constants.DEFAULT_SPARK_VERSION,
        type="Python",
        mode="cluster",
        image=constants.DEFAULT_SPARK_IMAGE,
        main_application_file="s3://bucket/job.py",
        driver=models.SparkV1beta2DriverSpec(
            cores=1,
            memory="1g",
        ),
        executor=models.SparkV1beta2ExecutorSpec(
            cores=2,
            memory="2g",
            instances=5,
        ),
    )


# --------------------------
# Test Helpers
# --------------------------


def sample_function():
    """Simple function for testing."""
    print("hello")


def sample_function_with_args(name: str, age: int):
    """Function with arguments for testing."""
    print(name, age)


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="Convert Gi to Spark g",
            config={"k8s_memory": "4Gi"},
            expected_output="4g",
        ),
        TestCase(
            name="Convert Mi to Spark m",
            config={"k8s_memory": "512Mi"},
            expected_output="512m",
        ),
        TestCase(
            name="Convert larger Gi value",
            config={"k8s_memory": "8Gi"},
            expected_output="8g",
        ),
        TestCase(
            name="Convert Ti to Spark t",
            config={"k8s_memory": "1Ti"},
            expected_output="1t",
        ),
        TestCase(
            name="Preserve lowercase g",
            config={"k8s_memory": "4g"},
            expected_output="4g",
        ),
        TestCase(
            name="Preserve lowercase m",
            config={"k8s_memory": "512m"},
            expected_output="512m",
        ),
        TestCase(
            name="Normalize uppercase G",
            config={"k8s_memory": "2G"},
            expected_output="2g",
        ),
        TestCase(
            name="Convert fractional Gi to Mi",
            config={"k8s_memory": "1.5Gi"},
            expected_output="1536m",
        ),
    ],
)
def test_memory_kubernetes_to_spark(test_case: TestCase) -> None:
    """Tests _memory_kubernetes_to_spark."""
    assert _memory_kubernetes_to_spark(test_case.config["k8s_memory"]) == test_case.expected_output


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="generate session name with prefix",
            expected_status=SUCCESS,
            expected_output="spark-connect-",
        ),
        TestCase(
            name="generate unique session names",
            expected_status=SUCCESS,
            expected_output=10,
        ),
    ],
)
def test_generate_session_name(test_case: TestCase) -> None:
    """Tests generate_session_name."""

    print("Executing test:", test_case.name)

    assert test_case.expected_status == SUCCESS

    if test_case.name == "generate session name with prefix":
        name = generate_session_name()

        assert name.startswith(test_case.expected_output)
        assert len(name) > len(test_case.expected_output)

    elif test_case.name == "generate unique session names":
        names = {generate_session_name() for _ in range(10)}

        assert len(names) == test_case.expected_output

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid spark connect url",
            expected_status=SUCCESS,
            config={"url": "sc://localhost:15002"},
            expected_output=True,
        ),
        TestCase(
            name="valid spark connect server url",
            expected_status=SUCCESS,
            config={"url": "sc://spark-server:15002"},
            expected_output=True,
        ),
        TestCase(
            name="invalid url scheme",
            expected_status=FAILED,
            config={"url": "http://localhost:15002"},
            expected_error=ValueError,
            expected_output="Invalid scheme",
        ),
        TestCase(
            name="missing port",
            expected_status=FAILED,
            config={"url": "sc://localhost"},
            expected_error=ValueError,
            expected_output="Port is required",
        ),
    ],
)
def test_validate_spark_connect_url(test_case: TestCase) -> None:
    """Tests validate_spark_connect_url."""

    print("Executing test:", test_case.name)

    if test_case.expected_status == SUCCESS:
        assert validate_spark_connect_url(test_case.config["url"]) == test_case.expected_output
    else:
        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            validate_spark_connect_url(test_case.config["url"])

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="build service url with service name",
            expected_status=SUCCESS,
            config={
                "info": SparkConnectInfo(
                    name="my-session",
                    namespace="spark",
                    state=SparkConnectState.READY,
                    service_name="my-session-svc",
                ),
            },
            expected_output="sc://my-session-svc.spark.svc.cluster.local:15002",
        ),
        TestCase(
            name="build service url without service name",
            expected_status=SUCCESS,
            config={
                "info": SparkConnectInfo(
                    name="my-session",
                    namespace="default",
                    state=SparkConnectState.READY,
                ),
            },
            expected_output="my-session-svc",
        ),
    ],
)
def test_build_service_url(test_case: TestCase) -> None:
    """Tests build_service_url."""

    print("Executing test:", test_case.name)

    url = build_service_url(test_case.config["info"])

    assert test_case.expected_status == SUCCESS

    if test_case.name == "build service url with service name":
        assert url == test_case.expected_output
    elif test_case.name == "build service url without service name":
        assert test_case.expected_output in url

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="minimal spark connect cr",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="spark connect cr with num executors",
            expected_status=SUCCESS,
            config={
                "num_executors": 3,
            },
        ),
        TestCase(
            name="spark connect cr with executor resources",
            expected_status=SUCCESS,
            config={
                "resources_per_executor": {
                    "cpu": "2",
                    "memory": "4Gi",
                },
            },
        ),
        TestCase(
            name="spark connect cr with spark conf",
            expected_status=SUCCESS,
            config={
                "spark_conf": {
                    "spark.sql.adaptive.enabled": "true",
                },
            },
        ),
        TestCase(
            name="spark conf overrides grpc binding address",
            expected_status=SUCCESS,
            config={
                "spark_conf": {
                    "spark.connect.grpc.binding.address": "127.0.0.1",
                },
            },
        ),
        TestCase(
            name="spark connect cr with driver image",
            expected_status=SUCCESS,
            config={
                "driver": Driver(image="custom-spark:v1"),
            },
        ),
        TestCase(
            name="spark connect cr with driver resources",
            expected_status=SUCCESS,
            config={
                "driver": Driver(
                    resources={
                        "cpu": "2",
                        "memory": "2Gi",
                    },
                ),
            },
        ),
        TestCase(
            name="spark connect cr with service account",
            expected_status=SUCCESS,
            config={
                "driver": Driver(service_account="spark-sa"),
            },
        ),
        TestCase(
            name="spark connect cr with executor config",
            expected_status=SUCCESS,
            config={
                "executor": Executor(
                    num_instances=5,
                    resources_per_executor={
                        "cpu": "4",
                        "memory": "8Gi",
                    },
                ),
            },
        ),
        TestCase(
            name="spark connect cr with app name",
            expected_status=SUCCESS,
            config={
                "spark_conf": {
                    "spark.app.name": "my-spark-app",
                },
            },
        ),
        TestCase(
            name="executor config overrides num executors",
            expected_status=SUCCESS,
            config={
                "num_executors": 5,
                "executor": Executor(
                    num_instances=10,
                ),
            },
        ),
        TestCase(
            name="executor config overrides executor resources",
            expected_status=SUCCESS,
            config={
                "resources_per_executor": {
                    "cpu": "4",
                    "memory": "8Gi",
                },
                "executor": Executor(
                    resources_per_executor={
                        "cpu": "8",
                        "memory": "16Gi",
                    },
                ),
            },
        ),
        TestCase(
            name="kep107 level2 simple mode",
            expected_status=SUCCESS,
            config={
                "num_executors": 5,
                "resources_per_executor": {
                    "cpu": "5",
                    "memory": "10Gi",
                },
            },
        ),
        TestCase(
            name="kep107 level3 advanced mode",
            expected_status=SUCCESS,
            config={
                "driver": Driver(
                    resources={
                        "cpu": "4",
                        "memory": "8Gi",
                    },
                    service_account="spark-driver-prod",
                ),
                "executor": Executor(
                    num_instances=20,
                    resources_per_executor={
                        "cpu": "8",
                        "memory": "32Gi",
                    },
                ),
            },
        ),
    ],
)
def test_build_spark_connect_cr(test_case: TestCase) -> None:
    """Tests build_spark_connect_cr."""
    print("Executing test:", test_case.name)

    spark_connect = build_spark_connect_cr(
        name="test-session",
        namespace="default",
        **test_case.config,
    )

    assert test_case.expected_status == SUCCESS

    if test_case.name == "minimal spark connect cr":
>>>>>>> upstream/main
        assert (
            spark_connect.api_version
            == f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}"
        )
        assert spark_connect.kind == constants.SPARK_CONNECT_KIND
        assert spark_connect.metadata.name == "test-session"
        assert spark_connect.metadata.namespace == "default"
        assert spark_connect.spec.spark_version == constants.DEFAULT_SPARK_VERSION
        assert spark_connect.spec.executor.instances == constants.DEFAULT_NUM_EXECUTORS
        assert spark_connect.spec.executor.cores == constants.DEFAULT_EXECUTOR_CPU
        assert spark_connect.spec.executor.memory == "512m"
        assert spark_connect.spec.server.cores == constants.DEFAULT_DRIVER_CPU
        assert spark_connect.spec.server.memory == "512m"
        assert spark_connect.spec.spark_conf["spark.connect.grpc.binding.address"] == "0.0.0.0"

<<<<<<< HEAD
    def test_with_num_executors(self):
        """U02: Build CR with num_executors."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            num_executors=3,
        )
        assert spark_connect.spec.executor.instances == 3

    def test_with_resources(self):
        """U03: Build CR with resources_per_executor."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            resources_per_executor={"cpu": "2", "memory": "4Gi"},
        )
        assert spark_connect.spec.executor.cores == 2
        assert spark_connect.spec.executor.memory == "4g"

    def test_with_spark_conf(self):
        """U04: Build CR with spark_conf."""
        spark_conf = {"spark.sql.adaptive.enabled": "true"}
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            spark_conf=spark_conf,
        )
        assert spark_connect.spec.spark_conf["spark.jars"].endswith(
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert spark_connect.spec.spark_conf["spark.sql.adaptive.enabled"] == "true"

    def test_spark_conf_overrides_binding_address(self):
        """User spark_conf can override default grpc binding address."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            spark_conf={"spark.connect.grpc.binding.address": "127.0.0.1"},
        )
        assert spark_connect.spec.spark_conf["spark.connect.grpc.binding.address"] == "127.0.0.1"

    def test_with_driver_image(self):
        """U05: Build CR with custom image via Driver."""
        driver = Driver(image="custom-spark:v1")
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert spark_connect.spec.image == "custom-spark:v1"

    def test_with_driver_config(self):
        """U06: Build CR with Driver config (KEP-107 resources dict)."""
        driver = Driver(resources={"cpu": "2", "memory": "2Gi"})
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert spark_connect.spec.server.cores == 2
        assert spark_connect.spec.server.memory == "2g"

    def test_with_service_account(self):
        """U07: Build CR with service account."""
        driver = Driver(service_account="spark-sa")
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
        )
        assert spark_connect.spec.server.template.spec.service_account_name == "spark-sa"

    def test_with_executor_config(self):
        """Build CR with Executor config (KEP-107 resources_per_executor)."""
        executor = Executor(
            num_instances=5,
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
        )
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            executor=executor,
        )
=======
    elif test_case.name == "spark connect cr with num executors":
        assert spark_connect.spec.executor.instances == 3

    elif test_case.name == "spark connect cr with executor resources":
        assert spark_connect.spec.executor.cores == 2
        assert spark_connect.spec.executor.memory == "4g"

    elif test_case.name == "spark connect cr with spark conf":
        assert spark_connect.spec.spark_conf["spark.jars"].endswith(
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-"
            f"{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert spark_connect.spec.spark_conf["spark.sql.adaptive.enabled"] == "true"

    elif test_case.name == "spark conf overrides grpc binding address":
        assert spark_connect.spec.spark_conf["spark.connect.grpc.binding.address"] == "127.0.0.1"

    elif test_case.name == "spark connect cr with driver image":
        assert spark_connect.spec.image == "custom-spark:v1"

    elif test_case.name == "spark connect cr with driver resources":
        assert spark_connect.spec.server.cores == 2
        assert spark_connect.spec.server.memory == "2g"

    elif test_case.name == "spark connect cr with service account":
        assert spark_connect.spec.server.template.spec.service_account_name == "spark-sa"

    elif test_case.name == "spark connect cr with executor config":
>>>>>>> upstream/main
        assert spark_connect.spec.executor.instances == 5
        assert spark_connect.spec.executor.cores == 4
        assert spark_connect.spec.executor.memory == "8g"

<<<<<<< HEAD
    def test_app_name(self):
        """Build CR with spark.app.name via spark_conf."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            spark_conf={"spark.app.name": "my-spark-app"},
        )
        assert spark_connect.spec.spark_conf["spark.jars"].endswith(
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert spark_connect.spec.spark_conf["spark.app.name"] == "my-spark-app"

    def test_precedence_executor_instances(self):
        """Test precedence: executor.num_instances > num_executors."""
        executor = Executor(num_instances=10)
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            num_executors=5,
            executor=executor,
        )
        # Executor object should override simple parameter
        assert spark_connect.spec.executor.instances == 10

    def test_precedence_executor_resources(self):
        """Test precedence: executor.resources_per_executor > resources_per_executor."""
        executor = Executor(
            resources_per_executor={"cpu": "8", "memory": "16Gi"},
        )
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            resources_per_executor={"cpu": "4", "memory": "8Gi"},
            executor=executor,
        )
        # Executor object should override simple parameter
        assert spark_connect.spec.executor.cores == 8
        assert spark_connect.spec.executor.memory == "16g"

    def test_kep107_level2_simple(self):
        """Test KEP-107 Level 2 (simple mode) example."""
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            num_executors=5,
            resources_per_executor={"cpu": "5", "memory": "10Gi"},
        )
=======
    elif test_case.name == "spark connect cr with app name":
        assert spark_connect.spec.spark_conf["spark.jars"].endswith(
            f"spark-connect_{constants.SPARK_CONNECT_PACKAGE_SCALA_VERSION}-"
            f"{constants.DEFAULT_SPARK_VERSION}.jar"
        )
        assert spark_connect.spec.spark_conf["spark.app.name"] == "my-spark-app"

    elif test_case.name == "executor config overrides num executors":
        assert spark_connect.spec.executor.instances == 10

    elif test_case.name == "executor config overrides executor resources":
        assert spark_connect.spec.executor.cores == 8
        assert spark_connect.spec.executor.memory == "16g"

    elif test_case.name == "kep107 level2 simple mode":
>>>>>>> upstream/main
        assert spark_connect.spec.executor.instances == 5
        assert spark_connect.spec.executor.cores == 5
        assert spark_connect.spec.executor.memory == "10g"

<<<<<<< HEAD
    def test_kep107_level3_advanced(self):
        """Test KEP-107 Level 3 (advanced mode) example."""
        driver = Driver(
            resources={"cpu": "4", "memory": "8Gi"},
            service_account="spark-driver-prod",
        )
        executor = Executor(
            num_instances=20,
            resources_per_executor={"cpu": "8", "memory": "32Gi"},
        )
        spark_connect = build_spark_connect_cr(
            name="test-session",
            namespace="default",
            driver=driver,
            executor=executor,
        )
=======
    elif test_case.name == "kep107 level3 advanced mode":
>>>>>>> upstream/main
        assert spark_connect.spec.server.cores == 4
        assert spark_connect.spec.server.memory == "8g"
        assert spark_connect.spec.server.template.spec.service_account_name == "spark-driver-prod"
        assert spark_connect.spec.executor.instances == 20
        assert spark_connect.spec.executor.cores == 8
        assert spark_connect.spec.executor.memory == "32g"

<<<<<<< HEAD

class TestGetSparkConnectInfoFromCr:
    """Tests for get_spark_connect_info_from_cr function."""

    @pytest.fixture
    def minimal_spec(self):
        """Create minimal spec required for SparkConnect model."""
        return models.SparkV1alpha1SparkConnectSpec(
            sparkVersion=constants.DEFAULT_SPARK_VERSION,
            server=models.SparkV1alpha1ServerSpec(),
            executor=models.SparkV1alpha1ExecutorSpec(),
        )

    def test_parse_ready_status(self, minimal_spec):
        """U08: Parse CR with Ready state."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="my-session",
                namespace="default",
                creationTimestamp="2025-01-12T10:30:00Z",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(
                state="Ready",
                server=models.SparkV1alpha1SparkConnectServerStatus(
                    podName="my-session-server-0",
                    podIp="10.0.0.5",
                    serviceName="my-session-svc",
                ),
            ),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.name == "my-session"
        assert info.namespace == "default"
        assert info.state == SparkConnectState.READY
        assert info.pod_name == "my-session-server-0"
        assert info.pod_ip == "10.0.0.5"
        assert info.service_name == "my-session-svc"
        assert info.creation_timestamp is not None

    def test_parse_provisioning_status(self, minimal_spec):
        """U09: Parse CR with Provisioning state."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="new-session",
                namespace="spark",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(state="Provisioning"),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.name == "new-session"
        assert info.namespace == "spark"
        assert info.state == SparkConnectState.PROVISIONING

    def test_parse_failed_status(self, minimal_spec):
        """U10: Parse CR with Failed state."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="failed-session",
                namespace="default",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(state="Failed"),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.state == SparkConnectState.FAILED

    def test_parse_running_status(self, minimal_spec):
        """Parse CR with Running state (operator may set this when server is up)."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="run-session",
                namespace="default",
            ),
            spec=minimal_spec,
            status=models.SparkV1alpha1SparkConnectStatus(
                state="Running",
                server=models.SparkV1alpha1SparkConnectServerStatus(
                    podName="run-session-server",
                    serviceName="run-session-svc",
                ),
            ),
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)
        assert info.state == SparkConnectState.RUNNING
        assert info.service_name == "run-session-svc"

    def test_parse_empty_status(self, minimal_spec):
        """Parse CR with empty status."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="new-session",
                namespace="default",
            ),
            spec=minimal_spec,
        )
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        assert info.state == SparkConnectState.PROVISIONING
        assert info.pod_name is None

    def test_invalid_cr_missing_name_raises_error(self, minimal_spec):
        """Test that CR without name in metadata raises ValueError."""
        spark_connect_cr = models.SparkV1alpha1SparkConnect(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                namespace="default",
            ),
            spec=minimal_spec,
        )
        with pytest.raises(ValueError, match="SparkConnect CR is invalid"):
            get_spark_connect_info_from_cr(spark_connect_cr)
=======
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="ready status",
            expected_status=SUCCESS,
            config={
                "metadata": {
                    "name": "my-session",
                    "namespace": "default",
                    "creationTimestamp": "2025-01-12T10:30:00Z",
                },
                "status": models.SparkV1alpha1SparkConnectStatus(
                    state="Ready",
                    server=models.SparkV1alpha1SparkConnectServerStatus(
                        podName="my-session-server-0",
                        podIp="10.0.0.5",
                        serviceName="my-session-svc",
                    ),
                ),
            },
        ),
        TestCase(
            name="provisioning status",
            expected_status=SUCCESS,
            config={
                "metadata": {
                    "name": "new-session",
                    "namespace": "spark",
                },
                "status": models.SparkV1alpha1SparkConnectStatus(
                    state="Provisioning",
                ),
            },
        ),
        TestCase(
            name="failed status",
            expected_status=SUCCESS,
            config={
                "metadata": {
                    "name": "failed-session",
                    "namespace": "default",
                },
                "status": models.SparkV1alpha1SparkConnectStatus(
                    state="Failed",
                ),
            },
        ),
        TestCase(
            name="running status",
            expected_status=SUCCESS,
            config={
                "metadata": {
                    "name": "run-session",
                    "namespace": "default",
                },
                "status": models.SparkV1alpha1SparkConnectStatus(
                    state="Running",
                    server=models.SparkV1alpha1SparkConnectServerStatus(
                        podName="run-session-server",
                        serviceName="run-session-svc",
                    ),
                ),
            },
        ),
        TestCase(
            name="empty status",
            expected_status=SUCCESS,
            config={
                "metadata": {
                    "name": "new-session",
                    "namespace": "default",
                },
            },
        ),
        TestCase(
            name="missing name",
            expected_status=FAILED,
            config={
                "metadata": {
                    "namespace": "default",
                },
            },
            expected_error=ValueError,
            expected_output="SparkConnect CR is invalid",
        ),
    ],
)
def test_get_spark_connect_info_from_cr(
    test_case: TestCase,
    minimal_spec,
) -> None:
    """Tests get_spark_connect_info_from_cr."""

    print("Executing test:", test_case.name)

    spark_connect_cr = models.SparkV1alpha1SparkConnect(
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            **test_case.config["metadata"],
        ),
        spec=minimal_spec,
        status=test_case.config.get("status"),
    )

    if test_case.expected_status == SUCCESS:
        info = get_spark_connect_info_from_cr(spark_connect_cr)

        if test_case.name == "ready status":
            assert info.name == "my-session"
            assert info.namespace == "default"
            assert info.state == SparkConnectState.READY
            assert info.driver_pod_name == "my-session-server-0"
            assert info.pod_ip == "10.0.0.5"
            assert info.service_name == "my-session-svc"
            assert info.creation_timestamp is not None

        elif test_case.name == "provisioning status":
            assert info.name == "new-session"
            assert info.namespace == "spark"
            assert info.state == SparkConnectState.PROVISIONING

        elif test_case.name == "failed status":
            assert info.state == SparkConnectState.FAILED

        elif test_case.name == "running status":
            assert info.state == SparkConnectState.RUNNING
            assert info.service_name == "run-session-svc"

        elif test_case.name == "empty status":
            assert info.state == SparkConnectState.PROVISIONING
            assert info.driver_pod_name is None

    else:
        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            get_spark_connect_info_from_cr(spark_connect_cr)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="generate unique job name",
            expected_status=SUCCESS,
            expected_output="spark-job-",
        ),
        TestCase(
            name="generate different job names",
            expected_status=SUCCESS,
            expected_output=10,
        ),
    ],
)
def test_generate_job_name(test_case: TestCase) -> None:
    """Tests generate_job_name."""

    print("Executing test:", test_case.name)

    assert test_case.expected_status == SUCCESS

    if test_case.name == "generate unique job name":
        name = generate_job_name()

        assert name.startswith(test_case.expected_output)
        assert len(name) > len(test_case.expected_output)

    elif test_case.name == "generate different job names":
        names = {generate_job_name() for _ in range(10)}

        assert len(names) == test_case.expected_output

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid cpu values",
            expected_status=SUCCESS,
            config={
                "cases": [
                    ("1", 1),
                    ("4", 4),
                    ("1.5", 2),
                    ("500m", 1),
                    ("1500m", 2),
                    ("2500m", 3),
                    (" 1500m ", 2),
                    (2, 2),
                    (16, 16),
                ],
            },
        ),
        TestCase(
            name="invalid cpu values",
            expected_status=FAILED,
            config={
                "cases": [
                    None,
                    "",
                    " ",
                    "abc",
                    "50O0m",
                    "0",
                    "-1",
                    "-500m",
                    "1.5m",
                    "nan",
                    "inf",
                    0,
                    -1,
                    2048,
                ],
            },
            expected_error=ValueError,
        ),
    ],
)
def test_validate_cpu_value(test_case: TestCase) -> None:
    """Tests _validate_cpu_value."""

    print("Executing test:", test_case.name)

    if test_case.expected_status == SUCCESS:
        for cpu, expected in test_case.config["cases"]:
            assert _validate_cpu_value(cpu) == expected

    else:
        for cpu in test_case.config["cases"]:
            with pytest.raises(test_case.expected_error):
                _validate_cpu_value(cpu)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default driver resources",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="custom driver resources",
            expected_status=SUCCESS,
            config={
                "driver": Driver(
                    resources={
                        "cpu": "2",
                        "memory": "4Gi",
                    },
                ),
            },
        ),
        TestCase(
            name="fractional driver memory",
            expected_status=SUCCESS,
            config={
                "driver": Driver(
                    resources={
                        "cpu": "2",
                        "memory": "1.5Gi",
                    },
                ),
            },
        ),
    ],
)
def test_resolve_driver_resources(test_case: TestCase) -> None:
    """Tests _resolve_driver_resources."""

    print("Executing test:", test_case.name)

    cores, memory = _resolve_driver_resources(
        test_case.config.get("driver"),
    )

    assert test_case.expected_status == SUCCESS

    if test_case.name == "default driver resources":
        assert cores == constants.DEFAULT_DRIVER_CPU
        assert memory == _memory_kubernetes_to_spark(
            constants.DEFAULT_DRIVER_MEMORY,
        )

    elif test_case.name == "custom driver resources":
        assert cores == 2
        assert memory == "4g"

    elif test_case.name == "fractional driver memory":
        assert cores == 2
        assert memory == "1536m"

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default executor resources",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="simple executor parameters",
            expected_status=SUCCESS,
            config={
                "num_executors": 3,
                "resources_per_executor": {
                    "cpu": "2",
                    "memory": "4Gi",
                },
            },
        ),
        TestCase(
            name="executor configuration precedence",
            expected_status=SUCCESS,
            config={
                "executor": Executor(
                    num_instances=5,
                    resources_per_executor={
                        "cpu": "8",
                        "memory": "16Gi",
                    },
                ),
                "num_executors": 2,
                "resources_per_executor": {
                    "cpu": "4",
                    "memory": "8Gi",
                },
            },
        ),
        TestCase(
            name="fractional executor memory",
            expected_status=SUCCESS,
            config={
                "resources_per_executor": {
                    "cpu": "2",
                    "memory": "1.5Gi",
                },
            },
        ),
    ],
)
def test_resolve_executor_resources(test_case: TestCase) -> None:
    """Tests _resolve_executor_resources."""

    print("Executing test:", test_case.name)

    instances, cores, memory = _resolve_executor_resources(
        executor=test_case.config.get("executor"),
        num_executors=test_case.config.get("num_executors"),
        resources_per_executor=test_case.config.get("resources_per_executor"),
    )

    assert test_case.expected_status == SUCCESS

    if test_case.name == "default executor resources":
        assert instances == constants.DEFAULT_NUM_EXECUTORS
        assert cores == constants.DEFAULT_EXECUTOR_CPU
        assert memory == _memory_kubernetes_to_spark(
            constants.DEFAULT_EXECUTOR_MEMORY,
        )

    elif test_case.name == "simple executor parameters":
        assert instances == 3
        assert cores == 2
        assert memory == "4g"

    elif test_case.name == "executor configuration precedence":
        assert instances == 5
        assert cores == 8
        assert memory == "16g"

    elif test_case.name == "fractional executor memory":
        assert instances == constants.DEFAULT_NUM_EXECUTORS
        assert cores == 2
        assert memory == "1536m"

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="read logs",
            expected_status=SUCCESS,
            config={
                "follow": False,
            },
            expected_output=[
                "log line 1",
                "log line 2",
            ],
        ),
        TestCase(
            name="follow logs",
            expected_status=SUCCESS,
            config={
                "follow": True,
            },
            expected_output=[
                "log line 1",
                "log line 2",
            ],
        ),
        TestCase(
            name="timeout",
            expected_status=FAILED,
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error",
            expected_status=FAILED,
            expected_error=RuntimeError,
        ),
    ],
)
def test_read_pod_logs(test_case: TestCase) -> None:
    """Tests read_pod_logs."""

    print("Executing test:", test_case.name)

    core_api = Mock()
    thread = Mock()

    if test_case.name == "read logs":
        thread.get.return_value = "log line 1\nlog line 2"

    elif test_case.name == "follow logs":
        stream = Mock()
        stream.stream.return_value = iter(
            [
                b"log line 1\n",
                b"log line 2\n",
            ]
        )
        thread.get.return_value = stream

    elif test_case.name == "timeout":
        thread.get.side_effect = multiprocessing.TimeoutError()

    elif test_case.name == "runtime error":
        thread.get.side_effect = RuntimeError()

    core_api.read_namespaced_pod_log.return_value = thread

    if test_case.expected_status == SUCCESS:
        logs = list(
            read_pod_logs(
                core_api=core_api,
                namespace="default",
                pod_name="driver-pod",
                follow=test_case.config["follow"],
            )
        )

        assert logs == test_case.expected_output

        if test_case.name == "read logs":
            core_api.read_namespaced_pod_log.assert_called_once_with(
                name="driver-pod",
                namespace="default",
                async_req=True,
            )

        elif test_case.name == "follow logs":
            core_api.read_namespaced_pod_log.assert_called_once_with(
                name="driver-pod",
                namespace="default",
                follow=True,
                _preload_content=False,
                async_req=True,
            )

    else:
        with pytest.raises(test_case.expected_error):
            list(
                read_pod_logs(
                    core_api=core_api,
                    namespace="default",
                    pod_name="driver-pod",
                )
            )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default spark job driver spec",
            expected_status=SUCCESS,
            config={},
        ),
    ],
)
def test_get_spark_job_driver_spec(test_case: TestCase) -> None:
    """Tests get_spark_job_driver_spec."""

    print("Executing test:", test_case.name)

    spec = get_spark_job_driver_spec()

    assert test_case.expected_status == SUCCESS

    assert spec.cores == constants.DEFAULT_DRIVER_CPU
    assert spec.memory == _memory_kubernetes_to_spark(
        constants.DEFAULT_DRIVER_MEMORY,
    )
    assert spec.service_account == constants.DEFAULT_SERVICE_ACCOUNT

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="build function job init container",
            expected_status=SUCCESS,
            config={
                "command": [
                    "bash",
                    "-c",
                    "printf 'print(\"hello\")' > /opt/spark/app/main.py",
                ],
            },
        ),
    ],
)
def test_get_func_job_init_container(test_case: TestCase) -> None:
    """Tests get_func_job_init_container."""

    print("Executing test:", test_case.name)

    container = get_func_job_init_container(
        test_case.config["command"],
    )

    assert test_case.expected_status == SUCCESS

    assert container.name == constants.FUNC_JOB_INIT_CONTAINER_NAME
    assert container.image == constants.DEFAULT_SPARK_IMAGE

    assert container.command == test_case.config["command"]

    assert container.volume_mounts is not None
    assert len(container.volume_mounts) == 1
    assert container.volume_mounts[0].name == constants.FUNC_JOB_VOLUME_NAME
    assert container.volume_mounts[0].mount_path == constants.FUNC_JOB_SCRIPT_DIR

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="build script without args",
            expected_status=SUCCESS,
            config={
                "func": sample_function,
                "func_args": None,
            },
        ),
        TestCase(
            name="build script with args",
            expected_status=SUCCESS,
            config={
                "func": sample_function_with_args,
                "func_args": {
                    "name": "Alice",
                    "age": 20,
                },
            },
        ),
        TestCase(
            name="non callable",
            expected_status=FAILED,
            config={
                "func": "not_a_function",
                "func_args": None,
            },
            expected_error=ValueError,
            expected_output="Expected a callable function",
        ),
    ],
)
def test_get_command_using_spark_func(test_case: TestCase) -> None:
    """Tests get_command_using_spark_func."""

    print("Executing test:", test_case.name)

    if test_case.expected_status == SUCCESS:
        command = get_command_using_spark_func(
            test_case.config["func"],
            test_case.config["func_args"],
        )

        assert command[0] == "bash"
        assert command[1] == "-c"

        shell_script = command[2]

        if test_case.name == "build script without args":
            assert "def sample_function" in shell_script
            assert 'print("hello")' in shell_script
            assert "sample_function()" in shell_script

        elif test_case.name == "build script with args":
            assert "def sample_function_with_args" in shell_script
            assert "sample_function_with_args(**" in shell_script
            assert "'name': 'Alice'" in shell_script
            assert "'age': 20" in shell_script
    else:
        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            get_command_using_spark_func(
                test_case.config["func"],
                test_case.config["func_args"],
            )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="default spark job executor spec",
            expected_status=SUCCESS,
            config={},
        ),
    ],
)
def test_get_spark_job_executor_spec(test_case: TestCase) -> None:
    """Tests get_spark_job_executor_spec."""

    print("Executing test:", test_case.name)

    spec = get_spark_job_executor_spec()

    assert test_case.expected_status == SUCCESS

    assert spec.cores == constants.DEFAULT_EXECUTOR_CPU
    assert spec.memory == _memory_kubernetes_to_spark(
        constants.DEFAULT_EXECUTOR_MEMORY,
    )
    assert spec.instances == constants.DEFAULT_NUM_EXECUTORS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="build spark application for remote uri job",
            expected_status=SUCCESS,
            config={
                "name": "test-job",
                "namespace": "default",
                "main_file": "s3://bucket/job.py",
                "arguments": ["--date", "2026-06-30"],
                "num_executors": 3,
                "resources_per_executor": {
                    "cpu": "2",
                    "memory": "4Gi",
                },
            },
        ),
    ],
)
def test_get_spark_application_cr_from_file_job(test_case: TestCase) -> None:
    """Tests build_spark_application_cr."""

    print("Executing test:", test_case.name)

    app = get_spark_application_cr_from_file_job(
        name=test_case.config["name"],
        namespace=test_case.config["namespace"],
        main_file=test_case.config["main_file"],
        arguments=test_case.config["arguments"],
        num_executors=test_case.config["num_executors"],
        resources_per_executor=test_case.config["resources_per_executor"],
    )

    assert test_case.expected_status == SUCCESS

    assert app.metadata.name == test_case.config["name"]
    assert app.metadata.namespace == test_case.config["namespace"]

    assert app.spec.main_application_file == test_case.config["main_file"]
    assert app.spec.arguments == test_case.config["arguments"]

    assert app.spec.driver.cores == constants.DEFAULT_DRIVER_CPU
    assert app.spec.driver.memory == _memory_kubernetes_to_spark(
        constants.DEFAULT_DRIVER_MEMORY,
    )
    assert app.spec.driver.service_account == constants.DEFAULT_SERVICE_ACCOUNT

    assert app.spec.executor.instances == test_case.config["num_executors"]
    assert app.spec.executor.cores == 2
    assert app.spec.executor.memory == _memory_kubernetes_to_spark(
        "4Gi",
    )

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="build spark application for function job",
            expected_status=SUCCESS,
            config={
                "name": "test-job",
                "namespace": "default",
                "func": sample_function,
                "func_args": None,
                "num_executors": 3,
                "resources_per_executor": {
                    "cpu": "2",
                    "memory": "4Gi",
                },
            },
        ),
    ],
)
def test_get_spark_application_cr_from_func_job(
    test_case: TestCase,
) -> None:
    """Tests get_spark_application_cr_from_func_job."""

    print("Executing test:", test_case.name)

    app = get_spark_application_cr_from_func_job(
        name=test_case.config["name"],
        namespace=test_case.config["namespace"],
        func=test_case.config["func"],
        func_args=test_case.config["func_args"],
        num_executors=test_case.config["num_executors"],
        resources_per_executor=test_case.config["resources_per_executor"],
    )

    assert test_case.expected_status == SUCCESS

    assert app.metadata.name == test_case.config["name"]
    assert app.metadata.namespace == test_case.config["namespace"]

    assert app.spec.main_application_file == constants.FUNC_JOB_MAIN_FILE

    assert app.spec.driver.init_containers is not None
    assert len(app.spec.driver.init_containers) == 1
    assert app.spec.driver.init_containers[0].name == constants.FUNC_JOB_INIT_CONTAINER_NAME

    assert app.spec.driver.volume_mounts is not None
    assert len(app.spec.driver.volume_mounts) == 1
    assert app.spec.driver.volume_mounts[0].name == constants.FUNC_JOB_VOLUME_NAME

    assert app.spec.volumes is not None
    assert len(app.spec.volumes) == 1
    assert app.spec.volumes[0].name == constants.FUNC_JOB_VOLUME_NAME

    assert app.spec.executor.instances == 3
    assert app.spec.executor.cores == 2
    assert app.spec.executor.memory == "4g"

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="submitted status",
            expected_status=SUCCESS,
            config={
                "spark_state": "SUBMITTED",
                "job_status": SparkJobStatus.CREATED,
            },
        ),
        TestCase(
            name="running status",
            expected_status=SUCCESS,
            config={
                "spark_state": "RUNNING",
                "job_status": SparkJobStatus.RUNNING,
            },
        ),
        TestCase(
            name="succeeding status",
            expected_status=SUCCESS,
            config={
                "spark_state": "SUCCEEDING",
                "job_status": SparkJobStatus.RUNNING,
            },
        ),
        TestCase(
            name="suspending status",
            expected_status=SUCCESS,
            config={
                "spark_state": "SUSPENDING",
                "job_status": SparkJobStatus.RUNNING,
            },
        ),
        TestCase(
            name="suspended status",
            expected_status=SUCCESS,
            config={
                "spark_state": "SUSPENDED",
                "job_status": SparkJobStatus.RUNNING,
            },
        ),
        TestCase(
            name="resuming status",
            expected_status=SUCCESS,
            config={
                "spark_state": "RESUMING",
                "job_status": SparkJobStatus.RUNNING,
            },
        ),
        TestCase(
            name="completed status",
            expected_status=SUCCESS,
            config={
                "spark_state": "COMPLETED",
                "job_status": SparkJobStatus.COMPLETED,
            },
        ),
        TestCase(
            name="failed status",
            expected_status=SUCCESS,
            config={
                "spark_state": "FAILED",
                "job_status": SparkJobStatus.FAILED,
            },
        ),
        TestCase(
            name="submission failed status",
            expected_status=SUCCESS,
            config={
                "spark_state": "SUBMISSION_FAILED",
                "job_status": SparkJobStatus.FAILED,
            },
        ),
        TestCase(
            name="failing status",
            expected_status=SUCCESS,
            config={
                "spark_state": "FAILING",
                "job_status": SparkJobStatus.FAILED,
            },
        ),
        TestCase(
            name="pending rerun status",
            expected_status=SUCCESS,
            config={
                "spark_state": "PENDING_RERUN",
                "job_status": SparkJobStatus.FAILED,
            },
        ),
        TestCase(
            name="invalidating status",
            expected_status=SUCCESS,
            config={
                "spark_state": "INVALIDATING",
                "job_status": SparkJobStatus.FAILED,
            },
        ),
        TestCase(
            name="without status",
            expected_status=SUCCESS,
            config={
                "without_status": True,
                "job_name": "new-job",
            },
        ),
        TestCase(
            name="uses from operator state",
            expected_status=SUCCESS,
            config={
                "spark_state": "RUNNING",
                "patch_status": SparkJobStatus.RUNNING,
            },
        ),
        TestCase(
            name="invalid metadata",
            expected_status=FAILED,
            config={
                "invalid_metadata": True,
            },
            expected_error=ValueError,
            expected_output="SparkApplication CR is invalid",
        ),
    ],
)
def test_get_spark_application_info_from_cr(
    test_case: TestCase,
    spark_application_spec,
) -> None:
    """Tests get_spark_application_info_from_cr."""

    print("Executing test:", test_case.name)

    creation_timestamp = datetime.now()

    if test_case.expected_status == FAILED:
        spark_app = models.SparkV1beta2SparkApplication.model_construct(
            metadata=None,
            spec=spark_application_spec,
        )

        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            get_spark_application_info_from_cr(spark_app)

    elif test_case.name == "without status":
        spark_app = models.SparkV1beta2SparkApplication(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name=test_case.config["job_name"],
                namespace="default",
                creation_timestamp=creation_timestamp,
            ),
            spec=spark_application_spec,
        )

        job = get_spark_application_info_from_cr(spark_app)

        assert job.name == test_case.config["job_name"]
        assert job.namespace == "default"
        assert job.status == SparkJobStatus.CREATED
        assert job.driver_pod_name is None
        assert job.creation_timestamp == creation_timestamp
        assert job.num_executors == 5

    elif test_case.name == "uses from operator state":
        spark_app = models.SparkV1beta2SparkApplication(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="test-job",
                namespace="default",
                creation_timestamp=creation_timestamp,
            ),
            spec=spark_application_spec,
            status=models.SparkV1beta2SparkApplicationStatus(
                application_state=models.SparkV1beta2ApplicationState(
                    state=test_case.config["spark_state"],
                ),
                driver_info=models.SparkV1beta2DriverInfo(
                    pod_name="test-driver",
                ),
            ),
        )

        with patch.object(
            SparkJobStatus,
            "from_operator_state",
            return_value=test_case.config["patch_status"],
        ) as mock_from_operator_state:
            job = get_spark_application_info_from_cr(spark_app)

        mock_from_operator_state.assert_called_once_with(
            test_case.config["spark_state"],
        )

        assert job.name == "test-job"
        assert job.namespace == "default"
        assert job.status == test_case.config["patch_status"]
        assert job.driver_pod_name == "test-driver"
        assert job.creation_timestamp == creation_timestamp
        assert job.num_executors == 5

    else:
        spark_app = models.SparkV1beta2SparkApplication(
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name="test-job",
                namespace="default",
                creation_timestamp=creation_timestamp,
            ),
            spec=spark_application_spec,
            status=models.SparkV1beta2SparkApplicationStatus(
                application_state=models.SparkV1beta2ApplicationState(
                    state=test_case.config["spark_state"],
                ),
                driver_info=models.SparkV1beta2DriverInfo(
                    pod_name="test-driver",
                ),
            ),
        )

        job = get_spark_application_info_from_cr(spark_app)

        assert job.name == "test-job"
        assert job.namespace == "default"
        assert job.status == test_case.config["job_status"]
        assert job.driver_pod_name == "test-driver"
        assert job.creation_timestamp == creation_timestamp
        assert job.num_executors == 5

    print("test execution complete")
>>>>>>> upstream/main
