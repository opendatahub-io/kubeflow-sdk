# Copyright 2024 The Kubeflow Authors.
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

"""Experimental Kubernetes backend with instrumentation support."""

import inspect
import logging
import os
import textwrap
from typing import Optional, Union

from kubernetes import client

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer.backends.kubernetes import utils
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.trainer.constants import constants
from kubeflow.trainer.constants.constants import (
    ANNOTATION_FRAMEWORK,
    ANNOTATION_METRICS_POLL_INTERVAL,
    ANNOTATION_METRICS_PORT,
    ANNOTATION_PROGRESSION_TRACKING,
)
from kubeflow.trainer.experimental.instrumentation import get_transformers_trainer_wrapper_script
from kubeflow.trainer.options import kubernetes as k8s_options
from kubeflow.trainer.types import types
from kubeflow.trainer.types.experimental import TransformersTrainer

logger = logging.getLogger(__name__)


class ExperimentalKubernetesBackend(KubernetesBackend):
    """Kubernetes backend with experimental instrumentation support.

    EXPERIMENTAL: This API may change in future releases.
    """

    def __init__(
        self,
        cfg: Optional[KubernetesBackendConfig] = None,
        custom_api: Optional[client.CustomObjectsApi] = None,
        core_api: Optional[client.CoreV1Api] = None,
    ):
        """Initialize experimental backend with optional pre-existing API clients.

        Args:
            cfg: Backend configuration. If None, uses default KubernetesBackendConfig.
            custom_api: Optional pre-initialized CustomObjectsApi client to reuse.
            core_api: Optional pre-initialized CoreV1Api client to reuse.
        """
        if cfg is None:
            cfg = KubernetesBackendConfig()
        super().__init__(cfg)

        if custom_api is not None:
            self.custom_api = custom_api
        if core_api is not None:
            self.core_api = core_api

    def train(
        self,
        runtime: Optional[types.Runtime] = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[
            Union[
                types.CustomTrainer,
                types.CustomTrainerContainer,
                types.BuiltinTrainer,
                TransformersTrainer,
            ]
        ] = None,
        options: Optional[list] = None,
    ) -> str:
        """Train with experimental trainer support."""
        if trainer is None:
            raise ValueError("Trainer cannot be None")

        # Only instrument TransformersTrainer if progression tracking is enabled
        if isinstance(trainer, TransformersTrainer) and trainer.enable_progression_tracking:
            return self._train_transformers_instrumented(
                runtime=runtime,
                initializer=initializer,
                trainer=trainer,
                options=options,
            )

        return super().train(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            options=options,
        )

    def _get_instrumented_command(
        self,
        runtime: types.Runtime,
        trainer: TransformersTrainer,
        wrapper_script: str,
    ) -> list[str]:
        """Generate command with instrumentation wrapper injected.

        This follows the same pattern as get_command_using_train_func but injects
        the wrapper script before the user function call.
        """
        if not runtime.trainer:
            raise ValueError(f"Runtime must have a trainer: {runtime}")

        if not callable(trainer.func):
            raise ValueError(
                f"Training function must be callable, got function type: {type(trainer.func)}"
            )

        # Extract user function code
        func_code = inspect.getsource(trainer.func)
        func_file = os.path.basename(inspect.getfile(trainer.func))
        func_code = textwrap.dedent(func_code)

        # Build function call
        if trainer.func_args is None:
            func_call = f"{trainer.func.__name__}()"
        else:
            func_call = f"{trainer.func.__name__}(**{trainer.func_args})"

        # Inject wrapper before user function call
        # The wrapper script has {{user_func_import_and_call}} placeholder
        user_code = f"{func_code}\n{func_call}\n"
        full_code = wrapper_script.replace("{{user_func_import_and_call}}", user_code)

        is_mpi = runtime.trainer.command[0] == "mpirun"
        if is_mpi:
            func_file = os.path.join(constants.DEFAULT_MPI_USER_HOME, func_file)

        # Install packages if needed
        install_packages = ""
        if trainer.packages_to_install:
            install_packages = utils.get_script_for_python_packages(
                trainer.packages_to_install,
                trainer.pip_index_urls,
                is_mpi,
            )

        # Build final command
        command = []
        for c in runtime.trainer.command:
            if "{func_file}" in c:
                exec_script = c.format(func_code=full_code, func_file=func_file)
                if install_packages:
                    exec_script = install_packages + exec_script
                command.append(exec_script)
            else:
                command.append(c)

        return command

    def _train_transformers_instrumented(
        self,
        runtime: Optional[types.Runtime],
        initializer: Optional[types.Initializer],
        trainer: TransformersTrainer,
        options: Optional[list],
    ) -> str:
        """Train with Transformers instrumentation.

        Generates instrumented wrapper script and submits job with progression tracking.

        Args:
            runtime: Training runtime configuration.
            initializer: Dataset/model initializer configuration.
            trainer: TransformersTrainer with instrumentation settings.
            options: Additional job configuration options.

        Returns:
            Name of the created TrainJob.
        """
        if options is None:
            options = []

        # Add progression tracking annotations
        progression_annotations = {
            ANNOTATION_PROGRESSION_TRACKING: "enabled",
            ANNOTATION_METRICS_PORT: str(trainer.metrics_port),
            ANNOTATION_METRICS_POLL_INTERVAL: str(trainer.metrics_poll_interval_seconds),
            ANNOTATION_FRAMEWORK: "transformers",
        }

        options.append(k8s_options.Annotations(progression_annotations))

        # Call parent train() - instrumented command will be injected in _get_trainjob_spec
        return super().train(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            options=options,
        )

    def _get_trainjob_spec(
        self,
        runtime: Optional[types.Runtime] = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[
            Union[
                types.CustomTrainer,
                types.CustomTrainerContainer,
                types.BuiltinTrainer,
                TransformersTrainer,
            ]
        ] = None,
        trainer_overrides: Optional[dict] = None,
        spec_labels: Optional[dict[str, str]] = None,
        spec_annotations: Optional[dict[str, str]] = None,
        pod_template_overrides: Optional[dict] = None,
    ):
        """Override to inject instrumented command for TransformersTrainer.

        For TransformersTrainer, generates wrapper script with instrumentation
        callbacks and injects it into the trainer command.

        Args:
            runtime: Training runtime configuration.
            initializer: Dataset/model initializer configuration.
            trainer: Trainer configuration (CustomTrainer, BuiltinTrainer, etc.).
            trainer_overrides: Dict of trainer spec overrides.
            spec_labels: Labels for the TrainJob spec.
            spec_annotations: Annotations for the TrainJob spec.
            pod_template_overrides: Pod template overrides.

        Returns:
            TrainJob spec with instrumented command for TransformersTrainer.
        """
        spec = super()._get_trainjob_spec(
            runtime=runtime,
            initializer=initializer,
            trainer=trainer,
            trainer_overrides=trainer_overrides,
            spec_labels=spec_labels,
            spec_annotations=spec_annotations,
            pod_template_overrides=pod_template_overrides,
        )

        # Only inject instrumentation if TransformersTrainer has progression tracking enabled
        if (
            isinstance(trainer, TransformersTrainer)
            and trainer.enable_progression_tracking
            and spec.trainer
            and runtime
        ):
            wrapper_script = get_transformers_trainer_wrapper_script(
                metrics_port=trainer.metrics_port,
                custom_metrics=trainer.custom_metrics or {},
            )
            instrumented_command = self._get_instrumented_command(runtime, trainer, wrapper_script)
            spec.trainer.command = instrumented_command

        return spec
