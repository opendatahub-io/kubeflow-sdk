API Reference
=============

TrainerClient
-------------

.. autoclass:: kubeflow.trainer.TrainerClient
   :members:
   :show-inheritance:

Trainers
--------

.. autoclass:: kubeflow.trainer.CustomTrainer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.CustomTrainerContainer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.BuiltinTrainer
   :members:
   :show-inheritance:

Initializers
------------

.. autoclass:: kubeflow.trainer.Initializer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.HuggingFaceDatasetInitializer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.S3DatasetInitializer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.DataCacheInitializer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.HuggingFaceModelInitializer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.S3ModelInitializer
   :members:
   :show-inheritance:

Backend Configurations
----------------------

.. autoclass:: kubeflow.trainer.KubernetesBackendConfig
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.LocalProcessBackendConfig
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.ContainerBackendConfig
   :members:
   :show-inheritance:

Utilities
---------

.. autofunction:: kubeflow.trainer.backends.kubernetes.utils.update_trainjob_status

RHAI Speculator Training
------------------------

.. autoclass:: kubeflow.trainer.rhai.SpeculativeDecodingTrainer
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.rhai.SpeculatorConfig
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.rhai.VLLMSpeculativeConfig
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.rhai.VLLMEngineConfig
   :members:
   :show-inheritance:

.. autoclass:: kubeflow.trainer.rhai.SpeculatorVLLMConfig
   :members:
   :show-inheritance:

Configure speculative decoding separately from general vLLM engine arguments:

.. code-block:: python

   from kubeflow.trainer.rhai import (
       SpeculatorConfig,
       VLLMEngineConfig,
       VLLMSpeculativeConfig,
   )

   config = SpeculatorConfig(
       target_layer_ids=[2, 14, 25, 28],
       vllm_speculative=VLLMSpeculativeConfig(
           num_speculative_tokens=2,
           enforce_eager=True,
       ),
       vllm_engine=VLLMEngineConfig(
           max_model_len=2048,
           dtype="bfloat16",
           max_num_seqs=3,
           extra_args={"max_num_batched_tokens": "4096"},
       ),
   )

The SDK sends these settings to the managed sidecar in separate
``speculative_config`` and ``engine_args`` objects within
``SPECULATOR_VLLM_EXTRA_ARGS``. The selected runtime's sidecar launcher must merge the
speculative settings into its ``--speculative-config`` JSON and translate engine settings
to ``vllm serve`` flags. The SDK RuntimePatch does not replace the runtime's launcher.
