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

import logging

from kubernetes import client

try:
    from kube_authkit import AuthConfig
    from kube_authkit import get_k8s_client as kube_authkit_get_client

    KUBE_AUTHKIT_AVAILABLE = True
except ImportError:
    KUBE_AUTHKIT_AVAILABLE = False

from kubeflow.common.types import KubernetesBackendConfig

logger = logging.getLogger(__name__)


def get_kubernetes_client(cfg: KubernetesBackendConfig) -> client.ApiClient:
    """
    Get Kubernetes API client using kube-authkit or legacy configuration.

    Priority order:
    1. client_configuration (direct override) - highest priority
    2. auth_method (explicit kube-authkit method)
    3. config_file/context (legacy, mapped to kubeconfig method with deprecation warning)
    4. auto-detection (default)

    Args:
        cfg: KubernetesBackendConfig with authentication parameters

    Returns:
        kubernetes.client.ApiClient instance

    Raises:
        ImportError: If kube-authkit is not available
        ValueError: If invalid configuration is provided
    """
    if not KUBE_AUTHKIT_AVAILABLE:
        raise ImportError(
            "kube-authkit is required for authentication. "
            "Install it with: pip install kube-authkit"
        )

    # Priority 1: Use client_configuration if provided (bypass kube-authkit)
    if cfg.client_configuration is not None:
        logger.debug("Using provided client_configuration for authentication")
        return client.ApiClient(cfg.client_configuration)

    # Build AuthConfig for kube-authkit
    auth_config_params = {
        "verify_ssl": cfg.verify_ssl,
    }

    # Add core parameters if provided
    if cfg.k8s_api_host is not None:
        auth_config_params["k8s_api_host"] = cfg.k8s_api_host

    if cfg.kubeconfig_path is not None:
        auth_config_params["kubeconfig_path"] = cfg.kubeconfig_path

    if cfg.ca_cert is not None:
        auth_config_params["ca_cert"] = cfg.ca_cert

    # Priority 2: Explicit auth_method
    if cfg.auth_method is not None:
        logger.debug(f"Using explicit auth_method: {cfg.auth_method}")
        auth_config_params["method"] = cfg.auth_method

        # Add OIDC parameters if using OIDC
        if cfg.auth_method == "oidc":
            auth_config_params["oidc_issuer"] = cfg.oidc_issuer
            auth_config_params["client_id"] = cfg.client_id
            auth_config_params["client_secret"] = cfg.client_secret
            auth_config_params["use_device_flow"] = cfg.use_device_flow
            auth_config_params["oidc_callback_port"] = cfg.oidc_callback_port

            if cfg.scopes is not None:
                auth_config_params["scopes"] = cfg.scopes

        # Add token-based authentication (OpenShift, etc.)
        if cfg.auth_method == "openshift":
            if cfg.token is not None:
                auth_config_params["token"] = cfg.token

        auth_config_params["use_keyring"] = cfg.use_keyring

    # Priority 3: Legacy config_file or context (with deprecation warning)
    elif cfg.config_file is not None or cfg.context is not None:
        logger.warning(
            "The 'config_file' and 'context' parameters are deprecated and will be removed in "
            "a future version. Use 'kubeconfig_path' and 'auth_method=\"kubeconfig\"' instead."
        )
        auth_config_params["method"] = "kubeconfig"

        # Map legacy config_file to kubeconfig_path
        if cfg.config_file is not None:
            auth_config_params["kubeconfig_path"] = cfg.config_file

    # Priority 4: Auto-detection (default)
    else:
        logger.debug("Using auto-detection for authentication method")
        auth_config_params["method"] = "auto"

    # Create AuthConfig and get client
    auth_config = AuthConfig(**auth_config_params)
    api_client = kube_authkit_get_client(auth_config)

    logger.debug("Successfully authenticated with kube-authkit")
    return api_client
