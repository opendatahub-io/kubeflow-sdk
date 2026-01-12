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

from typing import Optional

from kubernetes import client
from pydantic import BaseModel


class KubernetesBackendConfig(BaseModel):
    # Existing fields (maintain for backward compatibility)
    namespace: Optional[str] = None
    config_file: Optional[str] = None  # LEGACY - will log deprecation warning
    context: Optional[str] = None  # LEGACY - will log deprecation warning
    client_configuration: Optional[client.Configuration] = None

    # Core kube-authkit fields
    auth_method: Optional[str] = None  # "auto", "kubeconfig", "incluster", "oidc", "openshift"
    k8s_api_host: Optional[str] = None  # Kubernetes API server URL (required for OpenShift)
    kubeconfig_path: Optional[str] = None  # Path to kubeconfig file

    # OIDC authentication fields
    oidc_issuer: Optional[str] = None
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    scopes: Optional[list] = None  # OIDC scopes (default: ["openid"])
    use_device_flow: bool = False
    oidc_callback_port: int = 8080  # OAuth callback port

    # OpenShift OAuth fields
    openshift_token: Optional[str] = None  # OpenShift OAuth token (for token-based auth)

    # Advanced options
    use_keyring: bool = False  # Persist tokens in system keyring
    verify_ssl: bool = True  # Verify SSL certificates
    ca_cert: Optional[str] = None  # Path to custom CA certificate

    class Config:
        arbitrary_types_allowed = True
