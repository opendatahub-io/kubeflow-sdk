# Authentication Guide

This guide covers all authentication methods supported by the Kubeflow SDK through the `kube-authkit` library.

## Table of Contents

- [Overview](#overview)
- [Authentication Methods](#authentication-methods)
  - [Auto-Detection](#auto-detection)
  - [Kubeconfig](#kubeconfig)
  - [In-Cluster (Service Account)](#in-cluster-service-account)
  - [OIDC](#oidc)
  - [OpenShift OAuth](#openshift-oauth)
- [Configuration Options](#configuration-options)
- [Environment Variables](#environment-variables)
- [Token Management](#token-management)
- [Migration Guide](#migration-guide)
- [Troubleshooting](#troubleshooting)

## Overview

The Kubeflow SDK uses `kube-authkit` to provide flexible, secure authentication across various Kubernetes and OpenShift environments. The SDK automatically detects the best authentication method for your environment, but you can also explicitly configure the method that works best for your use case.

### Supported Authentication Methods

- **Auto-Detection**: Automatically selects the best method (recommended)
- **Kubeconfig**: Uses your local `~/.kube/config` file
- **In-Cluster**: Uses service account tokens when running inside a pod
- **OIDC**: OpenID Connect authentication with support for:
  - Authorization Code Flow with PKCE
  - Device Code Flow (for CLI/headless environments)
  - Client Credentials Flow (for service-to-service)
- **OpenShift OAuth**: Native OpenShift authentication

## Authentication Methods

### Auto-Detection

**Recommended for most users.** The SDK automatically detects whether you're running inside a Kubernetes cluster or externally, and selects the appropriate authentication method.

```python
from kubeflow.trainer import TrainerClient

# Auto-detects: kubeconfig if outside cluster, service account if inside
client = TrainerClient()
```

**How it works:**
1. Checks if running inside a Kubernetes pod (looks for `/var/run/secrets/kubernetes.io/`)
2. If inside: Uses in-cluster service account authentication
3. If outside: Uses kubeconfig from `~/.kube/config` or `$KUBECONFIG`

### Kubeconfig

Use kubeconfig-based authentication for local development or when managing multiple clusters.

#### Basic Usage

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="kubeconfig"
    )
)
```

#### Using Different Kubeconfig Files

Set the `KUBECONFIG` environment variable to use a different kubeconfig file:

```python
import os

os.environ["KUBECONFIG"] = "/path/to/custom/kubeconfig"

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="kubeconfig"
    )
)
```

#### Selecting a Specific Context

To use a specific context from your kubeconfig, switch contexts before creating the client:

```bash
kubectl config use-context my-cluster-context
```

Then create the client as usual:

```python
client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="kubeconfig"
    )
)
```

### In-Cluster (Service Account)

When running inside a Kubernetes pod, use the pod's service account for authentication.

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="incluster"
    )
)
```

**Requirements:**
- Must be running inside a Kubernetes pod
- Service account must have appropriate RBAC permissions
- Service account token available at `/var/run/secrets/kubernetes.io/serviceaccount/token`

**Example Pod Manifest:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kubeflow-sdk-pod
spec:
  serviceAccountName: kubeflow-sdk-sa
  containers:
  - name: app
    image: my-app:latest
    command: ["python", "train.py"]
```

### OIDC

OpenID Connect (OIDC) authentication enables integration with corporate identity providers.

#### Authorization Code Flow with PKCE

Best for interactive applications:

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="oidc",
        oidc_issuer="https://your-issuer.example.com",
        client_id="your-client-id",
        client_secret="your-client-secret",
        use_device_flow=False,
        use_keyring=True,  # Persist tokens across sessions
    )
)
```

#### Device Code Flow

Best for CLI tools and headless environments:

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="oidc",
        oidc_issuer="https://your-issuer.example.com",
        client_id="your-client-id",
        client_secret="your-client-secret",
        use_device_flow=True,  # Enable device code flow
        use_keyring=True,
    )
)
```

**Device Code Flow:**
1. SDK displays a verification URL and code
2. User opens URL in a browser and enters the code
3. SDK polls for authentication completion
4. Access token is stored and reused

#### Client Credentials Flow

Best for service-to-service authentication:

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="oidc",
        oidc_issuer="https://your-issuer.example.com",
        client_id="service-account-id",
        client_secret="service-account-secret",
    )
)
```

### OpenShift OAuth

OpenShift supports multiple authentication methods including token-based and interactive OAuth flow.

#### Option 1: Token-Based Authentication (Recommended for CI/CD)

Use an existing OpenShift token:

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="openshift",
        k8s_api_host="https://api.cluster.example.com:6443",
        token="sha256~your-token-here"
    )
)
```

#### Option 2: Environment Variable

```bash
export AUTHKIT_TOKEN="sha256~your-token-here"
export AUTHKIT_K8S_API_HOST="https://api.cluster.example.com:6443"
```

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="openshift",
        k8s_api_host="https://api.cluster.example.com:6443"
    )
)
```

#### Option 3: Interactive OAuth Flow (For local development)

If no token is provided, the SDK will open a browser for interactive authentication:

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="openshift",
        k8s_api_host="https://api.cluster.example.com:6443",
        use_keyring=True  # Save token for future use
    )
)
```

**Getting OpenShift Tokens:**

1. **Using `oc` CLI:**
   ```bash
   # Login to OpenShift
   oc login https://api.cluster.example.com:6443

   # Get your token
   oc whoami -t
   ```

2. **From OpenShift Console:**
   - Click your username in the top right
   - Select "Copy login command"
   - Click "Display Token"
   - Copy the token value

**Requirements:**
- OpenShift cluster URL (`k8s_api_host`)
- Valid OpenShift token OR interactive browser access

## Configuration Options

All authentication configuration is done through the `KubernetesBackendConfig` class:

```python
class KubernetesBackendConfig:
    # Core configuration
    namespace: Optional[str] = None                    # Kubernetes namespace to use

    # Authentication method
    auth_method: Optional[str] = None                  # "auto", "kubeconfig", "incluster", "oidc", "openshift"
    k8s_api_host: Optional[str] = None                 # Kubernetes API server URL (required for OpenShift)
    kubeconfig_path: Optional[str] = None              # Path to kubeconfig file

    # OIDC configuration
    oidc_issuer: Optional[str] = None                  # OIDC provider URL
    client_id: Optional[str] = None                    # OAuth client ID
    client_secret: Optional[str] = None                # OAuth client secret
    scopes: Optional[list] = None                      # OIDC scopes (default: ["openid"])
    use_device_flow: bool = False                      # Enable device code flow for OIDC
    oidc_callback_port: int = 8080                     # OAuth callback port

    # Token-based authentication (OpenShift, etc.)
    token: Optional[str] = None                        # Authentication token (e.g., OpenShift OAuth token)

    # Advanced options
    use_keyring: bool = False                          # Persist tokens in system keyring
    verify_ssl: bool = True                            # Verify SSL certificates
    ca_cert: Optional[str] = None                      # Path to custom CA certificate

    # Legacy options (deprecated)
    config_file: Optional[str] = None                  # DEPRECATED: Use kubeconfig_path
    context: Optional[str] = None                      # DEPRECATED: Use kubectl to set context
    client_configuration: Optional[client.Configuration] = None  # Advanced: Pre-configured client
```

## Environment Variables

The SDK respects standard Kubernetes and kube-authkit environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `KUBECONFIG` | Path to kubeconfig file | `/path/to/kubeconfig` |
| `AUTHKIT_K8S_API_HOST` | Kubernetes API server URL | `https://api.cluster.example.com:6443` |
| `KUBERNETES_SERVICE_HOST` | Detected for in-cluster mode | `10.96.0.1` |
| `AUTHKIT_OIDC_ISSUER` | OIDC provider URL | `https://issuer.example.com` |
| `AUTHKIT_CLIENT_ID` | OAuth client ID | `my-client-id` |
| `AUTHKIT_CLIENT_SECRET` | OAuth client secret | `my-secret` |
| `AUTHKIT_TOKEN` | Authentication token (e.g., OpenShift) | `sha256~...` |

**Note:** kube-authkit 0.2.0 introduced the `AUTHKIT_` prefix for all authentication-related environment variables.

## Token Management

### Token Storage

By default, tokens are stored in memory and lost when the process ends. Enable keyring storage for persistence:

```python
backend_config = KubernetesBackendConfig(
    auth_method="oidc",
    oidc_issuer="https://issuer.example.com",
    client_id="client-id",
    client_secret="client-secret",
    use_keyring=True,  # Persist tokens in system keyring
)
```

**Keyring Storage:**
- Tokens are encrypted and stored in your system's keyring
- Automatically reused across sessions
- Supports macOS Keychain, Windows Credential Locker, Linux Secret Service

### Token Refresh

The SDK automatically refreshes expired tokens:
- OIDC tokens are refreshed using refresh tokens
- In-cluster service account tokens are automatically rotated by Kubernetes
- Kubeconfig credentials are managed by the Kubernetes client library

### Token Security

- Tokens are never logged or printed
- OIDC secrets are handled securely
- SSL/TLS verification is enabled by default

## Migration Guide

### Migrating from Legacy Authentication

If you're using the deprecated `config_file` or `context` parameters, migrate to the new `auth_method` parameter:

#### Before (Deprecated)

```python
# Old way - still works but logs deprecation warning
client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        config_file="/path/to/kubeconfig",
        context="my-context"
    )
)
```

#### After (Recommended)

```python
# New way - use environment variable or kubectl for context
import os

os.environ["KUBECONFIG"] = "/path/to/kubeconfig"

# Or use kubectl to set context:
# kubectl config use-context my-context

client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="kubeconfig"
    )
)
```

### Migration Checklist

- [ ] Identify current authentication method
- [ ] Update code to use `auth_method` parameter
- [ ] Test authentication in development environment
- [ ] Update documentation and examples
- [ ] Deploy to production
- [ ] Remove deprecated parameters

## Troubleshooting

### Common Issues

#### 1. "kube-authkit is required for authentication"

**Cause:** The `kube-authkit` package is not installed.

**Solution:**
```bash
pip install kube-authkit
```

#### 2. "Unable to authenticate with kubeconfig"

**Cause:** Kubeconfig file not found or invalid.

**Solution:**
- Verify kubeconfig exists: `ls -la ~/.kube/config`
- Check if kubeconfig is valid: `kubectl cluster-info`
- Set `KUBECONFIG` environment variable if using custom location

#### 3. "OIDC authentication failed"

**Cause:** Invalid OIDC configuration or expired credentials.

**Solution:**
- Verify `oidc_issuer` URL is correct and accessible
- Check `client_id` and `client_secret` are valid
- Clear cached tokens: `python -c "import keyring; keyring.delete_password('kube-authkit', 'oidc-token')"`

#### 4. "Permission denied" errors

**Cause:** Service account lacks necessary RBAC permissions.

**Solution:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: kubeflow-sdk-role
rules:
- apiGroups: ["trainer.kubeflow.org"]
  resources: ["trainjobs", "clustertrainingruntimes"]
  verbs: ["get", "list", "create", "delete", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kubeflow-sdk-rolebinding
subjects:
- kind: ServiceAccount
  name: kubeflow-sdk-sa
roleRef:
  kind: Role
  name: kubeflow-sdk-role
  apiGroup: rbac.authorization.k8s.io
```

#### 5. "SSL verification failed"

**Cause:** Self-signed certificates or certificate trust issues.

**Solution (NOT recommended for production):**
```python
# Only use this for development/testing environments
backend_config = KubernetesBackendConfig(
    auth_method="kubeconfig",
    verify_ssl=False  # Disable SSL verification
)
```

**Better Solution:** Add the CA certificate to your system's trust store.

### Debug Mode

Enable debug logging to troubleshoot authentication issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("kubeflow.common.auth")
logger.setLevel(logging.DEBUG)

# Now create your client - authentication logs will be verbose
client = TrainerClient(...)
```

### Getting Help

If you continue to experience authentication issues:

1. Check the [GitHub Issues](https://github.com/kubeflow/sdk/issues) for similar problems
2. Join the `#kubeflow-ml-experience` Slack channel
3. Review the [kube-authkit documentation](https://github.com/opendatahub-io/kube-authkit)

## Best Practices

1. **Use auto-detection when possible** - Let the SDK choose the best method
2. **Enable keyring storage for OIDC** - Avoid repeated authentication prompts
3. **Use device code flow for CLI tools** - Better UX for headless environments
4. **Never commit credentials** - Use environment variables or secure storage
5. **Enable SSL verification** - Only disable for development/testing
6. **Use service accounts in production** - More secure than user credentials
7. **Set appropriate RBAC permissions** - Follow principle of least privilege
8. **Monitor token expiration** - Implement error handling for expired tokens

## Examples

### Example 1: Development with Local Kubeconfig

```python
from kubeflow.trainer import TrainerClient

# Auto-detection uses your local kubeconfig
client = TrainerClient()
job_id = client.train(...)
```

### Example 2: Production with Service Account

```python
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

# Explicitly use in-cluster authentication
client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="incluster",
        namespace="ml-workloads"
    )
)
job_id = client.train(...)
```

### Example 3: Enterprise with OIDC

```python
import os
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

# Load credentials from environment
client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="oidc",
        oidc_issuer=os.environ["AUTHKIT_OIDC_ISSUER"],
        client_id=os.environ["AUTHKIT_CLIENT_ID"],
        client_secret=os.environ["AUTHKIT_CLIENT_SECRET"],
        use_keyring=True,
        namespace="ml-experiments"
    )
)
job_id = client.train(...)
```

### Example 4: OpenShift with Token

```python
import os
from kubeflow.trainer import TrainerClient
from kubeflow.common.types import KubernetesBackendConfig

# Using OpenShift token from environment
client = TrainerClient(
    backend_config=KubernetesBackendConfig(
        auth_method="openshift",
        k8s_api_host="https://api.cluster.example.com:6443",
        token=os.environ["AUTHKIT_TOKEN"],
        namespace="data-science"
    )
)
job_id = client.train(...)
```

## Additional Resources

- [kube-authkit GitHub Repository](https://github.com/opendatahub-io/kube-authkit)
- [kube-authkit PyPI Package](https://pypi.org/project/kube-authkit/)
- [Kubernetes Authentication Documentation](https://kubernetes.io/docs/reference/access-authn-authz/authentication/)
- [OIDC Specification](https://openid.net/specs/openid-connect-core-1_0.html)
- [OpenShift OAuth Documentation](https://docs.openshift.com/container-platform/latest/authentication/understanding-authentication.html)
