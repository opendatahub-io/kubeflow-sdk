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

"""HTTP metrics server code generator for training pods."""

from pathlib import Path


def get_http_server_code(port: int = 28080) -> str:
    """Generate HTTP server code from template with proper syntax highlighting.

    Reads the http_server_template.py file and renders it with the specified port.
    This approach provides better DX with IDE support, syntax highlighting, and linting.

    Args:
        port: Port number for the metrics server (default: 28080).

    Returns:
        Rendered Python code as string to inject into training containers.
    """
    template_path = Path(__file__).parent / "templates" / "http_server.py"
    template_code = template_path.read_text()

    # Replace template placeholder with actual port
    # Using simple string replacement since we only have one variable
    rendered_code = template_code.replace("__PORT_PLACEHOLDER__", str(port))

    return rendered_code
