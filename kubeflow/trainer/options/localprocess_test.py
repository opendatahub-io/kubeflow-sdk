# Copyright The Kubeflow Authors.
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

"""Unit tests for the localprocess options module."""

import pytest

from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="localprocess options module imports without error",
            expected_status=SUCCESS,
            config={},
        ),
    ],
)
def test_localprocess_options_module_imports(test_case):
    """Verify the localprocess options module can be imported."""
    print("Executing test:", test_case.name)

    import kubeflow.trainer.options.localprocess  # noqa: F401

    print("test execution complete")
