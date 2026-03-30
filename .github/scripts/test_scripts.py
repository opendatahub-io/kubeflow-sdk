#!/usr/bin/env python3
"""
Unit tests for GitHub Actions scripts.

Run with: uv run pytest .github/scripts/test_scripts.py -v
"""

from pathlib import Path
import subprocess
import sys
import tempfile
from tempfile import NamedTemporaryFile

import pytest

SCRIPTS_DIR = Path(__file__).parent


class TestCompareVersions:
    """Tests for compare_versions.py"""

    def run_compare(self, current: str, target: str) -> int:
        """Helper to run compare_versions.py and return exit code"""
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "compare_versions.py", current, target],
            capture_output=True,
            text=True,
        )
        return result.returncode

    def test_upgrade_needed(self):
        """Current version is less than target - should return 0 (upgrade needed)"""
        assert self.run_compare("2.0.0", "2.1.0") == 0
        assert self.run_compare("1.0.0", "2.0.0") == 0
        assert self.run_compare("2.0.0", "2.0.1") == 0

    def test_no_upgrade_needed(self):
        """Current version equals or exceeds target - should return 1 (no upgrade)"""
        assert self.run_compare("2.1.0", "2.0.0") == 1
        assert self.run_compare("2.0.0", "2.0.0") == 1
        assert self.run_compare("3.0.0", "2.9.9") == 1

    def test_pre_release_versions(self):
        """Test PEP 440 pre-release versions"""
        # Pre-release is less than final release
        assert self.run_compare("2.0.0rc1", "2.0.0") == 0
        assert self.run_compare("2.0.0a1", "2.0.0") == 0
        assert self.run_compare("2.0.0b1", "2.0.0") == 0

        # Final release is greater than pre-release
        assert self.run_compare("2.0.0", "2.0.0rc1") == 1

    def test_post_release_versions(self):
        """Test PEP 440 post-release versions"""
        # Post-release is greater than base release
        assert self.run_compare("2.0.0", "2.0.0.post1") == 0
        assert self.run_compare("2.0.0.post1", "2.0.0") == 1

    def test_dev_versions(self):
        """Test PEP 440 dev versions"""
        # Dev versions are less than base release
        assert self.run_compare("2.0.0.dev1", "2.0.0") == 0
        assert self.run_compare("2.0.0", "2.0.0.dev1") == 1

    def test_complex_version_comparison(self):
        """Test complex multi-part version numbers"""
        assert self.run_compare("2.0.0.1", "2.0.0.2") == 0
        assert self.run_compare("2.10.0", "2.9.0") == 1  # 10 > 9, not string comparison
        assert self.run_compare("1.0.0rc1", "1.0.0rc2") == 0

    def test_invalid_version_error(self):
        """Invalid version should return error code 2"""
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "compare_versions.py", "invalid", "2.0.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "Error" in result.stderr

    def test_missing_arguments(self):
        """Missing arguments should return error code 2"""
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "compare_versions.py", "2.0.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2


class TestExtractVersion:
    """Tests for extract_version.py"""

    def run_extract(self, tree_output: str, package: str) -> tuple[int, str, str]:
        """Helper to run extract_version.py and return (exit_code, stdout, stderr)"""
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "extract_version.py", package],
            input=tree_output,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def test_simple_version_extraction(self):
        """Extract version from simple uv tree output"""
        tree_output = "requests v2.31.0"
        exit_code, version, _ = self.run_extract(tree_output, "requests")
        assert exit_code == 0
        assert version == "2.31.0"

    def test_version_with_metadata(self):
        """Extract version from tree output with metadata"""
        tree_output = "├── requests v2.31.0 (transitive)"
        exit_code, version, _ = self.run_extract(tree_output, "requests")
        assert exit_code == 0
        assert version == "2.31.0"

    def test_pre_release_version(self):
        """Extract pre-release version"""
        tree_output = "package v1.0.0rc1"
        exit_code, version, _ = self.run_extract(tree_output, "package")
        assert exit_code == 0
        assert version == "1.0.0rc1"

    def test_post_release_version(self):
        """Extract post-release version"""
        tree_output = "package v1.0.0.post1"
        exit_code, version, _ = self.run_extract(tree_output, "package")
        assert exit_code == 0
        assert version == "1.0.0.post1"

    def test_dev_version(self):
        """Extract dev version"""
        tree_output = "package v1.0.0.dev1"
        exit_code, version, _ = self.run_extract(tree_output, "package")
        assert exit_code == 0
        assert version == "1.0.0.dev1"

    def test_package_not_found(self):
        """Package not in tree output should return error"""
        tree_output = "other-package v1.0.0"
        exit_code, _, stderr = self.run_extract(tree_output, "requests")
        assert exit_code == 1
        assert "Could not find version" in stderr

    def test_multiline_tree_output(self):
        """Extract from realistic multi-line uv tree output"""
        tree_output = """
        project v1.0.0
        ├── requests v2.31.0
        │   ├── certifi v2023.7.22
        │   └── urllib3 v2.0.4
        └── pytest v7.4.0
        """
        exit_code, version, _ = self.run_extract(tree_output, "requests")
        assert exit_code == 0
        assert version == "2.31.0"


class TestUpdateOverrides:
    """Tests for update_overrides.py"""

    def run_update(
        self, pyproject_content: str, package: str, target: str, date: str, advisory: str
    ) -> tuple[int, str]:
        """Helper to run update_overrides.py with temp file"""
        with NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(pyproject_content)
            temp_path = Path(f.name)

        try:
            # Change to temp directory so script finds pyproject.toml
            import os

            original_dir = os.getcwd()
            os.chdir(temp_path.parent)

            # Create pyproject.toml in current directory
            Path("pyproject.toml").write_text(pyproject_content)

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPTS_DIR / "update_overrides.py",
                    package,
                    target,
                    date,
                    advisory,
                ],
                capture_output=True,
                text=True,
            )

            updated_content = Path("pyproject.toml").read_text()

            # Cleanup
            Path("pyproject.toml").unlink()
            os.chdir(original_dir)

            return result.returncode, updated_content
        finally:
            temp_path.unlink(missing_ok=True)

    def test_create_tool_uv_section(self):
        """Create [tool.uv] section if it doesn't exist"""
        pyproject = "[project]\nname = 'test'\n"
        exit_code, updated = self.run_update(
            pyproject, "requests", "requests==2.31.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        assert "[tool.uv]" in updated
        assert "override-dependencies = [" in updated
        assert '"requests==2.31.0",' in updated
        assert "# requests==2.31.0 - Added 2025-01-15" in updated

    def test_add_to_existing_tool_uv(self):
        """Add override-dependencies to existing [tool.uv]"""
        pyproject = "[project]\nname = 'test'\n\n[tool.uv]\n"
        exit_code, updated = self.run_update(
            pyproject, "requests", "requests==2.31.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        assert "override-dependencies = [" in updated
        assert '"requests==2.31.0",' in updated

    def test_update_existing_override(self):
        """Update existing package override to new version"""
        pyproject = """[project]
name = 'test'

[tool.uv]
# Security overrides - Review periodically and remove if parent constraints allow natural upgrade
override-dependencies = [
    # requests==2.30.0 - Added 2025-01-01 for security fix - https://old.com
    "requests==2.30.0",
]
"""
        exit_code, updated = self.run_update(
            pyproject, "requests", "requests==2.31.0", "2025-01-15", "https://new.com"
        )

        assert exit_code == 0
        assert '"requests==2.31.0",' in updated
        assert "2.30.0" not in updated  # Old version removed
        assert "2025-01-15" in updated  # New date
        assert "https://new.com" in updated  # New advisory

    def test_add_second_override(self):
        """Add second package to existing overrides"""
        pyproject = """[project]
name = 'test'

[tool.uv]
override-dependencies = [
    # requests==2.31.0 - Added 2025-01-15 for security fix - https://advisory.com
    "requests==2.31.0",
]
"""
        exit_code, updated = self.run_update(
            pyproject, "urllib3", "urllib3==2.0.5", "2025-01-16", "https://urllib-advisory.com"
        )

        assert exit_code == 0
        assert '"requests==2.31.0",' in updated
        assert '"urllib3==2.0.5",' in updated
        # Should be alphabetically sorted
        lines = updated.split("\n")
        requests_line = next(i for i, line in enumerate(lines) if "requests==" in line)
        urllib_line = next(i for i, line in enumerate(lines) if "urllib3==" in line)
        assert requests_line < urllib_line  # requests comes before urllib3

    def test_multiline_array_preserved(self):
        """Ensure output is always multi-line array format"""
        pyproject = "[project]\nname = 'test'\n"
        exit_code, updated = self.run_update(
            pyproject, "pkg", "pkg==1.0.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        # Check multi-line format
        assert "override-dependencies = [\n" in updated
        assert '    "pkg==1.0.0",\n' in updated
        assert "]\n" in updated

    def test_single_line_array_converted(self):
        """Single-line array should be parsed and converted to multi-line"""
        pyproject = """[project]
name = 'test'

[tool.uv]
override-dependencies = ["pkg1==1.0.0", "pkg2==2.0.0"]
"""
        exit_code, updated = self.run_update(
            pyproject, "pkg3", "pkg3==3.0.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        # Should preserve existing packages
        assert '"pkg1==1.0.0",' in updated
        assert '"pkg2==2.0.0",' in updated
        assert '"pkg3==3.0.0",' in updated
        # Should be in multi-line format
        assert "override-dependencies = [\n" in updated
        # Should not have duplicate override-dependencies
        assert updated.count("override-dependencies") == 1


class TestPinDependencies:
    """Tests for pin_dependencies.py"""

    # Minimal uv.lock content for testing.
    LOCK_TEMPLATE = """\
version = 1
revision = 2
requires-python = ">=3.10"

[[package]]
name = "kubernetes"
version = "35.0.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "pydantic"
version = "2.12.5"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "kubeflow-trainer-api"
version = "2.1.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "kubeflow-katib-api"
version = "0.19.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "docker"
version = "7.1.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "model-registry"
version = "0.3.6"
source = {{ registry = "https://pypi.org/simple" }}

{extra_packages}
"""

    PYPROJECT_BASIC = """\
[project]
name = "testpkg"
dependencies = [
  "kubernetes>=27.2.0",
  "pydantic>=2.10.0",
  "kubeflow-trainer-api>=2.0.0",
  "kubeflow-katib-api>=0.19.0",
]

[project.optional-dependencies]
docker = [
  "docker>=6.1.3",
]
hub = [
  "model-registry>=0.3.6",
]

[dependency-groups]
dev = [
  "pytest>=7.0",
  "ruff>=0.12.2",
]
"""

    def _run_pin(self, pyproject_content: str, lock_content: str) -> tuple[int, str, str, str]:
        """Helper to run pin_dependencies.py with temp files.

        Returns:
            Tuple of (exit_code, updated_pyproject, stdout, stderr).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject_path = Path(tmpdir) / "pyproject.toml"
            lock_path = Path(tmpdir) / "uv.lock"
            pyproject_path.write_text(pyproject_content)
            lock_path.write_text(lock_content)

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPTS_DIR / "pin_dependencies.py",
                    "--pyproject",
                    str(pyproject_path),
                    "--lockfile",
                    str(lock_path),
                ],
                capture_output=True,
                text=True,
            )
            updated = pyproject_path.read_text()
            return result.returncode, updated, result.stdout, result.stderr

    def test_core_dependencies_pinned(self):
        """Core dependencies get == pinned from lock file."""
        lock = self.LOCK_TEMPLATE.format(extra_packages="")
        exit_code, updated, stdout, _ = self._run_pin(self.PYPROJECT_BASIC, lock)

        assert exit_code == 0
        assert '"kubernetes==35.0.0"' in updated
        assert '"pydantic==2.12.5"' in updated
        assert '"kubeflow-trainer-api==2.1.0"' in updated
        assert '"kubeflow-katib-api==0.19.0"' in updated
        assert "Pinned" in stdout

    def test_optional_dependencies_pinned(self):
        """Optional dependencies get == pinned."""
        lock = self.LOCK_TEMPLATE.format(extra_packages="")
        exit_code, updated, _, _ = self._run_pin(self.PYPROJECT_BASIC, lock)

        assert exit_code == 0
        assert '"docker==7.1.0"' in updated
        # model-registry was already at 0.3.6 with >=, should pin to ==
        assert '"model-registry==0.3.6"' in updated

    def test_already_pinned_unchanged(self):
        """Dependencies already using == are left unchanged."""
        pyproject = """\
[project]
name = "testpkg"
dependencies = [
  "kubernetes==35.0.0",
]
"""
        lock = self.LOCK_TEMPLATE.format(extra_packages="")
        exit_code, updated, stdout, _ = self._run_pin(pyproject, lock)

        assert exit_code == 0
        assert '"kubernetes==35.0.0"' in updated
        assert "already pinned" in stdout.lower()

    def test_multi_version_resolution(self):
        """Multi-version packages resolve to the >=3.10 version."""
        extra = """\
[[package]]
name = "fsspec"
version = "2025.10.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = [
    "python_full_version < '3.10'",
]

[[package]]
name = "fsspec"
version = "2026.2.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = [
    "python_full_version >= '3.12' and python_full_version < '3.14'",
    "python_full_version >= '3.10' and python_full_version < '3.12'",
]
"""
        pyproject = """\
[project]
name = "testpkg"
dependencies = [
  "fsspec>=2025.3.0",
]
"""
        lock = self.LOCK_TEMPLATE.format(extra_packages=extra)
        exit_code, updated, _, _ = self._run_pin(pyproject, lock)

        assert exit_code == 0
        assert '"fsspec==2026.2.0"' in updated

    def test_multi_version_no_compatible_version_fails(self):
        """Multi-version package with no >=3.10 version fails with warning."""
        extra = """\
[[package]]
name = "oldpkg"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = [
    "python_full_version < '3.10'",
]

[[package]]
name = "oldpkg"
version = "1.1.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = [
    "python_full_version < '3.10'",
]
"""
        pyproject = """\
[project]
name = "testpkg"
dependencies = [
  "oldpkg>=1.0.0",
]
"""
        lock = self.LOCK_TEMPLATE.format(extra_packages=extra)
        exit_code, _, _, stderr = self._run_pin(pyproject, lock)

        assert exit_code == 1
        assert "WARNING" in stderr
        assert "not found" in stderr.lower()

    def test_environment_markers_preserved(self):
        """Environment markers after ; are preserved."""
        pyproject = """\
[project]
name = "testpkg"
dependencies = [
  "kubernetes>=27.2.0; python_version >= '3.11'",
]
"""
        lock = self.LOCK_TEMPLATE.format(extra_packages="")
        exit_code, updated, _, _ = self._run_pin(pyproject, lock)

        assert exit_code == 0
        assert "\"kubernetes==35.0.0; python_version >= '3.11'\"" in updated

    def test_missing_dependency_fails(self):
        """Missing dependency in lock file causes exit code 1."""
        pyproject = """\
[project]
name = "testpkg"
dependencies = [
  "nonexistent-pkg>=1.0.0",
]
"""
        lock = self.LOCK_TEMPLATE.format(extra_packages="")
        exit_code, _, _, stderr = self._run_pin(pyproject, lock)

        assert exit_code == 1
        assert "not found" in stderr.lower()

    def test_dependency_groups_untouched(self):
        """[dependency-groups] section is not modified."""
        lock = self.LOCK_TEMPLATE.format(extra_packages="")
        exit_code, updated, _, _ = self._run_pin(self.PYPROJECT_BASIC, lock)

        assert exit_code == 0
        # Dev deps should keep their original >= specifiers.
        assert '"pytest>=7.0"' in updated
        assert '"ruff>=0.12.2"' in updated

    def test_comments_preserved(self):
        """Comments in pyproject.toml are preserved."""
        pyproject = """\
[project]
name = "testpkg"
dependencies = [
  # Core Kubernetes client
  "kubernetes>=27.2.0",
]
"""
        lock = self.LOCK_TEMPLATE.format(extra_packages="")
        exit_code, updated, _, _ = self._run_pin(pyproject, lock)

        assert exit_code == 0
        assert "# Core Kubernetes client" in updated
        assert '"kubernetes==35.0.0"' in updated

    def test_extras_preserved(self):
        """Package extras like [security] are preserved."""
        extra = """\
[[package]]
name = "requests"
version = "2.31.0"
source = { registry = "https://pypi.org/simple" }
"""
        pyproject = """\
[project]
name = "testpkg"
dependencies = [
  "requests[security]>=2.28.0",
]
"""
        lock = self.LOCK_TEMPLATE.format(extra_packages=extra)
        exit_code, updated, _, _ = self._run_pin(pyproject, lock)

        assert exit_code == 0
        assert '"requests[security]==2.31.0"' in updated


if __name__ == "__main__":
    # Allow running with: python test_scripts.py
    pytest.main([__file__, "-v"])
