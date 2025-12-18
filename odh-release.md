# ODH Midstream Release Process

This document describes the release process for the OpenDataHub (ODH) midstream fork of Kubeflow SDK.

## Prerequisites

- Write permission for the opendatahub-io/kubeflow-sdk repository
- Access to trigger GitHub Actions workflows

## Versioning Policy

### Main Branch (Synced with Upstream)
The `main` branch version in `kubeflow/__init__.py` stays in sync with upstream Kubeflow SDK:
```python
__version__ = "0.2.0"  # Matches upstream
```

### Release Branches (RHAI Versioning)
Release branches use the RHAI version format: `vX.Y.Z+rhaiN` where:
- `X.Y.Z` is the upstream base version
- `+rhai` suffix indicates an RHAI midstream release
- `N` is an incrementing number starting at 0 for each upstream version

**Examples:**
- First release based on upstream 0.2.0: `v0.2.0+rhai0`
- Second release (with patches) on same base: `v0.2.0+rhai1`
- First release based on upstream 0.3.0: `v0.3.0+rhai0`

## Release Branches

Releases are cut from minor version branches:
- `release-0.2` for all `v0.2.x+rhaiN` releases
- `release-0.3` for all `v0.3.x+rhaiN` releases

The workflow automatically:
1. Creates these branches if they don't exist
2. Updates `__init__.py` on the release branch with the rhai version

## Release Process

### 1. Trigger Release Workflow

1. Go to [GitHub Actions](https://github.com/opendatahub-io/kubeflow-sdk/actions)
2. Select the **ODH Release** workflow
3. Click **Run workflow**
4. Enter the release version (e.g., `v0.2.0+rhai0`)
5. Click **Run workflow**

### 2. Workflow Steps

The automated workflow will:

1. **Validate & Prepare**:
   - Validate version format matches `vX.Y.Z+rhaiN`
   - Create release branch from main if it doesn't exist (e.g., `release-0.2`)
   - Or use existing release branch as-is
   - Update `__init__.py` on release branch with rhai version

2. **Build**:
   - Set up Python environment
   - Run tests (`make test-python`)
   - Verify version in code matches expected
   - Build package with `uv build`

3. **Create Tag**:
   - Create git tag with the version (e.g., `v0.2.0+rhai0`)
   - Push tag to repository

4. **GitHub Release**:
   - Create GitHub release with auto-generated release notes
   - Attach build artifacts to release

### 3. Verification

After the workflow completes:

1. Verify the tag was created:
   ```bash
   git fetch --tags
   git tag -l "v*+rhai*"
   ```

2. Check the [GitHub Releases page](https://github.com/opendatahub-io/kubeflow-sdk/releases)

3. Verify the release branch has the rhai version:
   ```bash
   git fetch origin
   git show origin/release-0.2:kubeflow/__init__.py
   ```
