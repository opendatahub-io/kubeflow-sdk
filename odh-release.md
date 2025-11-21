# ODH Midstream Release Process

This document describes the release process for the OpenDataHub (ODH) midstream fork of Kubeflow SDK.

## Prerequisites

- Write permission for the opendatahub-io/kubeflow-sdk repository
- Access to trigger GitHub Actions workflows

## Versioning Policy

### Base Version
The base version in `kubeflow/__init__.py` follows upstream Kubeflow SDK versioning (format: `X.Y.Z`).

### ODH Release Version
ODH releases use the format: `vX.Y.Z-odh-N` where:
- `X.Y.Z` is the upstream base version
- `-odh` suffix indicates an ODH midstream release
- `-N` is an auto-incrementing number starting at 1

**Examples:**
- First ODH release based on upstream 0.2.0: `v0.2.0-odh-1`
- Second ODH release (with patches) on same base: `v0.2.0-odh-2`
- First ODH release based on upstream 0.3.0: `v0.3.0-odh-1`

The release workflow automatically determines the next available `-N` increment.

## Changelog Management

### Changelog Structure

ODH maintains the same changelog structure as upstream:

```markdown
CHANGELOG/
├── CHANGELOG-0.1.md    # All 0.1.x releases
├── CHANGELOG-0.2.md    # All 0.2.x releases
└── CHANGELOG-0.3.md    # All 0.3.x releases
```

### Adding ODH-Specific Changes

When making ODH-specific changes, manually add them to the appropriate changelog file and prefix them with `[ODH]` to distinguish from upstream changes.

**Example:**

```markdown
# [0.2.0](https://github.com/kubeflow/sdk/releases/tag/0.2.0) (2025-11-06)

## New Features

- feat(optimizer): Add get_best_results API to OptimizerClient ([#152](https://github.com/kubeflow/sdk/pull/152)) by @kramaranya
- **[ODH] feat: Add support for ODH authentication** by @yourname

## Bug Fixes

- fix(ci): Update url for installing docker ([#151](https://github.com/kubeflow/sdk/pull/151)) by @Fiona-Waters
- **[ODH] fix: Container backend compatibility with ODH clusters** by @yourname

## Maintenance

- chore: Add HPO support to readme ([#141](https://github.com/kubeflow/sdk/pull/141)) by @kramaranya
- **[ODH] chore: Update documentation for ODH deployment** by @yourname
```

### Guidelines for ODH Changelog Entries

1. **Prefix with `[ODH]`**: All ODH-specific changes should be clearly marked
2. **Follow upstream conventions**: Use the same format as upstream (feat/fix/chore prefix)
3. **Add to appropriate section**: New Features, Bug Fixes, or Maintenance
4. **Keep it concise**: One line per change, focus on what and why

## Release Process

### 1. Update Changelog (if needed)

If you're releasing ODH-specific changes:

1. Edit the appropriate `CHANGELOG/CHANGELOG-X.Y.md` file
2. Add your ODH-specific changes with `[ODH]` prefix
3. Commit and push to `main`:
   ```bash
   git add CHANGELOG/CHANGELOG-X.Y.md
   git commit -m "chore: Update changelog for ODH release"
   git push origin main
   ```

### 2. Trigger Release Workflow

The release workflow is **manual-only** to give you full control:

1. Go to [GitHub Actions](https://github.com/opendatahub-io/kubeflow-sdk/actions)
2. Select the **ODH Release** workflow
3. Click **Run workflow**
4. Select the branch to release from (usually `main`)
5. Click **Run workflow**

### 3. Workflow Steps

The automated workflow will:

1. **Prepare**:
   - Extract base version from `kubeflow/__init__.py` (e.g., `0.2.0`)
   - Determine next ODH release version (e.g., `v0.2.0-odh-1`)

2. **Build**:
   - Set up Python environment
   - Run tests (`make test-python`)
   - Build package with `uv build`
   - Upload build artifacts

3. **Create Tag**:
   - Create git tag with release version
   - Push tag to repository

4. **GitHub Release**:
   - Extract changelog from `CHANGELOG/CHANGELOG-X.Y.md`
   - Create GitHub release with changelog as release notes
   - Attach build artifacts to release

### 4. Verification

After the workflow completes:

1. Verify the tag was created:
   ```bash
   git fetch --tags
   git tag -l "v*-odh-*"
   ```

2. Check the [GitHub Releases page](https://github.com/opendatahub-io/kubeflow-sdk/releases)

3. Verify release notes include both upstream and ODH changes
