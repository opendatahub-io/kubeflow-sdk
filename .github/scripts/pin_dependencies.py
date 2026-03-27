#!/usr/bin/env python3
"""
Pin dependency versions in pyproject.toml from uv.lock.

Reads the resolved versions from uv.lock and replaces version specifiers
in pyproject.toml's [project.dependencies] and [project.optional-dependencies]
with exact == pins. Environment markers and extras are preserved.

The [dependency-groups] section is not modified (dev deps stay unpinned).

Requires Python >= 3.11 (for tomllib). This script is only invoked in CI
where Python 3.11+ is guaranteed by the workflow's setup-python step.

Usage:
    python pin_dependencies.py [--pyproject pyproject.toml] [--lockfile uv.lock]
"""

import argparse
from pathlib import Path
import re
import sys

import tomllib


def normalize_name(name: str) -> str:
    """Normalize a package name per PEP 503.

    Args:
        name: The package name to normalize.

    Returns:
        Lowercase name with runs of [-_.] replaced by a single hyphen.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock_versions(lock_path: Path) -> dict[str, str]:
    """Build a package-name-to-version map from uv.lock.

    For packages with multiple resolved versions (different Python markers),
    the version compatible with Python >= 3.10 is preferred.

    Args:
        lock_path: Path to the uv.lock file.

    Returns:
        Dict mapping normalized package names to their resolved version strings.
    """
    with open(lock_path, "rb") as f:
        lock_data = tomllib.load(f)

    # Collect all versions per package: {normalized_name: [(version, markers)]}
    packages: dict[str, list[tuple[str, list[str] | None]]] = {}
    for pkg in lock_data.get("package", []):
        # Skip packages without a version (e.g., editable/workspace packages).
        if "version" not in pkg:
            continue
        name = normalize_name(pkg["name"])
        version = pkg["version"]
        markers = pkg.get("resolution-markers")
        packages.setdefault(name, []).append((version, markers))

    # Resolve to a single version per package
    resolved: dict[str, str] = {}
    for name, versions in packages.items():
        if len(versions) == 1:
            resolved[name] = versions[0][0]
            continue

        # Multiple versions: pick the one compatible with Python >= 3.10.
        # Exclude versions whose markers are exclusively for Python < 3.10.
        for version, markers in versions:
            if markers is None:
                # No markers means universal — use it.
                resolved[name] = version
                break
            # Check if any marker includes Python >= 3.10.
            has_compat_marker = False
            for marker in markers:
                # Markers that restrict to old Python look like:
                #   "python_full_version < '3.10'"
                # We want markers that don't exclude >= 3.10.
                if not re.search(r"python_full_version\s*<\s*'3\.10'", marker):
                    has_compat_marker = True
                    break
            if has_compat_marker:
                resolved[name] = version
                break

        # No compatible version found — warn and skip.
        if name not in resolved:
            print(
                f"WARNING: No Python >=3.10 compatible version found for '{name}' "
                f"in lock file. Available versions: "
                f"{', '.join(v for v, _ in versions)}",
                file=sys.stderr,
            )

    return resolved


def pin_dependencies(pyproject_path: Path, lock_path: Path) -> list[str]:
    """Pin dependency versions in pyproject.toml using resolved versions from uv.lock.

    Only modifies [project.dependencies] and [project.optional-dependencies].
    The [dependency-groups] section is left untouched.

    Args:
        pyproject_path: Path to pyproject.toml.
        lock_path: Path to uv.lock.

    Returns:
        List of pinned dependency descriptions (for logging).

    Raises:
        SystemExit: If a dependency cannot be found in the lock file.
    """
    resolved = parse_lock_versions(lock_path)
    content = pyproject_path.read_text()

    # Regex to match a quoted dependency string with a version specifier.
    # Captures: name (with optional extras), version operator+specifier, rest (marker etc.)
    # Example: "kubernetes>=27.2.0" or "fsspec>=2025.3.0" or "pkg[extra]>=1.0; marker"
    # NOTE: URL-based deps (e.g., "pkg @ https://...") and bare names without version
    # specifiers are intentionally not matched and will be left unchanged.
    dep_pattern = re.compile(
        r'"'
        r"(?P<name>[a-zA-Z0-9][-a-zA-Z0-9_.]*(?:\[[^\]]*\])?)"  # name with optional extras
        r"(?P<verspec>[><=!~][^;\"]*?)"  # version specifier(s)
        r"(?P<rest>;[^\"]*)?"  # optional environment markers
        r'"'
    )

    pinned_deps: list[str] = []
    errors: list[str] = []

    # Track which TOML section we're in.
    # "project" = [project] section (contains dependencies = [...])
    # "optional" = [project.optional-dependencies] section
    # "depgroups" = [dependency-groups] section (skip)
    # "other" = any other section
    current_section = "other"
    in_dep_array = False  # True when inside a dependencies = [ ... ] array

    lines = content.split("\n")
    result_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Detect TOML section headers like [project], [project.optional-dependencies], etc.
        section_match = re.match(r"^\[\s*([a-zA-Z0-9._-]+)\s*\]", stripped)
        if section_match:
            section_name = section_match.group(1)
            if section_name == "project":
                current_section = "project"
            elif section_name == "project.optional-dependencies":
                current_section = "optional"
            elif section_name == "dependency-groups":
                current_section = "depgroups"
            else:
                current_section = "other"
            in_dep_array = False

        # Detect start of a dependency array (e.g., "dependencies = [" or "docker = [")
        if current_section in ("project", "optional") and re.match(
            r"^[a-zA-Z_-]+\s*=\s*\[", stripped
        ):
            in_dep_array = True

        # Detect end of array.
        if in_dep_array and stripped == "]":
            in_dep_array = False

        # Skip dependency-groups entirely.
        if current_section == "depgroups":
            result_lines.append(line)
            continue

        # Only process lines inside dependency arrays in project/optional sections.
        if not in_dep_array:
            result_lines.append(line)
            continue

        # Check if this line has a dependency string to pin.
        match = dep_pattern.search(line)
        if match:
            raw_name = match.group("name")
            verspec = match.group("verspec")
            rest = match.group("rest") or ""

            # Extract the bare name (without extras) for lookup.
            bare_name = re.sub(r"\[.*\]", "", raw_name)
            normalized = normalize_name(bare_name)

            if normalized not in resolved:
                errors.append(f"Package '{bare_name}' not found in {lock_path}")
                result_lines.append(line)
                continue

            pinned_version = resolved[normalized]

            # Skip if already pinned to the exact version.
            if verspec.strip() == f"=={pinned_version}":
                result_lines.append(line)
                continue

            # Build the new dependency string.
            new_dep = f'"{raw_name}=={pinned_version}{rest}"'
            new_line = line[: match.start()] + new_dep + line[match.end() :]
            result_lines.append(new_line)
            pinned_deps.append(f"{bare_name}: {verspec.strip()} -> =={pinned_version}")
        else:
            result_lines.append(line)

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    pyproject_path.write_text("\n".join(result_lines))
    return pinned_deps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pin dependency versions in pyproject.toml from uv.lock."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml (default: ./pyproject.toml)",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=Path("uv.lock"),
        help="Path to uv.lock (default: ./uv.lock)",
    )
    args = parser.parse_args()

    if not args.pyproject.exists():
        print(f"Error: {args.pyproject} not found", file=sys.stderr)
        sys.exit(1)
    if not args.lockfile.exists():
        print(f"Error: {args.lockfile} not found", file=sys.stderr)
        sys.exit(1)

    pinned = pin_dependencies(args.pyproject, args.lockfile)

    if pinned:
        print(f"Pinned {len(pinned)} dependencies:")
        for dep in pinned:
            print(f"  {dep}")
    else:
        print("All dependencies already pinned.")


if __name__ == "__main__":
    main()
