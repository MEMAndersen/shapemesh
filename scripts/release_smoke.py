# Copyright (C) 2026 Mads Kruse
#
# This file is part of shapemesh.
#
# shapemesh is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# shapemesh is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with shapemesh. If not, see <https://www.gnu.org/licenses/>.

"""Smoke-test an installed shapemesh wheel before a release.

Run this from a clean environment with the candidate wheel installed. It
verifies the installed package metadata and files, builds a small mesh, and
round-trips it through MSH.

Usage:
    python scripts/release_smoke.py --expect-version 0.2.0
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, files, version
from pathlib import Path
from typing import Never

import shapemesh
from shapemesh import Box, Mesh, MeshSettings, build


def _fail(message: str) -> Never:
    """Exit the release smoke test with a failure message."""
    raise SystemExit(f"release smoke test failed: {message}")


def _check(condition: bool, message: str) -> None:
    """Fail the release smoke test when a condition is false."""
    if not condition:
        _fail(message)


def _check_version(expected: str | None) -> str:
    """Verify the installed and expected versions match the module."""
    try:
        installed_version = version("shapemesh")
    except PackageNotFoundError:
        _fail("shapemesh is not installed in this interpreter")

    package_version = shapemesh.__version__
    _check(
        installed_version == package_version,
        f"distribution metadata reports {installed_version}, "
        f"but the module reports {package_version}",
    )

    if expected is not None:
        _check(
            package_version == expected,
            f"expected version {expected}, but the installed version is {package_version}",
        )

    return package_version


def _check_install_location() -> Path:
    """Verify that the package was imported from the active environment."""
    module_file = shapemesh.__file__
    _check(module_file is not None, "the installed package has no __file__")

    package_dir = Path(module_file).resolve().parent
    _check(
        package_dir.is_relative_to(Path(sys.prefix).resolve()),
        f"imported {package_dir} instead of a package inside {sys.prefix}",
    )
    return package_dir


def _check_metadata_files() -> None:
    """Verify that the wheel installed its typing marker and licenses."""
    installed_files = files("shapemesh")
    _check(installed_files is not None, "the installed distribution has no file list")

    paths = {str(path).replace("\\", "/") for path in installed_files or []}
    _check("shapemesh/py.typed" in paths, "the installed wheel is missing shapemesh/py.typed")

    for name in ("COPYING", "COPYING.LESSER"):
        _check(
            any(path.endswith(f"licenses/{name}") for path in paths),
            f"the installed wheel is missing {name}",
        )


def _build_mesh() -> Mesh:
    """Build a small mesh and verify its basic structure."""
    mesh = build(
        shapes=Box(origin=(0, 0, 0), dx=1, dy=1, dz=1, label="box"),
        settings=MeshSettings(element_size=0.5),
    )

    _check(mesh.num_nodes > 0, "build() produced no nodes")
    _check(mesh.num_elements > 0, "build() produced no elements")
    _check(
        bool(mesh.physical_group(name="box", dim=3).entities),
        "the box physical group contains no entities",
    )
    return mesh


def _check_roundtrip(mesh: Mesh) -> None:
    """Write and reread the mesh, then exercise the group query APIs."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.msh"
        mesh.write_msh(path)
        _check(path.is_file() and path.stat().st_size > 0, "write_msh() produced no file")
        reloaded = Mesh.read_msh(path)

    _check(
        reloaded.num_nodes == mesh.num_nodes,
        f"reread {reloaded.num_nodes} nodes instead of {mesh.num_nodes}",
    )
    _check(
        reloaded.num_elements == mesh.num_elements,
        f"reread {reloaded.num_elements} elements instead of {mesh.num_elements}",
    )
    _check(
        reloaded.physical_group(name="box", dim=3).entities
        == mesh.physical_group(name="box", dim=3).entities,
        "the box physical group changed after the MSH round-trip",
    )
    _check(bool(reloaded.blocks_by_type()), "blocks_by_type() returned no blocks")
    _check(
        bool(reloaded.element_indices_in_group(name="box", dim=3)),
        "element_indices_in_group() returned no elements",
    )
    _check(
        reloaded.node_indices_in_group(name="box", dim=3).size > 0,
        "node_indices_in_group() returned no nodes",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the installed-wheel release smoke test.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every check passes.

    Raises:
        SystemExit: If a check fails or the arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-version",
        metavar="X.Y.Z",
        help="require the installed version to match this value",
    )
    args = parser.parse_args(argv)

    package_version = _check_version(args.expect_version)
    package_dir = _check_install_location()
    _check_metadata_files()
    mesh = _build_mesh()
    _check_roundtrip(mesh)

    print(
        f"OK: shapemesh {package_version} from {package_dir}; "
        f"{mesh.num_nodes} nodes, {mesh.num_elements} elements"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
