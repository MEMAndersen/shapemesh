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

"""Mesh settings for the Gmsh build pipeline.

Settings are a plain frozen dataclass (not a pydantic model): construction
rejects unknown keyword arguments and does no type coercion, and
`MeshSettings.with_` uses ``dataclasses.replace`` so changes are revalidated.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, replace
from enum import IntEnum

__all__ = ["Algorithm2D", "Algorithm3D", "MeshSettings"]


class Algorithm2D(IntEnum):
    """Gmsh 2D meshing algorithms (``Mesh.Algorithm``)."""

    MESH_ADAPT = 1
    AUTOMATIC = 2
    INITIAL_MESH_ONLY = 3
    DELAUNAY = 5
    FRONTAL_DELAUNAY = 6
    BAMG = 7
    FRONTAL_DELAUNAY_FOR_QUADS = 8
    PACKING_OF_PARALLELOGRAMS = 9
    QUASI_STRUCTURED_QUAD = 11


class Algorithm3D(IntEnum):
    """Gmsh 3D meshing algorithms (``Mesh.Algorithm3D``)."""

    DELAUNAY = 1
    INITIAL_MESH_ONLY = 3
    FRONTAL = 4
    MMG3D = 7
    R_TREE = 9
    HXT = 10


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


def _validate_advanced(
    advanced: tuple[tuple[str, float | int | str], ...],
) -> tuple[tuple[str, float | int | str], ...]:
    for pair in advanced:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"advanced entries must be (name, value) pairs, got {pair!r}")
        name, value = pair
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"advanced option name must be a non-empty string, got {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(
                f"advanced value for {name!r} must be an int, float, or str, "
                f"got {type(value).__name__}"
            )
    return tuple((name, value) for name, value in advanced)


@dataclass(frozen=True, slots=True)
class MeshSettings:
    """Validated settings applied to the Gmsh model before meshing.

    Args:
        element_size: Target element size (Gmsh's "characteristic length"),
            set at every model point before meshing; ``None`` uses Gmsh's
            default.
        order: Element order, ``1`` to ``5``.
        optimize: Whether to run mesh optimization after generation.
        verbose: Whether to show Gmsh's terminal output (silent by default).
        algorithm_2d: The 2D meshing algorithm, or ``None`` for Gmsh's default.
        algorithm_3d: The 3D meshing algorithm, or ``None`` for Gmsh's default.
        advanced: Extra ``(name, value)`` Gmsh options applied last.

    Raises:
        ValueError: If any field fails validation.
    """

    element_size: float | None = None  # Gmsh "characteristic length" at every point
    order: int = 1  # 1..5
    optimize: bool = True
    verbose: bool = False
    algorithm_2d: Algorithm2D | None = None
    algorithm_3d: Algorithm3D | None = None
    advanced: tuple[tuple[str, float | int | str], ...] = ()

    def __post_init__(self) -> None:
        if self.element_size is not None:
            object.__setattr__(
                self,
                "element_size",
                _positive_float(self.element_size, "element_size"),
            )
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise ValueError(f"order must be an int in 1..5, got {type(self.order).__name__}")
        if not 1 <= self.order <= 5:
            raise ValueError(f"order must be in 1..5, got {self.order}")
        if not isinstance(self.optimize, bool):
            raise ValueError(f"optimize must be a bool, got {type(self.optimize).__name__}")
        if not isinstance(self.verbose, bool):
            raise ValueError(f"verbose must be a bool, got {type(self.verbose).__name__}")
        if self.algorithm_2d is not None and not isinstance(self.algorithm_2d, Algorithm2D):
            raise ValueError(
                "algorithm_2d must be an Algorithm2D or None, "
                f"got {type(self.algorithm_2d).__name__}"
            )
        if self.algorithm_3d is not None and not isinstance(self.algorithm_3d, Algorithm3D):
            raise ValueError(
                "algorithm_3d must be an Algorithm3D or None, "
                f"got {type(self.algorithm_3d).__name__}"
            )
        object.__setattr__(self, "advanced", _validate_advanced(self.advanced))

    def with_(self, **changes: object) -> MeshSettings:
        """Return a copy with the given fields changed.

        Args:
            **changes: Fields to override.

        Returns:
            A new `MeshSettings` with ``changes`` applied (revalidated).

        Raises:
            TypeError: If an unknown field name is given.
            ValueError: If a changed field fails validation.
        """
        return replace(self, **changes)
