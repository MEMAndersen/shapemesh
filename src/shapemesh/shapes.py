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

"""Shape primitives and n-ary boolean operations.

Shapes are immutable, hashable frozen dataclasses. Geometry is validated on
construction (raising `GeometryError`), coordinates are normalized to `Vec3`,
and ``label`` is validated by the base class (raising `LabelError`). A ``label``
with surrounding whitespace is stripped, which emits a UserWarning so callers
know the stored label differs from the one they passed.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import numpy as np

from .constants import (
    COLLINEARITY_TOLERANCE,
    CONNECTIVITY_TOLERANCE,
    PLANARITY_TOLERANCE,
    PLANE_ORTHOGONALITY_TOLERANCE,
    TANGENCY_TOLERANCE,
)
from .errors import GeometryError, LabelError

__all__ = [
    "Box",
    "Circle",
    "CircleArc",
    "Cylinder",
    "Difference",
    "Disk",
    "Ellipse",
    "EllipticalDisk",
    "Extrude",
    "ExtrudeAlong",
    "Intersection",
    "Line",
    "Path",
    "Point",
    "Polygon",
    "Polyline",
    "Rectangle",
    "Revolve",
    "Rotate",
    "Shape",
    "Sphere",
    "Translate",
    "Union",
    "Vec3",
    "Vec3Like",
]


def _validate_label(label: str | None) -> str | None:
    """Return the normalized (stripped) label, or raise `LabelError`."""
    if label is None:
        return None
    if not isinstance(label, str):
        raise LabelError(f"label must be a string or None, got {type(label).__name__}")
    stripped = label.strip()
    if not stripped:
        raise LabelError("label must be non-empty after stripping whitespace")
    if len(stripped) > 127:
        raise LabelError("label must be at most 127 characters")
    if any(ch in stripped for ch in ('"', "'", "\n", "\r")):
        raise LabelError("label must not contain quotes or newlines")
    return stripped


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise GeometryError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError(f"{name} must be representable as a float, got {value!r}") from exc
    if not math.isfinite(converted):
        raise GeometryError(f"{name} must be finite, got {value}")
    return converted


def _positive_float(value: object, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0:
        raise GeometryError(f"{name} must be positive, got {value}")
    return converted


def _bbox_diagonal(points: Sequence[Vec3]) -> float:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    return math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2)


@dataclass(frozen=True, slots=True, kw_only=True)
class Shape:
    """Base class for all shapes.

    Concrete classes implement ``dimension`` (an int constant on primitives,
    computed on booleans) and validate their geometry in ``__post_init__``,
    calling ``Shape.__post_init__`` to validate the label.

    Args:
        label: An optional physical-group name. Surrounding whitespace is
            stripped (with a warning).
    """

    label: str | None = None

    @property
    def dimension(self) -> int:
        """Topological dimension of the shape (0, 1, 2, or 3)."""
        raise NotImplementedError

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        """Realize this shape as OCC entities, returning its top-dimension dimtags."""
        raise NotImplementedError

    def __post_init__(self) -> None:
        label = _validate_label(self.label)
        if label != self.label:
            warnings.warn(
                f"label {self.label!r} was stripped of surrounding whitespace to {label!r}",
                UserWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "label", label)

    def __or__(self, other: Shape) -> Union:
        return Union(self, other)

    def __and__(self, other: Shape) -> Intersection:
        return Intersection(self, other)

    def __sub__(self, other: Shape) -> Difference:
        return Difference(self, other)


@dataclass(frozen=True, slots=True)
class Vec3:
    """A 3D coordinate with float components.

    Args:
        x: The x coordinate.
        y: The y coordinate.
        z: The z coordinate (defaults to ``0.0``; two-component input pads
            ``z=0``).

    Raises:
        GeometryError: If a component is not a real number or cannot be
            represented as a finite float.
    """

    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise GeometryError(
                    f"Vec3.{name} must be a real number, got {type(value).__name__}"
                )
            try:
                converted = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise GeometryError(f"Vec3.{name} must be representable as a float") from exc
            if not math.isfinite(converted):
                raise GeometryError("Vec3 coordinates must be finite")
            object.__setattr__(self, name, converted)

    @classmethod
    def from_any(cls, value: Vec3Like) -> Vec3:
        """Normalize a `Vec3Like` into a `Vec3`.

        Accepts a ``Vec3``, a sequence of 2-3 real numbers, or an ndarray.
        Two-component input pads ``z=0``. A `Shape` (e.g. ``Point``) is
        rejected; pass ``.coord`` instead.

        Args:
            value: The value to normalize.

        Returns:
            The resulting `Vec3`.

        Raises:
            GeometryError: If ``value`` is not a supported coordinate or has
                an invalid component count or value.
        """
        if isinstance(value, Vec3):
            return value
        if isinstance(value, Shape):
            raise GeometryError("a Shape cannot be used as a vertex coordinate; pass `.coord`")
        if isinstance(value, (str, bytes)):
            raise GeometryError(
                f"expected a Vec3 or a sequence of 2-3 floats, got {type(value).__name__}"
            )
        if not isinstance(value, (Sequence, np.ndarray)):
            raise GeometryError(
                f"expected a Vec3 or a sequence of 2-3 floats, got {type(value).__name__}"
            )
        try:
            components = list(value)
        except TypeError as exc:
            raise GeometryError(
                f"expected a Vec3 or a sequence of 2-3 floats, got {type(value).__name__}"
            ) from exc
        if len(components) not in (2, 3):
            raise GeometryError(f"expected 2 or 3 components, got {len(components)}")
        return cls(components[0], components[1], components[2] if len(components) == 3 else 0.0)

    # -- vector arithmetic -------------------------------------------------

    def __add__(self, other: Vec3) -> Vec3:
        """Return the component-wise sum of two vectors."""
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        """Return the component-wise difference of two vectors."""
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> Vec3:
        """Return the negated vector."""
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: Vec3) -> float:
        """Return the dot product with ``other``."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        """Return the cross product with ``other``."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        """Return the Euclidean length (magnitude)."""
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vec3:
        """Return the unit vector in this direction.

        Raises:
            GeometryError: If the vector has zero length.
        """
        n = self.length()
        if n == 0.0:
            raise GeometryError("cannot normalize a zero-length vector")
        return Vec3(self.x / n, self.y / n, self.z / n)


type Vec3Like = Vec3 | Sequence[float] | np.ndarray


_GLOBAL_AXES = (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))


def _normalize_direction(value: Vec3Like, name: str) -> Vec3:
    """Return a unit direction using scale-safe normalization."""
    vector = Vec3.from_any(value)
    scale = max(abs(vector.x), abs(vector.y), abs(vector.z))
    if scale == 0.0:
        raise GeometryError(f"{name} must be nonzero")
    scaled = Vec3(vector.x / scale, vector.y / scale, vector.z / scale)
    length = math.sqrt(scaled.x**2 + scaled.y**2 + scaled.z**2)
    return Vec3(scaled.x / length, scaled.y / length, scaled.z / length)


def _project_to_plane(direction: Vec3, normal: Vec3) -> Vec3:
    """Project a direction onto the plane perpendicular to a unit normal."""
    alignment = direction.dot(normal)
    return Vec3(
        direction.x - alignment * normal.x,
        direction.y - alignment * normal.y,
        direction.z - alignment * normal.z,
    )


def _rotate_about_axis(vector: Vec3, angle: float, axis: Vec3) -> Vec3:
    """Rotate ``vector`` around a unit ``axis`` using Rodrigues' formula."""
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    cross = axis.cross(vector)
    projection = axis.dot(vector) * (1.0 - cos_angle)
    return Vec3(
        vector.x * cos_angle + cross.x * sin_angle + axis.x * projection,
        vector.y * cos_angle + cross.y * sin_angle + axis.y * projection,
        vector.z * cos_angle + cross.z * sin_angle + axis.z * projection,
    )


def _plane_axes(normal: Vec3Like, x_axis: Vec3Like | None) -> tuple[Vec3, Vec3, Vec3]:
    """Return normalized normal, x, and y directions for a planar shape."""
    unit_normal = _normalize_direction(normal, "normal")
    if x_axis is None:
        reference = min(_GLOBAL_AXES, key=lambda axis: abs(unit_normal.dot(axis)))
        unit_x = _normalize_direction(
            _project_to_plane(reference, unit_normal),
            "automatic x_axis",
        )
    else:
        unit_x = _normalize_direction(x_axis, "x_axis")
        if abs(unit_normal.dot(unit_x)) > PLANE_ORTHOGONALITY_TOLERANCE:
            raise GeometryError("x_axis must be perpendicular to normal")
        unit_x = _normalize_direction(
            _project_to_plane(unit_x, unit_normal),
            "x_axis",
        )
    unit_y = _normalize_direction(unit_normal.cross(unit_x), "plane y-axis")
    return unit_normal, unit_x, unit_y


def _xyz(vector: Vec3) -> tuple[float, float, float]:
    """Return vector components in the form expected by Gmsh."""
    return vector.x, vector.y, vector.z


def _normalize_point_sequence(points: Sequence[Vec3Like], shape_name: str) -> tuple[Vec3, ...]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise GeometryError(f"{shape_name} points must be a sequence of coordinates")
    return tuple(Vec3.from_any(point) for point in points)


# -- primitives ------------------------------------------------------------


@dataclass(frozen=True, slots=True, init=False)
class Point(Shape):
    """A single point (0-dimensional).

    Args:
        coord: The point location.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``coord`` is not a valid coordinate.
        LabelError: If ``label`` is invalid.
    """

    coord: Vec3

    def __init__(self, coord: Vec3Like, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "coord", Vec3.from_any(coord))
        self.__post_init__()

    def __post_init__(self) -> None:
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 0

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        tag = gmsh.model.occ.addPoint(self.coord.x, self.coord.y, self.coord.z, 0.0)
        return [(0, tag)]


@dataclass(frozen=True, slots=True, init=False)
class Line(Shape):
    """A straight line segment between two points (1-dimensional).

    Args:
        start: The start point.
        end: The end point.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``start`` and ``end`` coincide, or either is invalid.
        LabelError: If ``label`` is invalid.
    """

    start: Vec3
    end: Vec3

    def __init__(self, start: Vec3Like, end: Vec3Like, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "start", Vec3.from_any(start))
        object.__setattr__(self, "end", Vec3.from_any(end))
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise GeometryError("Line start and end points must be distinct")
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 1

    @property
    def start_tangent(self) -> Vec3:
        """The unit tangent at the line start, pointing toward the end."""
        return (self.end - self.start).normalized()

    @property
    def end_tangent(self) -> Vec3:
        """The unit tangent at the line end, pointing toward the end."""
        return self.start_tangent

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        occ = gmsh.model.occ
        p1 = occ.addPoint(self.start.x, self.start.y, self.start.z, 0.0)
        p2 = occ.addPoint(self.end.x, self.end.y, self.end.z, 0.0)
        return [(1, occ.addLine(p1, p2))]


@dataclass(frozen=True, slots=True, init=False)
class Polyline(Shape):
    """A connected chain of line segments (1-dimensional).

    The polyline is closed when the first and last points coincide (see
    ``closed``).

    Args:
        points: Two or more vertices.
        label: An optional physical-group name.

    Raises:
        GeometryError: If there are fewer than 2 points, consecutive points
            coincide, or a closed polyline has fewer than 3 distinct points.
        LabelError: If ``label`` is invalid.
    """

    points: tuple[Vec3, ...]

    def __init__(self, points: Sequence[Vec3Like], *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "points", _normalize_point_sequence(points, type(self).__name__))
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_polyline(self.points)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 1

    @property
    def closed(self) -> bool:
        """Whether the polyline is closed (first point equals last point)."""
        return self.points[0] == self.points[-1]

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        occ = gmsh.model.occ
        unique = self.points[:-1] if self.closed else self.points
        points = [occ.addPoint(p.x, p.y, p.z, 0.0) for p in unique]
        if self.closed:
            n = len(points)
            lines = [occ.addLine(points[i], points[(i + 1) % n]) for i in range(n)]
        else:
            lines = [occ.addLine(points[i], points[i + 1]) for i in range(len(points) - 1)]
        return [(1, t) for t in lines]


@dataclass(frozen=True, slots=True, init=False)
class Circle(Shape):
    """A circular curve in an arbitrary plane (1-dimensional).

    This is the boundary curve only; see `Disk` for the filled region.

    Args:
        center: The circle center.
        radius: The radius (positive).
        normal: The plane normal (nonzero). The default is the global z-axis.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``radius`` is not positive or ``normal`` is invalid.
        LabelError: If ``label`` is invalid.
    """

    center: Vec3
    radius: float
    normal: Vec3

    def __init__(
        self,
        center: Vec3Like,
        radius: float,
        *,
        normal: Vec3Like = (0, 0, 1),
        label: str | None = None,
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "center", Vec3.from_any(center))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "normal", normal)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive_float(self.radius, "radius"))
        normal, _, _ = _plane_axes(self.normal, None)
        object.__setattr__(self, "normal", normal)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 1

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        normal, x_axis, _ = _plane_axes(self.normal, None)
        curve = gmsh.model.occ.addCircle(
            self.center.x,
            self.center.y,
            self.center.z,
            self.radius,
            zAxis=_xyz(normal),
            xAxis=_xyz(x_axis),
        )
        return [(1, curve)]


@dataclass(frozen=True, slots=True, init=False)
class Ellipse(Shape):
    """An elliptical curve in an arbitrary plane (1-dimensional).

    This is the boundary curve only; see `EllipticalDisk` for the filled
    region. The local y-axis is the cross product of the plane normal and local x-axis.

    Args:
        center: The ellipse center.
        rx: The radius along the local x-axis (positive).
        ry: The radius along the local y-axis (positive).
        normal: The plane normal (nonzero). The default is the global z-axis.
        x_axis: The in-plane direction of ``rx``, perpendicular to ``normal``. If ``None``, a
            deterministic direction is selected; with the default normal, this is the global
            x-axis.
        label: An optional physical-group name.

    Raises:
        GeometryError: If a radius is not positive or the plane directions are invalid.
        LabelError: If ``label`` is invalid.
    """

    center: Vec3
    rx: float
    ry: float
    normal: Vec3
    x_axis: Vec3 | None

    def __init__(
        self,
        center: Vec3Like,
        rx: float,
        ry: float,
        *,
        normal: Vec3Like = (0, 0, 1),
        x_axis: Vec3Like | None = None,
        label: str | None = None,
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "center", Vec3.from_any(center))
        object.__setattr__(self, "rx", rx)
        object.__setattr__(self, "ry", ry)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "x_axis", x_axis)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rx", _positive_float(self.rx, "rx"))
        object.__setattr__(self, "ry", _positive_float(self.ry, "ry"))
        normal, x_axis, _ = _plane_axes(self.normal, self.x_axis)
        object.__setattr__(self, "normal", normal)
        if self.x_axis is not None:
            object.__setattr__(self, "x_axis", x_axis)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 1

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        normal, x_axis, y_axis = _plane_axes(self.normal, self.x_axis)
        major_radius = max(self.rx, self.ry)
        minor_radius = min(self.rx, self.ry)
        major_axis = x_axis if self.rx >= self.ry else y_axis
        curve = gmsh.model.occ.addEllipse(
            self.center.x,
            self.center.y,
            self.center.z,
            major_radius,
            minor_radius,
            zAxis=_xyz(normal),
            xAxis=_xyz(major_axis),
        )
        return [(1, curve)]


@dataclass(frozen=True, slots=True, init=False)
class CircleArc(Shape):
    """A circular arc in an arbitrary plane (1-dimensional).

    The arc is the trajectory of rotating ``start`` around ``center`` by
    ``angle`` radians about ``normal``. Positive angles follow the right-hand
    rule; the sign of ``angle`` selects the sweep direction. Use `Circle` for a
    full circle.

    Args:
        center: The circle center.
        start: The arc start point on the circle.
        angle: The signed sweep angle in radians, with
            ``0 < abs(angle) < 2 * math.pi``.
        normal: The plane normal (nonzero). The default is the global z-axis.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``start`` coincides with ``center``, ``angle`` is zero
            or at least a full turn, ``normal`` is invalid, or ``start`` is not
            in the plane perpendicular to ``normal`` through ``center``.
        LabelError: If ``label`` is invalid.
    """

    center: Vec3
    start: Vec3
    angle: float
    normal: Vec3

    def __init__(
        self,
        center: Vec3Like,
        start: Vec3Like,
        angle: float,
        *,
        normal: Vec3Like = (0, 0, 1),
        label: str | None = None,
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "center", Vec3.from_any(center))
        object.__setattr__(self, "start", Vec3.from_any(start))
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "normal", normal)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "angle", _finite_float(self.angle, "angle"))
        normal = _normalize_direction(self.normal, "normal")
        object.__setattr__(self, "normal", normal)
        radial = self.start - self.center
        radius = radial.length()
        if radius == 0.0:
            raise GeometryError("CircleArc start and center must be distinct")
        if abs(normal.dot(radial)) > PLANE_ORTHOGONALITY_TOLERANCE * radius:
            raise GeometryError(
                "CircleArc start must lie in the plane perpendicular to normal through center"
            )
        if self.angle == 0.0:
            raise GeometryError("CircleArc angle must be nonzero")
        if abs(self.angle) >= 2.0 * math.pi:
            raise GeometryError(
                "CircleArc angle magnitude must be less than 2 * math.pi; "
                "use Circle for a full circle"
            )
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 1

    @property
    def radius(self) -> float:
        """The arc radius."""
        return (self.start - self.center).length()

    @property
    def end(self) -> Vec3:
        """The arc end point."""
        return self.center + _rotate_about_axis(
            self.start - self.center,
            self.angle,
            self.normal,
        )

    @property
    def start_tangent(self) -> Vec3:
        """The unit tangent at the arc start, pointing along the sweep."""
        tangent = self.normal.cross(self.start - self.center).normalized()
        return tangent if self.angle > 0.0 else -tangent

    @property
    def end_tangent(self) -> Vec3:
        """The unit tangent at the arc end, pointing along the sweep."""
        tangent = self.normal.cross(self.end - self.center).normalized()
        return tangent if self.angle > 0.0 else -tangent

    @property
    def _midpoint(self) -> Vec3:
        """The point halfway along the arc."""
        return self.center + _rotate_about_axis(
            self.start - self.center,
            self.angle / 2.0,
            self.normal,
        )

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        occ = gmsh.model.occ
        start = occ.addPoint(self.start.x, self.start.y, self.start.z, 0.0)
        midpoint = self._midpoint
        through = occ.addPoint(midpoint.x, midpoint.y, midpoint.z, 0.0)
        end = occ.addPoint(self.end.x, self.end.y, self.end.z, 0.0)
        curve = occ.addCircleArc(start, through, end, center=False)
        occ.remove([(0, through)], recursive=False)
        return [(1, curve)]


type PathSegment = Line | CircleArc


@dataclass(frozen=True, slots=True, init=False)
class Path(Shape):
    """An ordered, tangent-continuous chain of one-dimensional segments.

    Every consecutive pair of segments must connect end-to-start and share the
    same tangent direction. A path is closed when its first start and last end
    coincide; the closure joint must also be tangent-continuous. `Path` is the
    supported multi-segment guide for `ExtrudeAlong`.

    Args:
        segments: Two or more `Line` or `CircleArc` segments in path order.
        label: An optional physical-group name.

    Raises:
        GeometryError: If there are fewer than 2 segments, a segment is not a
            `Line` or `CircleArc`, consecutive segments are disconnected, or the
            path is not tangent-continuous.
        LabelError: If ``label`` is invalid.
    """

    segments: tuple[PathSegment, ...]

    def __init__(
        self,
        *args: Shape,
        segments: tuple[Shape, ...] | None = None,
        label: str | None = None,
    ) -> None:
        if segments is not None:
            if args:
                raise TypeError("segments cannot be supplied both positionally and by keyword")
            args = segments
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "segments", cast(tuple[PathSegment, ...], tuple(args)))
        self.__post_init__()

    def __post_init__(self) -> None:
        if len(self.segments) < 2:
            raise GeometryError("Path requires at least 2 segments")
        for segment in self.segments:
            if not isinstance(segment, Shape):
                raise GeometryError(
                    f"Path segments must be Shape instances, got {type(segment).__name__}"
                )
            if not isinstance(segment, (Line, CircleArc)):
                raise GeometryError("Path segments must be Line or CircleArc instances")
        scale = self._scale()
        for previous, current in pairwise(self.segments):
            if (previous.end - current.start).length() > CONNECTIVITY_TOLERANCE * scale:
                raise GeometryError("Path segments must connect end-to-start")
            if not _tangents_align(previous.end_tangent, current.start_tangent):
                raise GeometryError("Path segments must be tangent-continuous")
        if self.closed:
            last = self.segments[-1]
            first = self.segments[0]
            if not _tangents_align(last.end_tangent, first.start_tangent):
                raise GeometryError("closed Path must be tangent-continuous at the closure")
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 1

    @property
    def start(self) -> Vec3:
        """The start point of the first segment."""
        return self.segments[0].start

    @property
    def end(self) -> Vec3:
        """The end point of the last segment."""
        return self.segments[-1].end

    @property
    def closed(self) -> bool:
        """Whether the path is closed (first start equals last end)."""
        return (self.segments[0].start - self.segments[-1].end).length() <= (
            CONNECTIVITY_TOLERANCE * self._scale()
        )

    def _scale(self) -> float:
        points: list[Vec3] = []
        for segment in self.segments:
            points.append(segment.start)
            points.append(segment.end)
            if isinstance(segment, CircleArc):
                points.append(segment._midpoint)
        scale = _bbox_diagonal(points)
        return scale if scale > 0.0 else 1.0

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        curves: list[tuple[int, int]] = []
        for segment in self.segments:
            realized = segment._realize(gmsh)
            if len(realized) != 1 or realized[0][0] != 1:
                raise GeometryError("Path segments must realize as exactly one curve")
            curves.extend(realized)
        return curves


@dataclass(frozen=True, slots=True, init=False)
class Polygon(Shape):
    """A planar polygon (2-dimensional).

    Vertices must be distinct, non-collinear, and coplanar. The boundary is
    closed implicitly, so the final point must not repeat the first.

    Args:
        points: Three or more vertices.
        label: An optional physical-group name.

    Raises:
        GeometryError: If the vertices are not distinct, non-collinear, and
            coplanar.
        LabelError: If ``label`` is invalid.
    """

    points: tuple[Vec3, ...]

    def __init__(self, points: Sequence[Vec3Like], *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "points", _normalize_point_sequence(points, type(self).__name__))
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_polygon(self.points)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 2

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        occ = gmsh.model.occ
        points = [occ.addPoint(p.x, p.y, p.z, 0.0) for p in self.points]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        loop = occ.addCurveLoop(lines)
        return [(2, occ.addPlaneSurface([loop]))]


@dataclass(frozen=True, slots=True, init=False)
class Rectangle(Shape):
    """An axis-aligned rectangle in the plane ``z = origin.z`` (2-dimensional).

    Args:
        origin: The lower-left corner.
        dx: The width along ``x`` (positive).
        dy: The height along ``y`` (positive).
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``dx`` or ``dy`` is not positive.
        LabelError: If ``label`` is invalid.
    """

    origin: Vec3
    dx: float
    dy: float

    def __init__(self, origin: Vec3Like, dx: float, dy: float, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "origin", Vec3.from_any(origin))
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dx", _positive_float(self.dx, "dx"))
        object.__setattr__(self, "dy", _positive_float(self.dy, "dy"))
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 2

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        surf = gmsh.model.occ.addRectangle(
            self.origin.x, self.origin.y, self.origin.z, self.dx, self.dy
        )
        return [(2, surf)]


@dataclass(frozen=True, slots=True, init=False)
class Disk(Shape):
    """A filled circular disk in an arbitrary plane (2-dimensional).

    Args:
        center: The disk center.
        radius: The radius (positive).
        normal: The plane normal (nonzero). The default is the global z-axis.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``radius`` is not positive or ``normal`` is invalid.
        LabelError: If ``label`` is invalid.
    """

    center: Vec3
    radius: float
    normal: Vec3

    def __init__(
        self,
        center: Vec3Like,
        radius: float,
        *,
        normal: Vec3Like = (0, 0, 1),
        label: str | None = None,
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "center", Vec3.from_any(center))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "normal", normal)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive_float(self.radius, "radius"))
        normal, _, _ = _plane_axes(self.normal, None)
        object.__setattr__(self, "normal", normal)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 2

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        normal, x_axis, _ = _plane_axes(self.normal, None)
        surf = gmsh.model.occ.addDisk(
            self.center.x,
            self.center.y,
            self.center.z,
            self.radius,
            self.radius,
            zAxis=_xyz(normal),
            xAxis=_xyz(x_axis),
        )
        return [(2, surf)]


@dataclass(frozen=True, slots=True, init=False)
class EllipticalDisk(Shape):
    """A filled elliptical disk in an arbitrary plane (2-dimensional).

    The local y-axis is the cross product of the plane normal and local x-axis.

    Args:
        center: The ellipse center.
        rx: The radius along the local x-axis (positive).
        ry: The radius along the local y-axis (positive).
        normal: The plane normal (nonzero). The default is the global z-axis.
        x_axis: The in-plane direction of ``rx``, perpendicular to ``normal``. If ``None``, a
            deterministic direction is selected; with the default normal, this is the global
            x-axis.
        label: An optional physical-group name.

    Raises:
        GeometryError: If a radius is not positive or the plane directions are invalid.
        LabelError: If ``label`` is invalid.
    """

    center: Vec3
    rx: float
    ry: float
    normal: Vec3
    x_axis: Vec3 | None

    def __init__(
        self,
        center: Vec3Like,
        rx: float,
        ry: float,
        *,
        normal: Vec3Like = (0, 0, 1),
        x_axis: Vec3Like | None = None,
        label: str | None = None,
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "center", Vec3.from_any(center))
        object.__setattr__(self, "rx", rx)
        object.__setattr__(self, "ry", ry)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "x_axis", x_axis)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rx", _positive_float(self.rx, "rx"))
        object.__setattr__(self, "ry", _positive_float(self.ry, "ry"))
        normal, x_axis, _ = _plane_axes(self.normal, self.x_axis)
        object.__setattr__(self, "normal", normal)
        if self.x_axis is not None:
            object.__setattr__(self, "x_axis", x_axis)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 2

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        normal, x_axis, y_axis = _plane_axes(self.normal, self.x_axis)
        major_radius = max(self.rx, self.ry)
        minor_radius = min(self.rx, self.ry)
        major_axis = x_axis if self.rx >= self.ry else y_axis
        surf = gmsh.model.occ.addDisk(
            self.center.x,
            self.center.y,
            self.center.z,
            major_radius,
            minor_radius,
            zAxis=_xyz(normal),
            xAxis=_xyz(major_axis),
        )
        return [(2, surf)]


@dataclass(frozen=True, slots=True, init=False)
class Sphere(Shape):
    """A sphere (3-dimensional).

    Args:
        center: The center.
        radius: The radius (positive).
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``radius`` is not positive.
        LabelError: If ``label`` is invalid.
    """

    center: Vec3
    radius: float

    def __init__(self, center: Vec3Like, radius: float, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "center", Vec3.from_any(center))
        object.__setattr__(self, "radius", radius)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive_float(self.radius, "radius"))
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 3

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        vol = gmsh.model.occ.addSphere(self.center.x, self.center.y, self.center.z, self.radius)
        return [(3, vol)]


@dataclass(frozen=True, slots=True, init=False)
class Box(Shape):
    """An axis-aligned rectangular box (3-dimensional).

    Args:
        origin: The lower corner.
        dx: The extent along ``x`` (positive).
        dy: The extent along ``y`` (positive).
        dz: The extent along ``z`` (positive).
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``dx``, ``dy``, or ``dz`` is not positive.
        LabelError: If ``label`` is invalid.
    """

    origin: Vec3
    dx: float
    dy: float
    dz: float

    def __init__(
        self, origin: Vec3Like, dx: float, dy: float, dz: float, *, label: str | None = None
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "origin", Vec3.from_any(origin))
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        object.__setattr__(self, "dz", dz)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dx", _positive_float(self.dx, "dx"))
        object.__setattr__(self, "dy", _positive_float(self.dy, "dy"))
        object.__setattr__(self, "dz", _positive_float(self.dz, "dz"))
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 3

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        vol = gmsh.model.occ.addBox(
            self.origin.x, self.origin.y, self.origin.z, self.dx, self.dy, self.dz
        )
        return [(3, vol)]


@dataclass(frozen=True, slots=True, init=False)
class Cylinder(Shape):
    """A right circular cylinder (3-dimensional).

    The axis runs from ``start`` to ``end``; ``radius`` is the cylinder's
    radius.

    Args:
        start: One endpoint of the axis.
        end: The other endpoint of the axis (must differ from ``start``).
        radius: The radius (positive).
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``start`` and ``end`` coincide, or ``radius`` is not
            positive.
        LabelError: If ``label`` is invalid.
    """

    start: Vec3
    end: Vec3
    radius: float

    def __init__(
        self, start: Vec3Like, end: Vec3Like, radius: float, *, label: str | None = None
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "start", Vec3.from_any(start))
        object.__setattr__(self, "end", Vec3.from_any(end))
        object.__setattr__(self, "radius", radius)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive_float(self.radius, "radius"))
        if self.start == self.end:
            raise GeometryError("Cylinder start and end points must be distinct")
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return 3

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        axis = self.end - self.start
        vol = gmsh.model.occ.addCylinder(
            self.start.x, self.start.y, self.start.z, axis.x, axis.y, axis.z, self.radius
        )
        return [(3, vol)]


# -- validation helpers ----------------------------------------------------


def _tangents_align(first: Vec3, second: Vec3) -> bool:
    """Return whether two unit tangents point the same way within tolerance."""
    return first.dot(second) > 0.0 and first.cross(second).length() <= TANGENCY_TOLERANCE


def _validate_polyline(points: tuple[Vec3, ...]) -> None:
    if len(points) < 2:
        raise GeometryError("Polyline requires at least 2 points")
    for a, b in pairwise(points):
        if a == b:
            raise GeometryError("Polyline has consecutive identical points")
    if points[0] == points[-1] and len(set(points)) < 3:
        raise GeometryError("a closed Polyline requires at least 3 distinct points")


def _validate_polygon(points: tuple[Vec3, ...]) -> None:
    if len(points) < 3:
        raise GeometryError("Polygon requires at least 3 points")
    unique_count = len(set(points))
    if unique_count < 3:
        raise GeometryError("Polygon requires at least 3 distinct points")
    if points[0] == points[-1]:
        raise GeometryError("Polygon must not repeat the first point as the final point")
    for a, b in pairwise(points):
        if a == b:
            raise GeometryError("Polygon has consecutive identical points")
    if unique_count != len(points):
        raise GeometryError("Polygon vertices must be distinct")
    scale = _bbox_diagonal(points)
    normal = _find_polygon_normal(points)
    if normal is None:
        raise GeometryError("Polygon points are collinear")
    tol = PLANARITY_TOLERANCE * scale
    origin = points[0]
    for p in points[1:]:
        if abs(normal.dot(p - origin)) > tol:
            raise GeometryError("Polygon points are not coplanar")


def _find_polygon_normal(points: tuple[Vec3, ...]) -> Vec3 | None:
    """Return a unit normal from the first non-collinear triple, else ``None``."""
    origin = points[0]
    for i in range(1, len(points) - 1):
        v1 = points[i] - origin
        l1 = v1.length()
        if l1 == 0.0:
            continue
        for j in range(i + 1, len(points)):
            v2 = points[j] - origin
            l2 = v2.length()
            if l2 == 0.0:
                continue
            cross = v1.cross(v2)
            if cross.length() / (l1 * l2) > COLLINEARITY_TOLERANCE:
                return cross.normalized()
    return None


# -- booleans --------------------------------------------------------------


def _coerce_operands(
    operands: tuple[Shape | Sequence[Shape], ...],
) -> tuple[Shape, ...]:
    """Flatten a single-sequence boolean argument into individual operands.

    Booleans accept either variadic shapes (``Difference(a, b)``) or a single
    sequence (``Difference([a, b])``).
    """
    if len(operands) == 1:
        single = operands[0]
        if isinstance(single, (str, bytes)):
            raise GeometryError("boolean operands must be Shape instances or a sequence of Shape")
        if isinstance(single, Sequence):
            return tuple(single)
    return cast(tuple[Shape, ...], operands)


@dataclass(frozen=True, slots=True, init=False)
class _Boolean(Shape):
    """Base class for the n-ary boolean operations."""

    operands: tuple[Shape, ...]

    def __init__(
        self,
        *args: Shape | Sequence[Shape],
        label: str | None = None,
        operands: tuple[Shape, ...] | None = None,
    ) -> None:
        if operands is not None:
            if args:
                raise TypeError("operands cannot be supplied both positionally and by keyword")
            args = (operands,)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operands", _coerce_operands(args))
        self.__post_init__()

    @property
    def dimension(self) -> int:
        """The topological dimension of the shape."""
        return self.operands[0].dimension

    def __post_init__(self) -> None:
        _validate_boolean_operands(self.operands)
        Shape.__post_init__(self)

    def _filter_result(self, dimtags: list[tuple[int, int]]) -> list[tuple[int, int]]:
        result = [d for d in dimtags if d[0] == self.dimension]
        if not result:
            raise GeometryError(f"{type(self).__name__} produced an empty result")
        return result


class Union(_Boolean):
    """Boolean union of two or more same-dimension operands.

    ``Union(a, b, c)`` realizes ``a | b | c``. Union is associative, so the
    n-ary form equals nested binary unions.

    Args:
        operands: Two or more same-dimension shapes to unite, or a single
            sequence of them.
        label: An optional physical-group name.

    Raises:
        GeometryError: If there are fewer than 2 operands, the operands have
            mixed dimensions, or the result is empty.
        LabelError: If ``label`` is invalid.
    """

    __slots__ = ()

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        operands = [operand._realize(gmsh) for operand in self.operands]
        tools = [dimtag for operand in operands[1:] for dimtag in operand]
        result, _ = gmsh.model.occ.fuse(operands[0], tools, removeObject=True, removeTool=True)
        return self._filter_result(result)


class Intersection(_Boolean):
    """Boolean intersection of two or more same-dimension operands.

    ``Intersection(a, b, c)`` realizes ``a & b & c`` (the region common to all
    operands), by chaining pairwise intersections. Intersection is associative,
    so this equals ``Intersection(Intersection(a, b), c)``.

    Note: Gmsh's ``occ.intersect(object, tools)`` computes the common part of
    the object and the union of the tools instead, so the n-ary intersection
    cannot be expressed in a single Gmsh call.

    Args:
        operands: Two or more same-dimension shapes to intersect, or a single
            sequence of them.
        label: An optional physical-group name.

    Raises:
        GeometryError: If there are fewer than 2 operands, the operands have
            mixed dimensions, or the result is empty.
        LabelError: If ``label`` is invalid.
    """

    __slots__ = ()

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        result = self.operands[0]._realize(gmsh)
        for operand in self.operands[1:]:
            result, _ = gmsh.model.occ.intersect(
                result, operand._realize(gmsh), removeObject=True, removeTool=True
            )
        return self._filter_result(result)


class Difference(_Boolean):
    """Boolean difference: the first operand minus the union of the rest.

    ``Difference(a, b, c)`` realizes ``a - (b | c)``.

    Args:
        operands: The shape to subtract from, followed by one or more shapes
            to subtract; may also be a single sequence of shapes.
        label: An optional physical-group name.

    Raises:
        GeometryError: If there are fewer than 2 operands, the operands have
            mixed dimensions, or the result is empty.
        LabelError: If ``label`` is invalid.
    """

    __slots__ = ()

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        object_dimtags = self.operands[0]._realize(gmsh)
        tools = [dimtag for operand in self.operands[1:] for dimtag in operand._realize(gmsh)]
        result, _ = gmsh.model.occ.cut(object_dimtags, tools, removeObject=True, removeTool=True)
        return self._filter_result(result)


def _validate_boolean_operands(operands: tuple[Shape, ...]) -> None:
    if len(operands) < 2:
        raise GeometryError("a boolean operation requires at least 2 operands")
    for op in operands:
        if not isinstance(op, Shape):
            raise GeometryError(
                f"boolean operands must be Shape instances, got {type(op).__name__}"
            )
    if len({op.dimension for op in operands}) != 1:
        raise GeometryError("boolean operands must share the same topological dimension")


# -- rigid transforms -------------------------------------------------------


def _validate_transform_operand(operand: Shape) -> None:
    if not isinstance(operand, Shape):
        raise GeometryError(
            f"transform operand must be a Shape instance, got {type(operand).__name__}"
        )


@dataclass(frozen=True, slots=True, init=False)
class Translate(Shape):
    """Translation of a shape by an offset.

    The operand is realized first and its transient OCC entities are then translated. The operand's
    label is consumed; only the transform node's own label is used.

    Args:
        operand: The shape to translate.
        offset: The translation offset.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``operand`` is not a shape or ``offset`` is invalid.
        LabelError: If ``label`` is invalid.
    """

    operand: Shape
    offset: Vec3

    def __init__(self, operand: Shape, offset: Vec3Like, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "offset", Vec3.from_any(offset))
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_transform_operand(self.operand)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the transformed shape."""
        return self.operand.dimension

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        dimtags = self.operand._realize(gmsh)
        gmsh.model.occ.translate(dimtags, self.offset.x, self.offset.y, self.offset.z)
        return dimtags


@dataclass(frozen=True, slots=True, init=False)
class Rotate(Shape):
    """Rotation of a shape around an arbitrary axis.

    The angle follows the right-hand rule around the directed axis. The operand is realized first
    and its transient OCC entities are then rotated. The operand's label is consumed; only the
    transform node's own label is used.

    Args:
        operand: The shape to rotate.
        angle: The rotation angle in radians.
        pivot: A point on the rotation axis. The default is the global origin.
        axis: The rotation-axis direction (nonzero). The default is the global z-axis.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``operand`` is not a shape or the rotation geometry is invalid.
        LabelError: If ``label`` is invalid.
    """

    operand: Shape
    angle: float
    pivot: Vec3
    axis: Vec3

    def __init__(
        self,
        operand: Shape,
        angle: float,
        *,
        pivot: Vec3Like = (0, 0, 0),
        axis: Vec3Like = (0, 0, 1),
        label: str | None = None,
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "pivot", Vec3.from_any(pivot))
        object.__setattr__(self, "axis", axis)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_transform_operand(self.operand)
        object.__setattr__(self, "angle", _finite_float(self.angle, "angle"))
        object.__setattr__(self, "axis", _normalize_direction(self.axis, "axis"))
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the transformed shape."""
        return self.operand.dimension

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        dimtags = self.operand._realize(gmsh)
        gmsh.model.occ.rotate(
            dimtags,
            self.pivot.x,
            self.pivot.y,
            self.pivot.z,
            self.axis.x,
            self.axis.y,
            self.axis.z,
            self.angle,
        )
        return dimtags


# -- extrusions -------------------------------------------------------------


def _validate_extrusion_operand(operand: Shape, operation: str) -> None:
    if not isinstance(operand, Shape):
        raise GeometryError(
            f"{operation} operand must be a Shape instance, got {type(operand).__name__}"
        )
    if operand.dimension == 3:
        raise GeometryError(f"{operation} requires an operand with dimension at most 2")


def _filter_extrusion_result(
    dimtags: list[tuple[int, int]],
    dimension: int,
    operation: str,
) -> list[tuple[int, int]]:
    result = [dimtag for dimtag in dimtags if dimtag[0] == dimension]
    if not result:
        raise GeometryError(f"{operation} produced an empty result")
    return result


@dataclass(frozen=True, slots=True, init=False)
class Extrude(Shape):
    """Linear extrusion of a shape along an offset.

    Extrusion increases the operand's dimension by one. The operand's label is consumed; only the
    operation node's own label is used.

    Args:
        operand: The 0D, 1D, or 2D shape to extrude.
        offset: The nonzero extrusion offset.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``operand`` is not a shape, is 3D, or ``offset`` is zero or invalid.
        LabelError: If ``label`` is invalid.
    """

    operand: Shape
    offset: Vec3

    def __init__(self, operand: Shape, offset: Vec3Like, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "offset", Vec3.from_any(offset))
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_extrusion_operand(self.operand, type(self).__name__)
        if max(abs(self.offset.x), abs(self.offset.y), abs(self.offset.z)) == 0.0:
            raise GeometryError("offset must be nonzero")
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the extruded shape."""
        return self.operand.dimension + 1

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        source = self.operand._realize(gmsh)
        out = gmsh.model.occ.extrude(source, self.offset.x, self.offset.y, self.offset.z)
        result = _filter_extrusion_result(out, self.dimension, type(self).__name__)
        gmsh.model.occ.remove(source, recursive=True)
        return result


@dataclass(frozen=True, slots=True, init=False)
class Revolve(Shape):
    """Rotational extrusion of a shape around an arbitrary axis.

    Revolution increases the operand's dimension by one. The angle follows the right-hand rule
    around the directed axis. The operand's label is consumed; only the operation node's own label
    is used.

    Args:
        operand: The 0D, 1D, or 2D shape to revolve.
        angle: The nonzero rotation angle in radians, with magnitude at most ``2 * math.pi``.
        pivot: A point on the rotation axis. The default is the global origin.
        axis: The rotation-axis direction (nonzero). The default is the global z-axis.
        label: An optional physical-group name.

    Raises:
        GeometryError: If ``operand`` is not a shape, is 3D, or the revolution geometry is invalid.
        LabelError: If ``label`` is invalid.
    """

    operand: Shape
    angle: float
    pivot: Vec3
    axis: Vec3

    def __init__(
        self,
        operand: Shape,
        angle: float,
        *,
        pivot: Vec3Like = (0, 0, 0),
        axis: Vec3Like = (0, 0, 1),
        label: str | None = None,
    ) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "pivot", Vec3.from_any(pivot))
        object.__setattr__(self, "axis", axis)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_extrusion_operand(self.operand, type(self).__name__)
        angle = _finite_float(self.angle, "angle")
        if angle == 0.0:
            raise GeometryError("angle must be nonzero")
        if abs(angle) > 2 * math.pi:
            raise GeometryError("angle magnitude must not exceed 2 * math.pi")
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "axis", _normalize_direction(self.axis, "axis"))
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the revolved shape."""
        return self.operand.dimension + 1

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        source = self.operand._realize(gmsh)
        out = gmsh.model.occ.revolve(
            source,
            self.pivot.x,
            self.pivot.y,
            self.pivot.z,
            self.axis.x,
            self.axis.y,
            self.axis.z,
            self.angle,
        )
        result = _filter_extrusion_result(out, self.dimension, type(self).__name__)
        gmsh.model.occ.remove(source, recursive=True)
        return result


@dataclass(frozen=True, slots=True, init=False)
class ExtrudeAlong(Shape):
    """Extrusion of a profile along a smooth guide curve or path.

    The guide must be either exactly one curve suitable for `occ.addPipe` or a
    tangent-continuous `Path`. Lines, circles, ellipses, circular arcs, and
    revolution-generated arcs are supported as single-curve guides. Multi-curve
    `Union` and `Polyline` guides and sharp paths are rejected. Profile placement
    and orientation follow OCC pipe semantics; they are not inferred automatically.
    Extrusion increases the profile's dimension by one.

    Labels on both the profile and guide are consumed; only the operation node's own label is used.

    Args:
        operand: The 0D, 1D, or 2D profile to sweep.
        guide: The 1D shape defining the sweep path.
        label: An optional physical-group name.

    Raises:
        GeometryError: If the profile or guide has an invalid type or dimension, or the guide is
            not a single smooth curve or a `Path`.
        LabelError: If ``label`` is invalid.
    """

    operand: Shape
    guide: Shape

    def __init__(self, operand: Shape, guide: Shape, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(self, "guide", guide)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_extrusion_operand(self.operand, type(self).__name__)
        if not isinstance(self.guide, Shape):
            raise GeometryError(f"guide must be a Shape instance, got {type(self.guide).__name__}")
        if self.guide.dimension != 1:
            raise GeometryError("guide must be one-dimensional")
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        """The topological dimension of the swept shape."""
        return self.operand.dimension + 1

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        source = self.operand._realize(gmsh)
        guide = self.guide._realize(gmsh)
        if isinstance(self.guide, Path):
            if len(guide) != len(self.guide.segments):
                raise GeometryError("Path guide must realize as exactly one curve per segment")
        elif len(guide) != 1:
            raise GeometryError(
                "guide must realize as exactly one smooth curve or a Path; "
                "use Path(...) for multiple segments"
            )
        if any(dim != 1 for dim, _ in guide):
            raise GeometryError("guide must realize as one-dimensional curves")
        wire = gmsh.model.occ.addWire([tag for _, tag in guide], checkClosed=False)
        out = gmsh.model.occ.addPipe(source, wire, trihedron="DiscreteTrihedron")
        result = _filter_extrusion_result(out, self.dimension, type(self).__name__)
        gmsh.model.occ.remove(source, recursive=True)
        gmsh.model.occ.remove(guide, recursive=True)
        return result
