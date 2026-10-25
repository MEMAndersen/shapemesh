"""Shape primitives and n-ary boolean operations.

Shapes are immutable, hashable frozen dataclasses. Geometry is validated on
construction (raising :class:`GeometryError`), coordinates are normalized to
:class:`Vec3`, and ``label`` is validated by the base class (raising
:class:`LabelError`). A ``label`` with surrounding whitespace is stripped, which
emits a :class:`UserWarning` so callers know the stored label differs from the
one they passed.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from .constants import COLLINEARITY_TOLERANCE, PLANARITY_TOLERANCE
from .errors import GeometryError, LabelError

__all__ = [
    "Cube",
    "Cylinder",
    "Difference",
    "Intersection",
    "Line",
    "Point",
    "Polygon",
    "Polyline",
    "Rectangle",
    "Shape",
    "Sphere",
    "Union",
    "Vec3",
    "Vec3Like",
]


def _validate_label(label: str | None) -> str | None:
    """Return the normalized (stripped) label, or raise :class:`LabelError`."""
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


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise GeometryError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise GeometryError(f"{name} must be finite, got {value}")
    if value <= 0:
        raise GeometryError(f"{name} must be positive, got {value}")
    return float(value)


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
    """

    label: str | None = None

    @property
    def dimension(self) -> int:
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
    """A 3D coordinate with float components (2D input pads ``z=0``)."""

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
            if not math.isfinite(value):
                raise GeometryError("Vec3 coordinates must be finite")
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_any(cls, value: Vec3Like) -> Vec3:
        """Normalize a :data:`Vec3Like` into a :class:`Vec3`.

        Accepts a ``Vec3``, a sequence of 2-3 real numbers, or an ndarray.
        Two-component input pads ``z=0``. A :class:`Shape` (e.g. ``Point``) is
        rejected; pass ``.coord`` instead.
        """
        if isinstance(value, Vec3):
            return value
        if isinstance(value, Shape):
            raise GeometryError("a Shape cannot be used as a vertex coordinate; pass `.coord`")
        if isinstance(value, (str, bytes)):
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
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vec3:
        n = self.length()
        if n == 0.0:
            raise GeometryError("cannot normalize a zero-length vector")
        return Vec3(self.x / n, self.y / n, self.z / n)


type Vec3Like = Vec3 | Sequence[float] | np.ndarray


# -- primitives ------------------------------------------------------------


@dataclass(frozen=True, slots=True, init=False)
class Point(Shape):
    coord: Vec3

    def __init__(self, coord: Vec3Like, *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "coord", Vec3.from_any(coord))
        self.__post_init__()

    def __post_init__(self) -> None:
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        return 0

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        tag = gmsh.model.occ.addPoint(self.coord.x, self.coord.y, self.coord.z, 0.0)
        return [(0, tag)]


@dataclass(frozen=True, slots=True, init=False)
class Line(Shape):
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
        return 1

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        occ = gmsh.model.occ
        p1 = occ.addPoint(self.start.x, self.start.y, self.start.z, 0.0)
        p2 = occ.addPoint(self.end.x, self.end.y, self.end.z, 0.0)
        return [(1, occ.addLine(p1, p2))]


@dataclass(frozen=True, slots=True, init=False)
class Polyline(Shape):
    points: tuple[Vec3, ...]

    def __init__(self, points: Sequence[Vec3Like], *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "points", tuple(Vec3.from_any(p) for p in points))
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_polyline(self.points)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        return 1

    @property
    def closed(self) -> bool:
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
class Polygon(Shape):
    points: tuple[Vec3, ...]

    def __init__(self, points: Sequence[Vec3Like], *, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "points", tuple(Vec3.from_any(p) for p in points))
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_polygon(self.points)
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        return 2

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        occ = gmsh.model.occ
        points = [occ.addPoint(p.x, p.y, p.z, 0.0) for p in self.points]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        loop = occ.addCurveLoop(lines)
        return [(2, occ.addPlaneSurface([loop]))]


@dataclass(frozen=True, slots=True, init=False)
class Rectangle(Shape):
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
        if self.origin.z != 0.0:
            raise GeometryError("Rectangle origin must lie in the z = 0 plane")
        object.__setattr__(self, "dx", _positive_float(self.dx, "dx"))
        object.__setattr__(self, "dy", _positive_float(self.dy, "dy"))
        Shape.__post_init__(self)

    @property
    def dimension(self) -> int:
        return 2

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        surf = gmsh.model.occ.addRectangle(
            self.origin.x, self.origin.y, self.origin.z, self.dx, self.dy
        )
        return [(2, surf)]


@dataclass(frozen=True, slots=True, init=False)
class Sphere(Shape):
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
        return 3

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        vol = gmsh.model.occ.addSphere(self.center.x, self.center.y, self.center.z, self.radius)
        return [(3, vol)]


@dataclass(frozen=True, slots=True, init=False)
class Cube(Shape):
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
        return 3

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        vol = gmsh.model.occ.addBox(
            self.origin.x, self.origin.y, self.origin.z, self.dx, self.dy, self.dz
        )
        return [(3, vol)]


@dataclass(frozen=True, slots=True, init=False)
class Cylinder(Shape):
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
        return 3

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        axis = self.end - self.start
        vol = gmsh.model.occ.addCylinder(
            self.start.x, self.start.y, self.start.z, axis.x, axis.y, axis.z, self.radius
        )
        return [(3, vol)]


# -- validation helpers ----------------------------------------------------


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
    if len(set(points)) < 3:
        raise GeometryError("Polygon requires at least 3 distinct points")
    if points[0] == points[-1]:
        raise GeometryError("Polygon must not repeat the first point as the final point")
    for a, b in pairwise(points):
        if a == b:
            raise GeometryError("Polygon has consecutive identical points")
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


@dataclass(frozen=True, slots=True, init=False)
class _Boolean(Shape):
    operands: tuple[Shape, ...]

    def __init__(self, *operands: Shape, label: str | None = None) -> None:
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operands", operands)
        self.__post_init__()

    @property
    def dimension(self) -> int:
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

    ``Union(a, b, c)`` realizes the set union of all operands. Union is
    associative, so the n-ary form equals nested binary unions.
    """

    __slots__ = ()

    def _realize(self, gmsh: Any) -> list[tuple[int, int]]:
        operands = [operand._realize(gmsh) for operand in self.operands]
        tools = [dimtag for operand in operands[1:] for dimtag in operand]
        result, _ = gmsh.model.occ.fuse(operands[0], tools, removeObject=True, removeTool=True)
        return self._filter_result(result)


class Intersection(_Boolean):
    """Boolean intersection of two or more same-dimension operands.

    ``Intersection(a, b, c)`` realizes the region common to all operands, by
    chaining pairwise intersections. Intersection is associative, so this equals
    ``Intersection(Intersection(a, b), c)``.

    Note: Gmsh's ``occ.intersect(object, tools)`` computes the common part of
    the object and the union of the tools instead, so the n-ary intersection
    cannot be expressed in a single Gmsh call.
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

    ``Difference(a, b, c)`` realizes ``a`` minus the union of ``b`` and ``c``.
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
