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

"""Mesh data structures and MSH 4.1 ASCII reader/writer.

The `Mesh` model mirrors Gmsh's MSH 4.1 layout: a global node table, an
entity-classified element collection, entity records (boundary + bbox), and
physical groups. I/O needs no live Gmsh session.
"""

from __future__ import annotations

import shlex
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .errors import MeshFormatError

__all__ = [
    "ElementBlock",
    "ElementType",
    "EntityRecord",
    "Mesh",
    "MeshMetadata",
    "NodeTable",
    "NumpyElementBlock",
    "NumpyMesh",
    "PhysicalGroup",
]


# -- element types ---------------------------------------------------------


class ElementType(IntEnum):
    """Hand-curated fixed-node element types (MSH 4.1 codes).

    Only fixed-node types are listed; variable-node types (Polygon 34,
    Polyhedron 35) are excluded. Unknown codes are tolerated on read but not
    produced by the writer.
    """

    # points
    POINT1 = 15
    # lines
    LINE2 = 1
    LINE3 = 8
    LINE4 = 26
    LINE5 = 27
    LINE6 = 28
    # triangles
    TRI3 = 2
    TRI6 = 9
    TRI10 = 21
    TRI15 = 23
    TRI21 = 25
    # quadrilaterals
    QUAD4 = 3
    QUAD9 = 10
    QUAD8 = 16
    QUAD16 = 36
    QUAD25 = 37
    QUAD36 = 38
    # tetrahedra
    TET4 = 4
    TET10 = 11
    TET20 = 29
    TET35 = 30
    TET56 = 31
    # hexahedra
    HEX8 = 5
    HEX27 = 12
    HEX20 = 17
    HEX32 = 99
    HEX44 = 100
    HEX56 = 101
    HEX64 = 92
    HEX125 = 93
    HEX216 = 94
    # prisms
    PRISM6 = 6
    PRISM18 = 13
    PRISM15 = 18
    # pyramids
    PYRAMID5 = 7
    PYRAMID14 = 14
    PYRAMID13 = 19
    PYRAMID30 = 118
    PYRAMID55 = 119


_NODE_COUNT: dict[ElementType, int] = {
    ElementType.POINT1: 1,
    ElementType.LINE2: 2,
    ElementType.LINE3: 3,
    ElementType.LINE4: 4,
    ElementType.LINE5: 5,
    ElementType.LINE6: 6,
    ElementType.TRI3: 3,
    ElementType.TRI6: 6,
    ElementType.TRI10: 10,
    ElementType.TRI15: 15,
    ElementType.TRI21: 21,
    ElementType.QUAD4: 4,
    ElementType.QUAD9: 9,
    ElementType.QUAD8: 8,
    ElementType.QUAD16: 16,
    ElementType.QUAD25: 25,
    ElementType.QUAD36: 36,
    ElementType.TET4: 4,
    ElementType.TET10: 10,
    ElementType.TET20: 20,
    ElementType.TET35: 35,
    ElementType.TET56: 56,
    ElementType.HEX8: 8,
    ElementType.HEX27: 27,
    ElementType.HEX20: 20,
    ElementType.HEX32: 32,
    ElementType.HEX44: 44,
    ElementType.HEX56: 56,
    ElementType.HEX64: 64,
    ElementType.HEX125: 125,
    ElementType.HEX216: 216,
    ElementType.PRISM6: 6,
    ElementType.PRISM18: 18,
    ElementType.PRISM15: 15,
    ElementType.PYRAMID5: 5,
    ElementType.PYRAMID14: 14,
    ElementType.PYRAMID13: 13,
    ElementType.PYRAMID30: 30,
    ElementType.PYRAMID55: 55,
}

_CODE_TO_TYPE: dict[int, ElementType] = {int(t): t for t in ElementType}


# -- data structures -------------------------------------------------------


_DENSE_LOOKUP_MAX_RATIO = 4
_DENSE_LOOKUP_MAX_BYTES = 64 * 1024**2
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True, slots=True)
class EntityRecord:
    """A topological entity: dimension, tag, bounding box, and boundary.

    Args:
        dim: The entity dimension (0-3).
        tag: The entity tag.
        bbox: The bounding box ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
        boundary: Ordered ``(dim, tag)`` pairs of the entity's boundary.
    """

    dim: int
    tag: int
    bbox: tuple[float, float, float, float, float, float]
    boundary: tuple[tuple[int, int], ...] = ()

    def __repr__(self) -> str:
        return f"EntityRecord(dim={self.dim}, tag={self.tag})"


@dataclass(frozen=True, slots=True)
class _NodeLookup:
    strategy: str
    first_tag: int = 0
    min_tag: int = 0
    max_tag: int = 0
    direct: np.ndarray | None = None
    sorted_tags: np.ndarray | None = None
    order: np.ndarray | None = None

    def resolve(self, tags: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray]:
        if self.strategy == "contiguous":
            rows = tags - self.first_tag
            missing = (tags < 0) | (rows < 0) | (rows >= node_count)
            return rows.astype(np.intp, copy=False), missing

        if self.strategy == "direct":
            assert self.direct is not None
            in_range = (tags >= 0) & (tags >= self.min_tag) & (tags <= self.max_tag)
            safe_tags = np.where(in_range, tags, self.min_tag)
            offsets = (safe_tags - self.min_tag).astype(np.intp, copy=False)
            rows = self.direct[offsets]
            missing = ~in_range | (rows < 0)
            return rows.astype(np.intp, copy=False), missing

        assert self.sorted_tags is not None
        positions = np.searchsorted(self.sorted_tags, tags)
        missing = (tags < 0) | (positions >= node_count)
        if node_count == 0:
            return positions, missing
        safe_positions = np.minimum(positions, node_count - 1)
        missing |= self.sorted_tags[safe_positions] != tags
        rows = positions if self.order is None else self.order[safe_positions]
        return rows.astype(np.intp, copy=False), missing


@dataclass(frozen=True, slots=True, eq=False)
class NodeTable:
    """The global node table: tags, coordinates, and entity classification.

    Node tags are identifiers, not row indices. Use `NodeTable.indices_for` or
    `NodeTable.coords_for` to resolve them. All node-aligned arrays are frozen
    at construction.

    Args:
        tags: Node tags, shape ``(N,)``.
        coords: Node coordinates, shape ``(N, 3)``.
        entity_dim: Owning entity dimension per node, shape ``(N,)``.
        entity_tag: Owning entity tag per node, shape ``(N,)``.
    """

    tags: np.ndarray  # (N,) int
    coords: np.ndarray  # (N, 3) float
    entity_dim: np.ndarray | None = None  # (N,) int, -1 if unknown
    entity_tag: np.ndarray | None = None
    _lookup: _NodeLookup | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _freeze(self.tags)
        _freeze(self.coords)
        if self.entity_dim is not None:
            _freeze(self.entity_dim)
        if self.entity_tag is not None:
            _freeze(self.entity_tag)

    def __repr__(self) -> str:
        return f"NodeTable(nodes={int(self.tags.shape[0])})"

    def _get_lookup(self) -> _NodeLookup:
        lookup = self._lookup
        if lookup is None:
            lookup = _build_node_lookup(self.tags)
            object.__setattr__(self, "_lookup", lookup)
        return lookup

    def indices_for(self, tags: ArrayLike) -> int | np.ndarray:
        """Return zero-based row indices corresponding to node tags.

        Args:
            tags: An integer node tag or an arbitrarily shaped integer
                array-like of node tags.

        Returns:
            A Python ``int`` for scalar input; otherwise a writable array with
            the input shape and dtype ``np.intp``. Query order and duplicate
            tags are preserved.

        Raises:
            TypeError: If a non-empty query contains non-integer tags.
            KeyError: If any requested tag is missing or negative.
            ValueError: If the node table contains duplicate tags.
        """
        original, query, scalar = _normalize_node_query(tags)
        lookup = self._get_lookup()
        if query.size == 0:
            return np.empty(query.shape, dtype=np.intp)
        rows, missing = lookup.resolve(query, len(self.tags))
        if np.any(missing):
            _raise_missing_node_tags(original, missing)
        if scalar:
            return int(rows.item())
        return rows

    def coords_for(self, tags: ArrayLike) -> np.ndarray:
        """Return a copy of the coordinates corresponding to node tags.

        Scalar input returns shape ``(3,)``. Array input returns shape
        ``tags.shape + (3,)``; large connectivity queries therefore materialize
        one coordinate triplet per requested tag.

        Args:
            tags: An integer node tag or an arbitrarily shaped integer
                array-like of node tags.

        Returns:
            A new, writable coordinate array. Modifying it does not modify the
            node table.

        Raises:
            TypeError: If a non-empty query contains non-integer tags.
            KeyError: If any requested tag is missing or negative.
            ValueError: If the node table contains duplicate tags.
        """
        rows = self.indices_for(tags)
        if isinstance(rows, int):
            return self.coords[rows].copy()
        return self.coords[rows]


@dataclass(frozen=True, slots=True, eq=False)
class ElementBlock:
    """A homogeneous block of elements on one entity.

    Args:
        entity_dim: The owning entity dimension.
        entity_tag: The owning entity tag.
        type: The element type.
        tags: Element tags, shape ``(M,)``.
        connectivity: Element-to-node connectivity, shape ``(M, k)``.
    """

    entity_dim: int
    entity_tag: int
    type: ElementType
    tags: np.ndarray  # (M,) int
    connectivity: np.ndarray  # (M, k) int, global node tags

    def __repr__(self) -> str:
        return (
            f"ElementBlock(entity=({self.entity_dim}, {self.entity_tag}), "
            f"type={self.type.name}, elements={int(self.tags.shape[0])})"
        )


@dataclass(frozen=True, slots=True)
class PhysicalGroup:
    """A named physical group: a label over a set of entities.

    Args:
        name: The group name (a shape label).
        dim: The group dimension.
        tag: The physical group tag.
        entities: The ``(dim, tag)`` entities in the group.
    """

    name: str
    dim: int
    tag: int
    entities: tuple[tuple[int, int], ...]

    def __repr__(self) -> str:
        return (
            f"PhysicalGroup(name={self.name!r}, dim={self.dim}, "
            f"tag={self.tag}, entities={len(self.entities)})"
        )


@dataclass(frozen=True, slots=True)
class MeshMetadata:
    """Descriptive metadata about a mesh.

    Args:
        dim: The model dimension.
        msh_version: The MSH format version.
        gmsh_version: The Gmsh version that produced the mesh, if known.
        source: The source path, if read from disk.
    """

    dim: int
    msh_version: str = "4.1"
    gmsh_version: str | None = None
    source: str | None = None

    def __repr__(self) -> str:
        return (
            f"MeshMetadata(dim={self.dim}, msh_version={self.msh_version!r}, "
            f"gmsh_version={self.gmsh_version!r}, source={self.source!r})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class NumpyElementBlock:
    """A numpy-view element block with 0-based index connectivity.

    Args:
        connectivity: Shape ``(M, k)``; 0-based indices into ``coords``.
        node_tags: Shape ``(M, k)``; the original Gmsh node tags.
    """

    connectivity: np.ndarray  # (M, k) int, 0-based indices into coords
    node_tags: np.ndarray  # (M, k) int, original Gmsh tags

    def __repr__(self) -> str:
        return (
            f"NumpyElementBlock(elements={int(self.connectivity.shape[0])}, "
            f"nodes_per_element={int(self.connectivity.shape[1])})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class NumpyMesh:
    """A mesh represented as numpy arrays (see `Mesh.to_numpy`).

    Args:
        coords: Global node coordinates, shape ``(N, 3)``.
        blocks: One `NumpyElementBlock` per element type.
    """

    coords: np.ndarray  # (N, 3) float, global
    blocks: dict[ElementType, NumpyElementBlock]

    def __repr__(self) -> str:
        types = ", ".join(t.name for t in self.blocks)
        return f"NumpyMesh(nodes={int(self.coords.shape[0])}, blocks=[{types}])"


# -- helpers ---------------------------------------------------------------


def _freeze(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def _normalize_node_query(tags: ArrayLike) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        original = np.asarray(tags)
    except (TypeError, ValueError) as exc:
        raise TypeError("node tags must be an integer scalar or array-like") from exc
    scalar = original.ndim == 0
    if original.size == 0:
        return original, np.empty(original.shape, dtype=np.int64), scalar

    if original.dtype == np.dtype(object):
        values = original.ravel()
        if not all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            raise TypeError("node tags must contain only integers")
        query = np.empty(original.shape, dtype=np.int64)
        for index, value in enumerate(values):
            integer = int(value)
            query.flat[index] = integer if _INT64_MIN <= integer <= _INT64_MAX else -1
        return original, query, scalar

    if not np.issubdtype(original.dtype, np.integer):
        raise TypeError("node tags must contain only integers")
    if np.issubdtype(original.dtype, np.unsignedinteger):
        too_large = original > _INT64_MAX
        query = original.astype(np.int64)
        query[too_large] = -1
        return original, query, scalar
    return original, original.astype(np.int64, copy=False), scalar


def _build_node_lookup(tags: np.ndarray) -> _NodeLookup:
    node_count = len(tags)
    if node_count == 0:
        return _NodeLookup("contiguous")
    if node_count == 1 or bool(np.all((tags[1:] - tags[:-1] == 1) & (tags[1:] > tags[:-1]))):
        return _NodeLookup("contiguous", first_tag=int(tags[0]))

    min_tag = int(tags.min())
    max_tag = int(tags.max())
    range_size = max_tag - min_tag + 1
    index_dtype = np.dtype(np.int32 if node_count < 2**31 else np.int64)
    if (
        min_tag >= 0
        and range_size <= _DENSE_LOOKUP_MAX_RATIO * node_count
        and range_size * index_dtype.itemsize <= _DENSE_LOOKUP_MAX_BYTES
    ):
        direct = np.full(range_size, -1, dtype=index_dtype)
        direct[tags - min_tag] = np.arange(node_count, dtype=index_dtype)
        if np.count_nonzero(direct >= 0) != node_count:
            raise ValueError("node tags must be unique")
        return _NodeLookup(
            "direct",
            min_tag=min_tag,
            max_tag=max_tag,
            direct=_freeze(direct),
        )

    if bool(np.all(tags[1:] > tags[:-1])):
        return _NodeLookup("sorted", sorted_tags=tags)

    order = np.argsort(tags)
    sorted_tags = tags[order]
    if np.any(sorted_tags[1:] == sorted_tags[:-1]):
        raise ValueError("node tags must be unique")
    return _NodeLookup(
        "sorted",
        sorted_tags=_freeze(sorted_tags),
        order=_freeze(order),
    )


def _raise_missing_node_tags(original: np.ndarray, missing: np.ndarray) -> None:
    missing_tags = original.ravel()[missing.ravel()]
    preview = missing_tags[:5].tolist()
    raise KeyError(f"node tags not found: {preview}")


def _iter_tokens(body: list[str]) -> Iterator[str]:
    for line in body:
        yield from line.split()


def _read_int(it: Iterator[str]) -> int:
    try:
        return int(next(it))
    except (StopIteration, ValueError) as exc:
        raise MeshFormatError("truncated or malformed MSH section") from exc


def _read_float(it: Iterator[str]) -> float:
    try:
        return float(next(it))
    except (StopIteration, ValueError) as exc:
        raise MeshFormatError("truncated or malformed MSH section") from exc


def _parse_int_line(line: str, section: str) -> list[int]:
    try:
        return [int(token) for token in line.split()]
    except ValueError as exc:
        raise MeshFormatError(f"malformed {section}") from exc


def _reject_extra_tokens(it: Iterator[str], section: str) -> None:
    try:
        next(it)
    except StopIteration:
        return
    raise MeshFormatError(f"{section} contains extra data")


def _validate_tag_summary(
    section: str,
    tags: list[int],
    declared_count: int,
    declared_min: int,
    declared_max: int,
) -> None:
    if len(tags) != declared_count:
        raise MeshFormatError(f"{section} declares {declared_count} entries but parsed {len(tags)}")
    actual_min, actual_max = (min(tags), max(tags)) if tags else (0, 0)
    if (actual_min, actual_max) != (declared_min, declared_max):
        raise MeshFormatError(
            f"{section} declares tag range {declared_min}..{declared_max}, "
            f"but parsed {actual_min}..{actual_max}"
        )


def _validate_unique_node_tags(tags: np.ndarray) -> None:
    if tags.size < 2:
        return
    sorted_tags = np.sort(tags)
    duplicate_mask = sorted_tags[1:] == sorted_tags[:-1]
    if np.any(duplicate_mask):
        duplicates = sorted_tags[:-1][duplicate_mask]
        preview = duplicates[:5].tolist()
        raise MeshFormatError(f"$Nodes contains duplicate node tags: {preview}")


def _read_bbox(it: Iterator[str]) -> tuple[float, float, float, float, float, float]:
    return (
        _read_float(it),
        _read_float(it),
        _read_float(it),
        _read_float(it),
        _read_float(it),
        _read_float(it),
    )


# -- reading ---------------------------------------------------------------


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    i = 0
    n = len(lines)
    while i < n:
        header = lines[i].strip()
        if header.startswith("$") and not header.startswith("$End"):
            name = header[1:].strip()
            body: list[str] = []
            i += 1
            while i < n:
                line = lines[i].strip()
                if line == f"$End{name}":
                    break
                body.append(line)
                i += 1
            if i >= n:
                raise MeshFormatError(f"missing $End{name}")
            sections[name] = body
        i += 1
    return sections


def _parse_mesh_format(body: list[str]) -> None:
    toks = list(_iter_tokens(body))
    if len(toks) != 3:
        raise MeshFormatError("malformed $MeshFormat")
    version = toks[0]
    try:
        file_type = int(toks[1])
    except ValueError as exc:
        raise MeshFormatError("malformed $MeshFormat") from exc
    if version != "4.1":
        raise MeshFormatError(f"unsupported MSH version {version!r}; only 4.1 is supported")
    if file_type != 0:
        raise MeshFormatError("binary MSH is not supported; only ASCII")


def _has_binary_mesh_format(data: bytes) -> bool:
    header = data.split(b"$EndMeshFormat", 1)[0]
    tokens = header.split()
    try:
        index = tokens.index(b"$MeshFormat")
    except ValueError:
        return False
    return len(tokens) >= index + 3 and tokens[index + 2] == b"1"


def _parse_physical_names(body: list[str]) -> list[tuple[str, int, int]]:
    lines = [ln for ln in body if ln.strip()]
    if not lines:
        return []
    try:
        count = int(lines[0])
    except ValueError as exc:
        raise MeshFormatError("malformed $PhysicalNames header") from exc
    if count < 0 or len(lines) != count + 1:
        raise MeshFormatError(
            f"$PhysicalNames declares {count} names but contains {len(lines) - 1}"
        )
    groups: list[tuple[str, int, int]] = []
    for line in lines[1 : 1 + count]:
        try:
            parts = shlex.split(line, posix=False)
        except ValueError as exc:
            raise MeshFormatError(f"malformed $PhysicalNames line: {line!r}") from exc
        if len(parts) != 3:
            raise MeshFormatError(f"malformed $PhysicalNames line: {line!r}")
        try:
            dim, tag = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise MeshFormatError(f"malformed $PhysicalNames line: {line!r}") from exc
        name = parts[2]
        if name[:1] in ('"', "'") or name[-1:] in ('"', "'"):
            if len(name) < 2 or name[0] != name[-1] or name[0] not in ('"', "'"):
                raise MeshFormatError(f"malformed $PhysicalNames line: {line!r}")
            name = name[1:-1]
        groups.append((name, dim, tag))
    return groups


def _parse_entities(body: list[str]) -> tuple[list[EntityRecord], list[list[int]]]:
    """Return entity records and, for each, its physical-group tags."""
    it = _iter_tokens(body)
    try:
        n_points = _read_int(it)
        n_curves = _read_int(it)
        n_surfaces = _read_int(it)
        n_volumes = _read_int(it)
    except MeshFormatError:
        raise MeshFormatError("malformed $Entities header") from None
    if any(count < 0 for count in (n_points, n_curves, n_surfaces, n_volumes)):
        raise MeshFormatError("$Entities counts must be non-negative")

    records: list[EntityRecord] = []
    phys_tags: list[list[int]] = []

    def _add(
        dim: int,
        tag: int,
        bbox: tuple[float, float, float, float, float, float],
        boundary: tuple[tuple[int, int], ...],
    ) -> None:
        nphys = _read_int(it)
        phys = [_read_int(it) for _ in range(nphys)]
        records.append(EntityRecord(dim, tag, bbox, boundary))
        phys_tags.append(phys)

    for _ in range(n_points):
        tag = _read_int(it)
        x, y, z = _read_float(it), _read_float(it), _read_float(it)
        _add(0, tag, (x, y, z, x, y, z), ())
    for _ in range(n_curves):
        tag = _read_int(it)
        bbox = _read_bbox(it)
        _add(1, tag, bbox, ())
        nbound = _read_int(it)
        boundary = tuple((0, _read_int(it)) for _ in range(nbound))
        records[-1] = EntityRecord(1, tag, bbox, boundary)
    for _ in range(n_surfaces):
        tag = _read_int(it)
        bbox = _read_bbox(it)
        _add(2, tag, bbox, ())
        nbound = _read_int(it)
        boundary = tuple((1, _read_int(it)) for _ in range(nbound))
        records[-1] = EntityRecord(2, tag, bbox, boundary)
    for _ in range(n_volumes):
        tag = _read_int(it)
        bbox = _read_bbox(it)
        _add(3, tag, bbox, ())
        nbound = _read_int(it)
        boundary = tuple((2, _read_int(it)) for _ in range(nbound))
        records[-1] = EntityRecord(3, tag, bbox, boundary)

    _reject_extra_tokens(it, "$Entities")
    return records, phys_tags


def _parse_nodes(body: list[str]) -> NodeTable:
    it = _iter_tokens(body)
    n_blocks = _read_int(it)
    n_nodes = _read_int(it)
    min_node_tag = _read_int(it)
    max_node_tag = _read_int(it)
    if n_blocks < 0 or n_nodes < 0:
        raise MeshFormatError("$Nodes counts must be non-negative")
    tags: list[int] = []
    coords: list[list[float]] = []
    dims: list[int] = []
    etags: list[int] = []
    for _ in range(n_blocks):
        dim = _read_int(it)
        tag = _read_int(it)
        parametric = _read_int(it)
        if parametric:
            raise MeshFormatError("parametric $Nodes blocks are not supported")
        n = _read_int(it)
        if n < 0:
            raise MeshFormatError("$Nodes block counts must be non-negative")
        block_tags = [_read_int(it) for _ in range(n)]
        block_coords = [[_read_float(it) for _ in range(3)] for _ in range(n)]
        tags.extend(block_tags)
        coords.extend(block_coords)
        dims.extend([dim] * n)
        etags.extend([tag] * n)
    _reject_extra_tokens(it, "$Nodes")
    _validate_tag_summary("$Nodes", tags, n_nodes, min_node_tag, max_node_tag)
    tag_array = np.array(tags, dtype=np.int64)
    _validate_unique_node_tags(tag_array)
    return NodeTable(
        tag_array,
        np.array(coords, dtype=np.float64).reshape(-1, 3),
        np.array(dims, dtype=np.int64),
        np.array(etags, dtype=np.int64),
    )


def _parse_elements(body: list[str]) -> tuple[list[ElementBlock], list[int]]:
    lines = [ln for ln in body if ln.strip()]
    if not lines:
        return [], []
    header = _parse_int_line(lines[0], "$Elements")
    if len(header) != 4:
        raise MeshFormatError("malformed $Elements header")
    n_blocks, n_elements, min_element_tag, max_element_tag = header
    if n_blocks < 0 or n_elements < 0:
        raise MeshFormatError("$Elements counts must be non-negative")

    idx = 1

    def next_line() -> str:
        nonlocal idx
        if idx >= len(lines):
            raise MeshFormatError("truncated $Elements section")
        line = lines[idx]
        idx += 1
        return line

    raw: list[tuple[int, int, ElementType, list[int], list[list[int]]]] = []
    skipped: list[int] = []
    all_tags: list[int] = []
    for _ in range(n_blocks):
        block_header = _parse_int_line(next_line(), "$Elements")
        if len(block_header) != 4:
            raise MeshFormatError("malformed $Elements block header")
        bdim, btag, type_code, count = block_header
        if count < 0:
            raise MeshFormatError("$Elements block counts must be non-negative")
        elem_tags: list[int] = []
        conn: list[list[int]] = []
        for _ in range(count):
            values = _parse_int_line(next_line(), "$Elements")
            elem_tags.append(values[0])
            conn.append(values[1:])
        all_tags.extend(elem_tags)
        try:
            etype = ElementType(type_code)
        except ValueError:
            skipped.append(type_code)
            continue
        expected = _NODE_COUNT[etype]
        for row in conn:
            if len(row) != expected:
                raise MeshFormatError(
                    f"element type {type_code} expects {expected} nodes per element, "
                    f"found {len(row)}"
                )
        raw.append((bdim, btag, etype, elem_tags, conn))

    if idx != len(lines):
        raise MeshFormatError("$Elements contains extra data")
    _validate_tag_summary("$Elements", all_tags, n_elements, min_element_tag, max_element_tag)

    merged: dict[tuple[int, int, ElementType], ElementBlock] = {}
    for bdim, btag, etype, tags, conn in raw:
        key = (bdim, btag, etype)
        tags_arr = np.array(tags, dtype=np.int64)
        conn_arr = np.array(conn, dtype=np.int64).reshape(len(conn), _NODE_COUNT[etype])
        if key in merged:
            prev = merged[key]
            merged[key] = ElementBlock(
                bdim,
                btag,
                etype,
                np.concatenate([prev.tags, tags_arr]),
                np.concatenate([prev.connectivity, conn_arr]),
            )
        else:
            merged[key] = ElementBlock(bdim, btag, etype, tags_arr, conn_arr)
    return list(merged.values()), skipped


def _synthesize_entities(blocks: list[ElementBlock], nodes: NodeTable) -> list[EntityRecord]:
    keys = {(b.entity_dim, b.entity_tag) for b in blocks}
    records: list[EntityRecord] = []
    for dim, tag in sorted(keys):
        mask = np.zeros(len(nodes.tags), dtype=bool)
        for b in blocks:
            if b.entity_dim == dim and b.entity_tag == tag:
                used = b.connectivity.ravel()
                # mark node indices whose tags are used
                mask |= np.isin(nodes.tags, used)
        if mask.any():
            sub = nodes.coords[mask]
            bbox = (
                float(sub[:, 0].min()),
                float(sub[:, 1].min()),
                float(sub[:, 2].min()),
                float(sub[:, 0].max()),
                float(sub[:, 1].max()),
                float(sub[:, 2].max()),
            )
        else:
            bbox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        records.append(EntityRecord(dim, tag, bbox, ()))
    return records


# -- Mesh ------------------------------------------------------------------


class Mesh:
    """An immutable, entity-classified mesh (MSH 4.1 model).

    A `Mesh` holds a global node table, per-entity element blocks, entity
    records, physical groups, and metadata. Construct one with `build` or read
    one from disk with `Mesh.read_msh`.
    """

    __slots__ = ("_elements", "_entities", "_groups", "_metadata", "_nodes")

    def __init__(
        self,
        nodes: NodeTable,
        elements: tuple[ElementBlock, ...],
        entities: tuple[EntityRecord, ...],
        groups: tuple[PhysicalGroup, ...],
        metadata: MeshMetadata,
    ) -> None:
        _freeze(nodes.tags)
        _freeze(nodes.coords)
        entity_dim = nodes.entity_dim
        entity_tag = nodes.entity_tag
        if entity_dim is not None:
            _freeze(entity_dim)
        if entity_tag is not None:
            _freeze(entity_tag)
        for block in elements:
            _freeze(block.tags)
            _freeze(block.connectivity)
        self._nodes = nodes
        self._elements = elements
        self._entities = entities
        self._groups = groups
        self._metadata = metadata

    def __repr__(self) -> str:
        x0, y0, z0, x1, y1, z1 = self.bbox
        return (
            f"Mesh(dim={self.metadata.dim}, nodes={self.num_nodes}, "
            f"elements={self.num_elements}, groups={len(self.groups)}, "
            f"bbox=({x0:g}, {y0:g}, {z0:g}, {x1:g}, {y1:g}, {z1:g}))"
        )

    # -- properties --------------------------------------------------------

    @property
    def nodes(self) -> NodeTable:
        """The global node table."""
        return self._nodes

    @property
    def element_blocks(self) -> tuple[ElementBlock, ...]:
        """The element blocks, grouped by entity and element type."""
        return self._elements

    @property
    def groups(self) -> tuple[PhysicalGroup, ...]:
        """The physical groups (one per label)."""
        return self._groups

    @property
    def metadata(self) -> MeshMetadata:
        """Descriptive metadata."""
        return self._metadata

    @property
    def num_nodes(self) -> int:
        """The number of nodes."""
        return int(self._nodes.tags.shape[0])

    @property
    def num_elements(self) -> int:
        """The total number of elements."""
        return sum(int(b.tags.shape[0]) for b in self._elements)

    @property
    def bbox(self) -> tuple[float, float, float, float, float, float]:
        """The mesh bounding box ``(xmin, ymin, zmin, xmax, ymax, zmax)``."""
        c = self._nodes.coords
        if c.shape[0] == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            float(c[:, 0].min()),
            float(c[:, 1].min()),
            float(c[:, 2].min()),
            float(c[:, 0].max()),
            float(c[:, 1].max()),
            float(c[:, 2].max()),
        )

    # -- queries -----------------------------------------------------------

    def entities(self, dim: int | None = None) -> tuple[EntityRecord, ...]:
        """Return entity records, optionally filtered by dimension.

        Args:
            dim: The dimension to filter on, or ``None`` for all entities.

        Returns:
            The matching entity records.
        """
        if dim is None:
            return self._entities
        return tuple(e for e in self._entities if e.dim == dim)

    def physical_group(self, name: str, dim: int) -> PhysicalGroup:
        """Return the physical group with the given name and dimension.

        Args:
            name: The group name.
            dim: The group dimension.

        Returns:
            The matching `PhysicalGroup`.

        Raises:
            KeyError: If no such group exists.
        """
        for group in self._groups:
            if group.name == name and group.dim == dim:
                return group
        raise KeyError(f"no physical group named {name!r} of dimension {dim}")

    def physical_groups_named(self, name: str) -> tuple[PhysicalGroup, ...]:
        """Return all physical groups with the given name.

        Args:
            name: The group name.

        Returns:
            The matching groups (possibly across multiple dimensions).
        """
        return tuple(g for g in self._groups if g.name == name)

    def elements_in_group(self, name: str, dim: int) -> tuple[ElementBlock, ...]:
        """Return the element blocks belonging to a physical group.

        Args:
            name: The group name.
            dim: The group dimension.

        Returns:
            The element blocks on the group's entities.

        Raises:
            KeyError: If no such group exists.
        """
        group = self.physical_group(name, dim)
        members = set(group.entities)
        return tuple(b for b in self._elements if (b.entity_dim, b.entity_tag) in members)

    def elements_on_entity(self, dim: int, tag: int) -> tuple[ElementBlock, ...]:
        """Return the element blocks on a specific entity.

        Args:
            dim: The entity dimension.
            tag: The entity tag.

        Returns:
            The element blocks on that entity.
        """
        return tuple(b for b in self._elements if b.entity_dim == dim and b.entity_tag == tag)

    # -- numpy view --------------------------------------------------------

    def to_numpy(self) -> NumpyMesh:
        """Return a numpy-view representation of the mesh.

        Returns:
            A `NumpyMesh` with 0-based index connectivity, suitable for
            solvers.

        Raises:
            ValueError: If an element references a missing node tag.
        """
        tags = self._nodes.tags
        index = {int(t): i for i, t in enumerate(tags)}
        blocks: dict[ElementType, NumpyElementBlock] = {}
        for block in self._elements:
            conn = block.connectivity
            idx = np.array(
                [[index.get(int(t), -1) for t in row] for row in conn], dtype=np.int64
            ).reshape(len(conn), _NODE_COUNT[block.type])
            if (idx < 0).any():
                raise ValueError("element references a missing node tag")
            if block.type in blocks:
                prev = blocks[block.type]
                blocks[block.type] = NumpyElementBlock(
                    _freeze(np.concatenate([prev.connectivity, idx])),
                    _freeze(np.concatenate([prev.node_tags, conn])),
                )
            else:
                blocks[block.type] = NumpyElementBlock(_freeze(idx), _freeze(conn))
        return NumpyMesh(self._nodes.coords, blocks)

    # -- I/O ---------------------------------------------------------------

    @classmethod
    def read_msh(cls, path: str | Path) -> Mesh:
        """Read an MSH 4.1 ASCII file.

        Args:
            path: The path to the ``.msh`` file.

        Returns:
            The parsed `Mesh`.

        Raises:
            MeshFormatError: If the file is malformed or uses an unsupported
                format (e.g. binary, or a version other than 4.1).
        """
        path = Path(path)
        data = path.read_bytes()
        if _has_binary_mesh_format(data):
            raise MeshFormatError("binary MSH is not supported; only ASCII")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MeshFormatError("MSH input is not valid UTF-8") from exc
        sections = _split_sections(text.splitlines())
        for forbidden in ("PartitionedEntities", "Periodic", "GhostElements", "ParametricNodes"):
            if forbidden in sections:
                raise MeshFormatError(f"{forbidden} sections are not supported")
        if "MeshFormat" not in sections:
            raise MeshFormatError("missing $MeshFormat")
        _parse_mesh_format(sections["MeshFormat"])

        names = _parse_physical_names(sections.get("PhysicalNames", []))
        nodes = _parse_nodes(sections.get("Nodes", []))
        blocks, skipped = _parse_elements(sections.get("Elements", []))
        if skipped:
            warnings.warn(
                f"skipped unsupported element type codes: {sorted(set(skipped))}",
                UserWarning,
                stacklevel=2,
            )

        if "Entities" in sections:
            records, phys_tags = _parse_entities(sections["Entities"])
        elif names:
            raise MeshFormatError("$PhysicalNames present but $Entities is missing")
        else:
            records = _synthesize_entities(blocks, nodes)
            phys_tags = [[] for _ in records]

        groups = _build_groups(names, records, phys_tags)
        dim = max((e.dim for e in records), default=0)
        metadata = MeshMetadata(dim=dim, msh_version="4.1", gmsh_version=None, source=str(path))
        return cls(
            nodes,
            tuple(blocks),
            tuple(records),
            tuple(groups),
            metadata,
        )

    def write_msh(self, path: str | Path) -> None:
        """Write the mesh as an MSH 4.1 ASCII file.

        Args:
            path: The destination path.

        Raises:
            ValueError: If the mesh data is inconsistent or uses an unsupported
                element type.
        """
        self._validate_for_write()
        path = Path(path)
        lines: list[str] = []

        lines.append("$MeshFormat")
        lines.append("4.1 0 8")
        lines.append("$EndMeshFormat")
        lines.append("")

        lines.append("$PhysicalNames")
        lines.append(str(len(self._groups)))
        for g in self._groups:
            lines.append(f'{g.dim} {g.tag} "{g.name}"')
        lines.append("$EndPhysicalNames")
        lines.append("")

        lines.append("$Entities")
        counts = [sum(1 for e in self._entities if e.dim == d) for d in (0, 1, 2, 3)]
        lines.append(" ".join(str(c) for c in counts))
        for e in sorted(self._entities, key=lambda e: (e.dim, e.tag)):
            lines.append(_entity_line(e, self._groups))
        lines.append("$EndEntities")
        lines.append("")

        lines.append("$Nodes")
        node_blocks = _build_node_blocks(self._nodes, self._elements)
        lines.append(
            f"{len(node_blocks)} {self.num_nodes} "
            f"{int(self._nodes.tags.min()) if self.num_nodes else 0} "
            f"{int(self._nodes.tags.max()) if self.num_nodes else 0}"
        )
        for dim, tag, tags, coords in node_blocks:
            lines.append(f"{dim} {tag} 0 {len(tags)}")
            for t in tags:
                lines.append(str(int(t)))
            for row in coords:
                lines.append(" ".join(_fmt_float(x) for x in row))
        lines.append("$EndNodes")
        lines.append("")

        lines.append("$Elements")
        lines.append(
            f"{len(self._elements)} {self.num_elements} "
            f"{self._min_element_tag()} {self._max_element_tag()}"
        )
        for block in self._elements:
            lines.append(
                f"{block.entity_dim} {block.entity_tag} {int(block.type)} {block.tags.shape[0]}"
            )
            for etag, row in zip(block.tags, block.connectivity, strict=True):
                lines.append(" ".join([str(int(etag)), *(str(int(t)) for t in row)]))
        lines.append("$EndElements")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _validate_for_write(self) -> None:
        entity_keys = {(entity.dim, entity.tag) for entity in self._entities}
        for group in self._groups:
            for dim, tag in group.entities:
                if dim != group.dim:
                    raise ValueError(
                        f"physical group {group.name!r} includes entity ({dim}, {tag}) "
                        f"with dimension {dim}, expected {group.dim}"
                    )
                if (dim, tag) not in entity_keys:
                    raise ValueError(
                        f"physical group {group.name!r} references missing entity ({dim}, {tag})"
                    )

        for block in self._elements:
            try:
                element_type = ElementType(block.type)
            except ValueError as exc:
                raise ValueError(f"unsupported element type {block.type!r}") from exc
            if block.tags.ndim != 1:
                raise ValueError("element tags must be a one-dimensional array")
            expected_shape = (block.tags.shape[0], _NODE_COUNT[element_type])
            if block.connectivity.shape != expected_shape:
                raise ValueError(
                    f"element connectivity must have shape {expected_shape}, "
                    f"got {block.connectivity.shape}"
                )

    def _min_element_tag(self) -> int:
        minima = [int(b.tags.min()) for b in self._elements if b.tags.shape[0]]
        return min(minima) if minima else 0

    def _max_element_tag(self) -> int:
        maxima = [int(b.tags.max()) for b in self._elements if b.tags.shape[0]]
        return max(maxima) if maxima else 0


# -- writing helpers -------------------------------------------------------


def _fmt_float(x: float) -> str:
    return repr(float(x))


def _build_groups(
    names: list[tuple[str, int, int]],
    records: list[EntityRecord],
    phys_tags: list[list[int]],
) -> list[PhysicalGroup]:
    groups: list[PhysicalGroup] = []
    for name, dim, tag in names:
        entities = tuple(
            (r.dim, r.tag)
            for r, tags in zip(records, phys_tags, strict=True)
            if tag in tags and r.dim == dim
        )
        groups.append(PhysicalGroup(name, dim, tag, entities))
    return groups


def _entity_physical_tags(dim: int, tag: int, groups: tuple[PhysicalGroup, ...]) -> list[int]:
    return sorted(g.tag for g in groups if (dim, tag) in g.entities)


def _entity_line(entity: EntityRecord, groups: tuple[PhysicalGroup, ...]) -> str:
    phys = _entity_physical_tags(entity.dim, entity.tag, groups)
    parts: list[str] = [str(entity.tag)]
    if entity.dim == 0:
        parts.extend(_fmt_float(x) for x in entity.bbox[:3])
    else:
        parts.extend(_fmt_float(x) for x in entity.bbox)
    parts.append(str(len(phys)))
    parts.extend(str(p) for p in phys)
    if entity.dim > 0:
        parts.append(str(len(entity.boundary)))
        parts.extend(str(signed) for _, signed in entity.boundary)
    return " ".join(parts)


def _build_node_blocks(
    nodes: NodeTable, elements: tuple[ElementBlock, ...]
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    if nodes.entity_dim is not None and nodes.entity_tag is not None:
        classification = list(
            zip(nodes.entity_dim.tolist(), nodes.entity_tag.tolist(), strict=True)
        )
    else:
        classification = _classify_from_elements(nodes, elements)

    order: dict[tuple[int, int], list[int]] = {}
    for i, (dim, tag) in enumerate(classification):
        order.setdefault((dim, tag), []).append(i)

    blocks: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for (dim, tag), idxs in sorted(order.items()):
        idx_arr = np.array(idxs, dtype=np.int64)
        blocks.append((dim, tag, nodes.tags[idx_arr], nodes.coords[idx_arr]))
    return blocks


def _classify_from_elements(
    nodes: NodeTable, elements: tuple[ElementBlock, ...]
) -> list[tuple[int, int]]:
    node_entity: dict[int, tuple[int, int]] = {}
    for block in elements:
        for row in block.connectivity:
            for ntag in row.tolist():
                ntag = int(ntag)
                current = node_entity.get(ntag)
                if current is None or block.entity_dim < current[0]:
                    node_entity[ntag] = (block.entity_dim, block.entity_tag)
    classification: list[tuple[int, int]] = []
    for tag in nodes.tags.tolist():
        tag = int(tag)
        if tag not in node_entity:
            raise ValueError(f"node {tag} cannot be classified; no element uses it")
        classification.append(node_entity[tag])
    return classification
