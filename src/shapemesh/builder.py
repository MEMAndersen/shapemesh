"""Gmsh build pipeline: shapes + settings -> a validated :class:`Mesh`.

The pipeline validates shapes and settings, realizes them into a Gmsh model,
fragments overlapping entities to preserve labels, meshes, and reads the
entity-classified mesh back. Gmsh is imported lazily and a process-wide lock
serializes builds.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from .constants import GMSH_TERMINAL_SILENT, GMSH_TERMINAL_VERBOSE
from .errors import BuildError, GeometryError, ShapeMeshError
from .mesh import (
    _NODE_COUNT,
    ElementBlock,
    ElementType,
    EntityRecord,
    Mesh,
    MeshMetadata,
    NodeTable,
    PhysicalGroup,
)
from .settings import MeshSettings
from .shapes import Shape

__all__ = ["build"]

_LOCK = threading.RLock()

_RESERVED_OPTIONS = frozenset({"Mesh.MeshSizeMax", "Mesh.Algorithm", "Mesh.Algorithm3D"})


def build(shapes: Shape | Sequence[Shape], settings: MeshSettings) -> Mesh:
    """Realize ``shapes`` with Gmsh and return the resulting :class:`Mesh`.

    ``settings`` is required. A single :class:`Shape` is normalized to a
    one-element sequence.
    """
    shape_list = _normalize_shapes(shapes)
    _validate_build(shape_list)
    _validate_settings(settings)

    with _LOCK:
        import gmsh

        owned = not gmsh.isInitialized()
        if owned:
            gmsh.initialize()
        added = False
        try:
            gmsh.model.add("shapemesh")
            added = True
            _apply_settings(gmsh, settings)
            return _build_mesh(gmsh, shape_list, settings)
        except ShapeMeshError:
            raise
        except Exception as exc:
            raise BuildError(str(exc), _capture_log(gmsh)) from exc
        finally:
            if added:
                gmsh.model.remove()
            if owned:
                gmsh.finalize()


def _normalize_shapes(shapes: Shape | Sequence[Shape]) -> list[Shape]:
    if isinstance(shapes, Shape):
        return [shapes]
    if isinstance(shapes, (str, bytes)):
        raise GeometryError("shapes must be a Shape or a sequence of Shape")
    return list(shapes)


def _validate_build(shapes: Sequence[Shape]) -> None:
    if not shapes:
        raise GeometryError("build requires at least one shape")
    for shape in shapes:
        if not isinstance(shape, Shape):
            raise GeometryError(f"build shapes must be Shape instances, got {type(shape).__name__}")


def _validate_settings(settings: MeshSettings) -> None:
    for name, _ in settings.advanced:
        if name in _RESERVED_OPTIONS:
            raise ValueError(f"advanced option {name!r} conflicts with an explicit setting")


def _apply_settings(gmsh: Any, settings: MeshSettings) -> None:
    gmsh.option.setNumber(
        "General.Terminal",
        GMSH_TERMINAL_VERBOSE if settings.verbose else GMSH_TERMINAL_SILENT,
    )
    if settings.characteristic_length is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMax", settings.characteristic_length)
    if settings.algorithm_2d is not None:
        gmsh.option.setNumber("Mesh.Algorithm", int(settings.algorithm_2d))
    if settings.algorithm_3d is not None:
        gmsh.option.setNumber("Mesh.Algorithm3D", int(settings.algorithm_3d))
    for name, value in settings.advanced:
        if isinstance(value, str):
            gmsh.option.setString(name, value)
        else:
            gmsh.option.setNumber(name, value)


def _capture_log(gmsh: Any) -> str:
    try:
        return "\n".join(gmsh.logger.get())
    except Exception:
        return ""


# -- pipeline --------------------------------------------------------------


def _build_mesh(gmsh: Any, shapes: Sequence[Shape], settings: MeshSettings) -> Mesh:
    occ = gmsh.model.occ
    label_map: dict[tuple[str, int], set[tuple[int, int]]] = {}
    live: list[tuple[int, int]] = []
    for shape in shapes:
        dimtags = shape._realize(gmsh)
        live.extend(dimtags)
        if shape.label is not None:
            label_map.setdefault((shape.label, shape.dimension), set()).update(dimtags)

    occ.synchronize()
    if len(live) > 1:
        label_map = _fragment_remap(gmsh, live, label_map)
    occ.synchronize()

    groups = _add_groups(gmsh, label_map)
    dim = gmsh.model.getDimension()
    gmsh.model.mesh.setOrder(settings.order)
    gmsh.model.mesh.generate(max(dim, 1))
    if settings.optimize:
        gmsh.model.mesh.optimize("" if settings.order == 1 else "HighOrder")
    return _read_mesh(gmsh, groups)


def _fragment_remap(
    gmsh: Any,
    live: list[tuple[int, int]],
    label_map: dict[tuple[str, int], set[tuple[int, int]]],
) -> dict[tuple[str, int], set[tuple[int, int]]]:
    occ = gmsh.model.occ
    _, outmap = occ.fragment(live, [], removeObject=True, removeTool=True)
    occ.synchronize()
    index = {dimtag: i for i, dimtag in enumerate(live)}
    remapped: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for (label, dim), dimtags in label_map.items():
        outputs: set[tuple[int, int]] = set()
        for dimtag in dimtags:
            for out in outmap[index[dimtag]]:
                if out[0] == dim:
                    outputs.add(out)
        remapped[(label, dim)] = outputs
    return remapped


def _add_groups(
    gmsh: Any, label_map: dict[tuple[str, int], set[tuple[int, int]]]
) -> list[PhysicalGroup]:
    groups: list[PhysicalGroup] = []
    for label, dim in sorted(label_map):
        dimtags = sorted(label_map[(label, dim)])
        if not dimtags:
            continue
        tags = [tag for _, tag in dimtags]
        group_tag = gmsh.model.addPhysicalGroup(dim, tags, name=label)
        groups.append(PhysicalGroup(label, dim, group_tag, tuple(dimtags)))
    return groups


def _read_mesh(gmsh: Any, groups: list[PhysicalGroup]) -> Mesh:
    entities: list[EntityRecord] = []
    for dim in range(4):
        for _, tag in gmsh.model.getEntities(dim):
            bbox = tuple(gmsh.model.getBoundingBox(dim, tag))
            boundary = tuple(gmsh.model.getBoundary([(dim, tag)], oriented=True))
            entities.append(EntityRecord(dim, tag, bbox, boundary))

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags_arr = np.asarray(node_tags, dtype=np.int64)
    coords_arr = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_index = {int(t): i for i, t in enumerate(node_tags)}
    entity_dim = np.full(len(node_tags), -1, dtype=np.int64)
    entity_tag = np.full(len(node_tags), -1, dtype=np.int64)
    for dim in range(4):
        for _, tag in gmsh.model.getEntities(dim):
            ent_tags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
            for node_tag in ent_tags:
                idx = tag_to_index[int(node_tag)]
                entity_dim[idx] = dim
                entity_tag[idx] = tag

    blocks: list[ElementBlock] = []
    for dim in range(4):
        for _, tag in gmsh.model.getEntities(dim):
            types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
            for type_code, etags, ntags in zip(types, elem_tags, node_tags, strict=True):
                element_type = ElementType(int(type_code))
                k = _NODE_COUNT[element_type]
                conn = np.asarray(ntags, dtype=np.int64).reshape(-1, k)
                blocks.append(
                    ElementBlock(dim, tag, element_type, np.asarray(etags, dtype=np.int64), conn)
                )

    model_dim = gmsh.model.getDimension()
    version = gmsh.option.getString("General.Version")
    metadata = MeshMetadata(dim=model_dim, msh_version="4.1", gmsh_version=version, source=None)
    return Mesh(
        NodeTable(node_tags_arr, coords_arr, entity_dim, entity_tag),
        tuple(blocks),
        tuple(entities),
        tuple(groups),
        metadata,
    )
