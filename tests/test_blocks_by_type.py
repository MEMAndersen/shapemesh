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

import numpy as np
import pytest

import shapemesh
from shapemesh import (
    BlockSlice,
    ElementBlock,
    ElementType,
    EntityRecord,
    Mesh,
    MeshMetadata,
    MeshSettings,
    NodeTable,
    PhysicalGroup,
    Rectangle,
    TypeBlockView,
    build,
)

SPARSE_UNORDERED_MSH = """\
$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "tri"
$EndPhysicalNames
$Entities
3 3 1 0
1 0 0 0 0
2 1 0 0 0
3 0 1 0 0
1 0 0 0 1 0 0 0 2 1 2
2 1 0 0 1 1 0 0 2 2 3
3 0 0 0 0 1 0 0 2 3 1
1 0 0 0 1 1 0 1 1 3 1 2 3
$EndEntities
$Nodes
1 3 10 30
2 1 0 3
30
10
20
0 0 0
1 0 0
0 1 0
$EndNodes
$Elements
1 1 7 7
2 1 2 1
7 10 30 20
$EndElements
"""


def _mesh() -> Mesh:
    nodes = NodeTable(
        np.array([30, 10, 50, 20, 40, 60], dtype=np.int64),
        np.arange(18, dtype=np.float64).reshape(6, 3),
    )
    blocks = (
        ElementBlock(
            2,
            1,
            ElementType.TRI3,
            np.array([101, 102], dtype=np.int64),
            np.array([[10, 30, 50], [20, 10, 40]], dtype=np.int64),
        ),
        ElementBlock(
            1,
            7,
            ElementType.LINE2,
            np.array([201, 202], dtype=np.int64),
            np.array([[10, 20], [30, 10]], dtype=np.int64),
        ),
        ElementBlock(
            2,
            2,
            ElementType.TRI3,
            np.array([103], dtype=np.int64),
            np.array([[40, 50, 60]], dtype=np.int64),
        ),
        ElementBlock(
            0,
            9,
            ElementType.POINT1,
            np.array([], dtype=np.int64),
            np.empty((0, 1), dtype=np.int64),
        ),
    )
    entities = tuple(
        EntityRecord(dim, tag, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        for dim, tag in ((2, 1), (2, 2), (1, 7), (0, 9), (2, 3))
    )
    groups = (
        PhysicalGroup("a", 2, 1, ((2, 1),)),
        PhysicalGroup("b", 2, 2, ((2, 1), (2, 2))),
        PhysicalGroup("line", 1, 3, ((1, 7),)),
        PhysicalGroup("points", 0, 4, ((0, 9),)),
        PhysicalGroup("empty", 2, 5, ((2, 3),)),
    )
    return Mesh(nodes, blocks, entities, groups, MeshMetadata(dim=2))


def test_blocks_by_type_flattens_arrays_and_slices():
    mesh = _mesh()

    views = mesh.blocks_by_type()
    assert list(views) == [ElementType.TRI3, ElementType.LINE2, ElementType.POINT1]

    triangles = views[ElementType.TRI3]
    assert isinstance(triangles, TypeBlockView)
    assert triangles.connectivity.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        triangles.connectivity,
        [[1, 0, 2], [3, 1, 4], [4, 2, 5]],
    )
    np.testing.assert_array_equal(triangles.element_tags, [101, 102, 103])
    assert triangles.slices == (
        BlockSlice(2, 1, 0, 2),
        BlockSlice(2, 2, 2, 3),
    )
    assert sum(s.stop - s.start for s in triangles.slices) == len(triangles.element_tags)
    np.testing.assert_array_equal(
        mesh.nodes.tags[triangles.connectivity],
        [[10, 30, 50], [20, 10, 40], [40, 50, 60]],
    )
    np.testing.assert_array_equal(
        mesh.nodes.coords[triangles.connectivity],
        mesh.nodes.coords[[[1, 0, 2], [3, 1, 4], [4, 2, 5]]],
    )

    assert views[ElementType.LINE2].slices == (BlockSlice(1, 7, 0, 2),)
    points = views[ElementType.POINT1]
    assert points.connectivity.shape == (0, 1)
    assert points.element_tags.shape == (0,)
    assert points.slices == (BlockSlice(0, 9, 0, 0),)


def test_views_are_cached_and_read_only():
    mesh = _mesh()

    views = mesh.blocks_by_type()
    assert mesh.blocks_by_type() is views
    assert views[ElementType.TRI3] is mesh.blocks_by_type()[ElementType.TRI3]
    assert not views[ElementType.TRI3].connectivity.flags.writeable
    assert not views[ElementType.TRI3].element_tags.flags.writeable
    with pytest.raises(TypeError):
        views[ElementType.QUAD4] = views[ElementType.TRI3]


def test_single_block_shares_element_tag_storage():
    mesh = _mesh()
    line_block = next(block for block in mesh.element_blocks if block.type is ElementType.LINE2)

    view = mesh.blocks_by_type()[ElementType.LINE2]
    assert np.shares_memory(view.element_tags, line_block.tags)


def test_element_indices_in_group():
    mesh = _mesh()

    np.testing.assert_array_equal(mesh.element_indices_in_group("a", 2)[ElementType.TRI3], [0, 1])
    np.testing.assert_array_equal(
        mesh.element_indices_in_group("b", 2)[ElementType.TRI3], [0, 1, 2]
    )
    np.testing.assert_array_equal(
        mesh.element_indices_in_group("line", 1)[ElementType.LINE2], [0, 1]
    )
    assert dict(mesh.element_indices_in_group("empty", 2)) == {}

    points = mesh.element_indices_in_group("points", 0)
    assert list(points) == [ElementType.POINT1]
    assert points[ElementType.POINT1].shape == (0,)
    assert not points[ElementType.POINT1].flags.writeable
    with pytest.raises(TypeError):
        points[ElementType.TRI3] = np.array([], dtype=np.intp)


def test_node_indices_in_group_are_unique_sorted_and_read_only():
    mesh = _mesh()

    a_nodes = mesh.node_indices_in_group("a", 2)
    np.testing.assert_array_equal(a_nodes, [0, 1, 2, 3, 4])
    assert a_nodes.dtype == np.dtype(np.intp)
    assert not a_nodes.flags.writeable
    np.testing.assert_array_equal(mesh.node_indices_in_group("b", 2), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(mesh.node_indices_in_group("line", 1), [0, 1, 3])
    assert mesh.node_indices_in_group("empty", 2).shape == (0,)
    assert mesh.node_indices_in_group("points", 0).shape == (0,)


def test_group_helpers_reject_unknown_group():
    mesh = _mesh()

    with pytest.raises(KeyError, match="missing"):
        mesh.element_indices_in_group("missing", 2)
    with pytest.raises(KeyError, match="missing"):
        mesh.node_indices_in_group("missing", 2)


def test_blocks_by_type_rejects_dangling_connectivity():
    nodes = NodeTable(
        np.array([1, 2], dtype=np.int64),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    block = ElementBlock(
        1,
        1,
        ElementType.LINE2,
        np.array([1]),
        np.array([[1, 3]]),
    )
    mesh = Mesh(nodes, (block,), (), (), MeshMetadata(dim=1))

    with pytest.raises(ValueError, match="element references a missing node tag"):
        mesh.blocks_by_type()


def test_read_msh_with_sparse_unordered_tags(tmp_path):
    path = tmp_path / "sparse.msh"
    path.write_text(SPARSE_UNORDERED_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    view = mesh.blocks_by_type()[ElementType.TRI3]
    np.testing.assert_array_equal(view.connectivity, [[1, 0, 2]])
    np.testing.assert_array_equal(mesh.nodes.tags[view.connectivity], [[10, 30, 20]])
    np.testing.assert_array_equal(mesh.element_indices_in_group("tri", 2)[ElementType.TRI3], [0])
    np.testing.assert_array_equal(mesh.node_indices_in_group("tri", 2), [0, 1, 2])


def test_build_group_selection_matches_entity_blocks():
    mesh = build(
        Rectangle(origin=(0, 0), dx=1, dy=1, label="plate"),
        MeshSettings(element_size=0.5),
    )

    group_blocks = mesh.elements_in_group("plate", dim=2)
    group_indices = mesh.element_indices_in_group("plate", dim=2)
    expected_count = sum(len(block.tags) for block in group_blocks)
    assert sum(len(indices) for indices in group_indices.values()) == expected_count

    for element_type, indices in group_indices.items():
        view = mesh.blocks_by_type()[element_type]
        assert mesh.nodes.coords[view.connectivity[indices]].shape == (
            len(indices),
            view.connectivity.shape[1],
            3,
        )

    node_indices = mesh.node_indices_in_group("plate", dim=2)
    assert node_indices.ndim == 1
    assert len(np.unique(node_indices)) == len(node_indices)
    np.testing.assert_array_equal(node_indices, np.sort(node_indices))


def test_old_numpy_api_is_removed():
    assert not hasattr(Mesh, "to_numpy")
    assert not hasattr(shapemesh, "NumpyMesh")
    assert not hasattr(shapemesh, "NumpyElementBlock")
    assert "TypeBlockView" in shapemesh.__all__
    assert "BlockSlice" in shapemesh.__all__
