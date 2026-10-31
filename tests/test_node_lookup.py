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

import shapemesh.mesh as mesh_module
from shapemesh import Mesh, MeshFormatError, NodeTable

DUPLICATE_NODES_MSH = """\
$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 3 1 2
0 1 0 3
1
2
2
0 0 0
1 0 0
2 0 0
$EndNodes
$Elements
0 0 0 0
$EndElements
"""


def _nodes(tags) -> NodeTable:
    tag_array = np.asarray(tags, dtype=np.int64)
    coords = np.arange(tag_array.size * 3, dtype=np.float64).reshape(-1, 3)
    return NodeTable(tag_array, coords)


def test_scalar_lookup_and_coordinate_copy():
    nodes = _nodes([11, 12, 13])

    rows = nodes.indices_for(12)
    assert isinstance(rows, int)
    assert rows == 1
    assert nodes.indices_for(np.array(12)) == 1

    coords = nodes.coords_for(12)
    assert coords.shape == (3,)
    assert coords.flags.writeable
    np.testing.assert_array_equal(coords, nodes.coords[1])
    coords[0] = 999.0
    assert nodes.coords[1, 0] != 999.0


def test_array_lookup_preserves_shape_order_and_duplicates():
    nodes = _nodes([11, 12, 13])

    rows = nodes.indices_for([[13, 11], [12, 13]])
    assert rows.shape == (2, 2)
    assert rows.dtype == np.dtype(np.intp)
    assert rows.flags.writeable
    np.testing.assert_array_equal(rows, [[2, 0], [1, 2]])

    coords = nodes.coords_for([[13, 11], [12, 13]])
    assert coords.shape == (2, 2, 3)
    np.testing.assert_array_equal(coords, nodes.coords[[[2, 0], [1, 2]]])
    coords[0, 0, 0] = 999.0
    assert nodes.coords[2, 0] != 999.0


def test_empty_queries_preserve_shape():
    nodes = _nodes([1, 2, 3])

    rows = nodes.indices_for([])
    assert rows.shape == (0,)
    assert rows.dtype == np.dtype(np.intp)
    assert rows.flags.writeable
    assert nodes.coords_for([]).shape == (0, 3)

    empty = np.empty((2, 0), dtype=np.float64)
    assert nodes.indices_for(empty).shape == (2, 0)
    assert nodes.coords_for(empty).shape == (2, 0, 3)


def test_empty_mesh():
    nodes = _nodes([])

    assert nodes.indices_for([]).shape == (0,)
    assert nodes.coords_for([]).shape == (0, 3)
    with pytest.raises(KeyError, match="node tags not found"):
        nodes.indices_for(1)


@pytest.mark.parametrize(
    ("tags", "strategy"),
    [
        ([5, 6, 7, 8], "contiguous"),
        ([3, 1, 2], "direct"),
        ([10, 12, 14], "direct"),
        ([5000, 10, 1000], "sorted"),
        ([10, 1000, 5000], "sorted"),
    ],
)
def test_lookup_strategies_and_cache_reuse(tags, strategy):
    nodes = _nodes(tags)

    np.testing.assert_array_equal(nodes.indices_for(tags[::-1]), np.arange(len(tags))[::-1])
    assert nodes._lookup is not None
    assert nodes._lookup.strategy == strategy
    lookup = nodes._lookup

    nodes.indices_for(tags[0])
    nodes.coords_for(tags[-1])
    assert nodes._lookup is lookup


def test_contiguous_lookup_uses_no_cached_arrays():
    nodes = _nodes([7, 8, 9, 10])
    nodes.indices_for(8)

    lookup = nodes._lookup
    assert lookup is not None
    assert lookup.strategy == "contiguous"
    assert lookup.direct is None
    assert lookup.sorted_tags is None
    assert lookup.order is None


def test_dense_lookup_relative_budget():
    dense = _nodes([10, 15, 21])  # range 12 == 4 * N
    sparse = _nodes([10, 15, 22])  # range 13 > 4 * N

    dense.indices_for(10)
    sparse.indices_for(10)
    assert dense._lookup is not None and dense._lookup.strategy == "direct"
    assert sparse._lookup is not None and sparse._lookup.strategy == "sorted"
    assert dense._lookup.direct is not None
    assert dense._lookup.direct.nbytes == 12 * np.dtype(np.int32).itemsize


def test_dense_lookup_absolute_budget(monkeypatch):
    monkeypatch.setattr(mesh_module, "_DENSE_LOOKUP_MAX_BYTES", 4)
    nodes = _nodes([20, 10, 30])

    nodes.indices_for(10)
    assert nodes._lookup is not None
    assert nodes._lookup.strategy == "sorted"


def test_ascending_sorted_lookup_reuses_tag_array():
    nodes = _nodes([10, 1000, 5000])
    nodes.indices_for(1000)

    lookup = nodes._lookup
    assert lookup is not None
    assert lookup.strategy == "sorted"
    assert lookup.sorted_tags is nodes.tags
    assert lookup.order is None


@pytest.mark.parametrize("tags", [[2, 1, 2], [1, 1000, 1000]])
def test_manual_duplicate_tags_fail_at_first_lookup(tags):
    nodes = _nodes(tags)

    with pytest.raises(ValueError, match="node tags must be unique"):
        nodes.indices_for(1)


def test_read_rejects_duplicate_node_tags(tmp_path):
    path = tmp_path / "duplicates.msh"
    path.write_text(DUPLICATE_NODES_MSH, encoding="utf-8")

    with pytest.raises(MeshFormatError, match=r"duplicate node tags: \[2\]"):
        Mesh.read_msh(path)


def test_missing_tags_list_first_five():
    nodes = _nodes([10, 20, 30])

    with pytest.raises(KeyError, match=r"node tags not found: \[5, 7, 9, 11, 13\]"):
        nodes.indices_for([5, 7, 9, 11, 13, 15])
    with pytest.raises(KeyError, match="node tags not found"):
        nodes.coords_for(40)


def test_negative_query_tag_is_missing_even_if_present():
    nodes = _nodes([-1, 0, 1])

    assert nodes.indices_for(0) == 1
    with pytest.raises(KeyError, match=r"node tags not found: \[-1\]"):
        nodes.indices_for(-1)


def test_direct_lookup_rejects_gaps_and_out_of_range_tags():
    nodes = _nodes([10, 12, 14])

    with pytest.raises(KeyError, match=r"node tags not found: \[9, 11, 15\]"):
        nodes.indices_for([9, 11, 15])


@pytest.mark.parametrize("tag", [-(2**100), 2**100, np.uint64(2**64 - 1)])
def test_out_of_range_tags_are_missing(tag):
    nodes = _nodes([1, 2, 3])

    with pytest.raises(KeyError, match="node tags not found"):
        nodes.indices_for(tag)


@pytest.mark.parametrize(
    "tags",
    [1.0, np.array([1.0]), True, np.array([False]), "1", [1, "2"], None],
)
def test_non_integer_queries_raise_type_error(tags):
    nodes = _nodes([1, 2, 3])

    with pytest.raises(TypeError, match="node tags"):
        nodes.indices_for(tags)


def test_read_only_query_is_not_mutated():
    nodes = _nodes([3, 1, 2])
    query = np.array([2, 3, 1])
    query.setflags(write=False)
    original = query.copy()

    rows = nodes.indices_for(query)
    np.testing.assert_array_equal(query, original)
    np.testing.assert_array_equal(rows, [2, 0, 1])
    assert rows.flags.writeable


def test_node_arrays_freeze_at_construction():
    arrays = [
        np.array([1, 2]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([0, 0]),
        np.array([1, 1]),
    ]
    nodes = NodeTable(*arrays)

    for array in (nodes.tags, nodes.coords, nodes.entity_dim, nodes.entity_tag):
        assert array is not None
        assert not array.flags.writeable
