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

import pytest

from shapemesh import (
    Box,
    GeometryError,
    Intersection,
    Mesh,
    MeshSettings,
    Union,
    build,
)


def test_build_single_box():
    mesh = build(Box((0, 0, 0), 1, 1, 1, label="box"), MeshSettings())
    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert mesh.metadata.dim == 3
    assert len(mesh.entities(dim=3)) == 1
    assert mesh.physical_group("box", 3).entities == ((3, 1),)


def test_build_two_overlapping_boxes_fragment():
    a = Box((0, 0, 0), 1, 1, 1, label="a")
    b = Box((0.5, 0, 0), 1, 1, 1, label="b")
    mesh = build([a, b], MeshSettings())

    assert len(mesh.entities(dim=3)) == 3
    group_a = set(mesh.physical_group("a", 3).entities)
    group_b = set(mesh.physical_group("b", 3).entities)
    assert len(group_a) == 2
    assert len(group_b) == 2
    # the overlap piece belongs to both groups
    assert len(group_a & group_b) == 1


def test_build_single_shape_is_normalized():
    mesh = build(Box((0, 0, 0), 1, 1, 1), MeshSettings())
    assert mesh.num_elements > 0


def test_element_size_controls_mesh_size():
    coarse = build(Box((0, 0, 0), 1, 1, 1), MeshSettings(element_size=1.0))
    fine = build(Box((0, 0, 0), 1, 1, 1), MeshSettings(element_size=0.25))
    assert fine.num_elements > coarse.num_elements


def test_boolean_union():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0, 0), 1, 1, 1)
    mesh = build(Union(a, b, label="u"), MeshSettings())
    assert len(mesh.entities(dim=3)) == 1
    assert mesh.physical_group("u", 3).entities == ((3, 1),)


def test_empty_intersection_raises():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((5, 5, 5), 1, 1, 1)  # disjoint
    with pytest.raises(GeometryError):
        build(Intersection(a, b), MeshSettings())


def test_shape_reuse_allowed():
    a = Box((0, 0, 0), 1, 1, 1)
    mesh = build([a, a], MeshSettings())  # reusing the same object is allowed
    assert mesh.num_elements > 0


def test_roundtrip_through_msh(tmp_path):
    mesh = build(Box((0, 0, 0), 1, 1, 1, label="box"), MeshSettings())
    path = tmp_path / "box.msh"
    mesh.write_msh(path)
    reread = Mesh.read_msh(path)
    assert reread.num_nodes == mesh.num_nodes
    assert reread.num_elements == mesh.num_elements
    assert reread.physical_group("box", 3).entities == mesh.physical_group("box", 3).entities


def test_same_label_and_dimension_merge_into_one_group():
    a = Box((0, 0, 0), 1, 1, 1, label="wall")
    b = Box((2, 0, 0), 1, 1, 1, label="wall")
    mesh = build([a, b], MeshSettings())
    group = mesh.physical_group("wall", 3)
    assert len(group.entities) == 2
