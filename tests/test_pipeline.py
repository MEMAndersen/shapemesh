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

from shapemesh import (
    Algorithm2D,
    Algorithm3D,
    Box,
    BuildError,
    Cylinder,
    Difference,
    Disk,
    ElementType,
    GeometryError,
    Intersection,
    Mesh,
    MeshSettings,
    Point,
    Polygon,
    Polyline,
    Sphere,
    Union,
    build,
)


def _mesh_volume(mesh: Mesh) -> float:
    """Compute the volume of a 3D mesh by summing TET4 element volumes."""
    tags = mesh.nodes.tags
    index = {int(t): i for i, t in enumerate(tags)}
    coords = mesh.nodes.coords
    volume = 0.0
    has_tets = False
    for block in mesh.element_blocks:
        if block.entity_dim != 3 or block.type != ElementType.TET4:
            continue
        has_tets = True
        for tet in block.connectivity:
            i0, i1, i2, i3 = (index[int(t)] for t in tet)
            v0 = coords[i0]
            v1 = coords[i1]
            v2 = coords[i2]
            v3 = coords[i3]
            volume += abs(float(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0)))) / 6.0
    if not has_tets:
        raise ValueError("mesh has no TET4 volume elements")
    return volume


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


def test_order_controls_generated_element_types():
    mesh = build(
        Box((0, 0, 0), 1, 1, 1),
        MeshSettings(element_size=1.0, order=3, optimize=False),
    )

    assert {block.type for block in mesh.element_blocks} == {
        ElementType.POINT1,
        ElementType.LINE4,
        ElementType.TRI10,
        ElementType.TET20,
    }


def test_preinitialized_gmsh_options_are_restored():
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        numeric_options = {
            name: gmsh.option.getNumber(name)
            for name in (
                "General.Terminal",
                "Mesh.Algorithm",
                "Mesh.Algorithm3D",
                "Mesh.ElementOrder",
                "Mesh.MinimumCirclePoints",
            )
        }
        socket_name = gmsh.option.getString("Solver.SocketName")

        def assert_options_restored():
            for name, value in numeric_options.items():
                assert gmsh.option.getNumber(name) == value
            assert gmsh.option.getString("Solver.SocketName") == socket_name

        settings = MeshSettings(
            element_size=0.3,
            order=2,
            optimize=False,
            algorithm_2d=Algorithm2D.DELAUNAY,
            algorithm_3d=Algorithm3D.HXT,
            advanced=(
                ("Mesh.MinimumCirclePoints", 12),
                ("Solver.SocketName", "shapemesh-option-test"),
            ),
        )
        build(Disk((0, 0), 1), settings)
        assert_options_restored()

        build(Disk((0, 0), 1), MeshSettings(element_size=0.3, optimize=False))
        assert_options_restored()

        disjoint = Intersection(Box((0, 0, 0), 1, 1, 1), Box((5, 5, 5), 1, 1, 1))
        with pytest.raises(GeometryError):
            build(disjoint, settings)
        assert_options_restored()
    finally:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.finalize()


@pytest.mark.parametrize(
    "name",
    ["General.Terminal", "Mesh.Algorithm", "Mesh.Algorithm3D", "Mesh.ElementOrder"],
)
def test_advanced_options_cannot_override_explicit_settings(name):
    settings = MeshSettings(advanced=((name, 1),))

    with pytest.raises(ValueError, match="conflicts with an explicit setting"):
        build(Box((0, 0, 0), 1, 1, 1), settings)


def test_point_only_high_order_meshing():
    mesh = build(Point((0, 0)), MeshSettings(order=2))

    assert mesh.num_nodes == 1
    assert mesh.num_elements == 1
    assert mesh.metadata.dim == 0


def test_build_error_log_includes_gmsh_last_error():
    settings = MeshSettings(advanced=(("Mesh.DefinitelyInvalid", 1),))

    with pytest.raises(BuildError) as caught:
        build(Box((0, 0, 0), 1, 1, 1), settings)
    assert "Mesh.DefinitelyInvalid" in caught.value.log


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
    assert reread.metadata.dim == mesh.metadata.dim
    assert reread.physical_group("box", 3).entities == mesh.physical_group("box", 3).entities

    # Node tags and coordinates survive exactly.
    assert np.array_equal(reread.nodes.tags, mesh.nodes.tags)
    assert np.allclose(reread.nodes.coords, mesh.nodes.coords)

    # Element blocks survive with the same keys and connectivity shape.
    def block_key(block):
        return (block.entity_dim, block.entity_tag, int(block.type))

    original = {block_key(b): b for b in mesh.element_blocks}
    reread_blocks = {block_key(b): b for b in reread.element_blocks}
    assert set(reread_blocks) == set(original)
    for key, orig in original.items():
        rb = reread_blocks[key]
        assert rb.connectivity.shape == orig.connectivity.shape
        assert np.array_equal(rb.connectivity, orig.connectivity)
        assert np.array_equal(rb.tags, orig.tags)


def test_same_label_and_dimension_merge_into_one_group():
    a = Box((0, 0, 0), 1, 1, 1, label="wall")
    b = Box((2, 0, 0), 1, 1, 1, label="wall")
    mesh = build([a, b], MeshSettings())
    group = mesh.physical_group("wall", 3)
    assert len(group.entities) == 2


# --- §9.5 overlapping-group MSH roundtrip --------------------------------


def test_overlapping_groups_roundtrip_through_gmsh_and_reader(tmp_path):
    a = Box((0, 0, 0), 1, 1, 1, label="a")
    b = Box((0.5, 0, 0), 1, 1, 1, label="b")
    mesh = build([a, b], MeshSettings(element_size=0.5))
    path = tmp_path / "overlap.msh"
    mesh.write_msh(path)

    # Reopen with Gmsh and inspect physical groups.
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))

        groups = gmsh.model.getPhysicalGroups(dim=3)
        group_map = {}
        for dim, tag in groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            group_map[name] = set(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))

        assert set(group_map) == {"a", "b"}
        shared = group_map["a"] & group_map["b"]
        assert len(shared) == 1
        all_volumes = gmsh.model.getEntities(dim=3)
        assert len(all_volumes) == 3
    finally:
        gmsh.finalize()

    # Reopen with shapemesh and verify membership and connectivity.
    reread = Mesh.read_msh(path)
    assert len(reread.entities(dim=3)) == 3

    group_a = set(reread.physical_group("a", 3).entities)
    group_b = set(reread.physical_group("b", 3).entities)
    assert len(group_a) == 2
    assert len(group_b) == 2
    assert len(group_a & group_b) == 1

    overlap_entity = next(iter(group_a & group_b))
    containing = {g.name for g in reread.groups if overlap_entity in set(g.entities)}
    assert containing == {"a", "b"}

    original_blocks = {(b.entity_dim, b.entity_tag, b.type) for b in mesh.element_blocks}
    reread_blocks = {(b.entity_dim, b.entity_tag, b.type) for b in reread.element_blocks}
    assert reread_blocks == original_blocks


# --- boolean geometric semantics -----------------------------------------


def test_boolean_union_and_intersection_volumes():
    a = Box((0, 0, 0), 1, 1, 1)
    b = Box((0.5, 0, 0), 1, 1, 1)
    settings = MeshSettings(element_size=0.4, order=1)

    union = build(Union(a, b, label="u"), settings)
    inter = build(Intersection(a, b, label="i"), settings)
    diff = build(Difference(a, b, label="d"), settings)

    assert _mesh_volume(union) == pytest.approx(1.5, rel=0.05)
    assert _mesh_volume(inter) == pytest.approx(0.5, rel=0.05)
    assert _mesh_volume(diff) == pytest.approx(0.5, rel=0.05)


def test_box_minus_sphere_volume():
    box = Box((0, 0, 0), 1, 1, 1)
    sphere = Sphere((0.5, 0.5, 0.5), 0.2)
    mesh = build(Difference(box, sphere, label="cut"), MeshSettings(element_size=0.15, order=1))
    expected = 1.0 - (4.0 / 3.0) * np.pi * 0.2**3
    assert _mesh_volume(mesh) == pytest.approx(expected, rel=0.05)


def test_nested_boolean_union():
    a = Box((0, 0, 0), 2, 2, 2)
    b = Box((0.5, 0.5, 0.5), 1, 1, 1)
    c = Box((1.5, 0, 0), 1, 1, 1)
    mesh = build(Union(Difference(a, b, label="cut"), c, label="nested"), MeshSettings())
    assert mesh.num_elements > 0
    assert len(mesh.physical_group("nested", 3).entities) == 1


# --- primitive physical-group labeling -----------------------------------


@pytest.mark.parametrize(
    "shape, dim",
    [
        (Point((0, 0, 0), label="p"), 0),
        (Polyline([(0, 0, 0), (1, 0, 0)], label="line"), 1),
        (Polygon([(0, 0, 0), (1, 0, 0), (0, 1, 0)], label="surface"), 2),
        (Disk((0, 0, 0), 1, label="disk"), 2),
        (Sphere((0, 0, 0), 1, label="sphere"), 3),
        (Cylinder((0, 0, 0), (0, 0, 1), 0.5, label="cyl"), 3),
    ],
)
def test_primitive_labels_create_physical_group(shape, dim):
    mesh = build(shape, MeshSettings(element_size=0.5))
    group = mesh.physical_group(shape.label, dim)
    assert len(group.entities) >= 1
