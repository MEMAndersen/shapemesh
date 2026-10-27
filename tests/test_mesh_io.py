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
    ElementBlock,
    ElementType,
    Mesh,
    MeshFormatError,
    MeshSettings,
    NodeTable,
    NumpyElementBlock,
    NumpyMesh,
    PhysicalGroup,
    Point,
    build,
)

MINIMAL_MSH = """\
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
1 3 1 3
2 1 0 3
1
2
3
0 0 0
1 0 0
0 1 0
$EndNodes
$Elements
1 1 1 1
2 1 2 1
1 1 2 3
$EndElements
"""

UNKNOWN_TYPE_MSH = """\
$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 0 0 0
$EndEntities
$Nodes
1 4 1 4
0 1 0 4
1
2
3
4
0 0 0
1 0 0
0 1 0
0 0 1
$EndNodes
$Elements
1 2 1 2
0 1 999 2
1 1 2 3
2 4 1 2
$EndElements
"""

PARAMETRIC_NODES_MSH = """\
$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 1 0 0
1 0 0 0 1 0 0 0 0
$EndEntities
$Nodes
1 2 1 2
1 1 1 2
1
2
0 0 0 0.25
1 0 0 0.75
$EndNodes
$Elements
1 1 1 1
1 1 1 1
1 1 2
$EndElements
"""

EMPTY_ELEMENTS_MSH = MINIMAL_MSH.replace(
    "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements",
    "$Elements\n1 0 0 0\n0 1 15 0\n$EndElements",
)


def _read_minimal_mesh(tmp_path):
    path = tmp_path / "minimal.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    return Mesh.read_msh(path)


def test_read_minimal(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    assert mesh.num_nodes == 3
    assert mesh.num_elements == 1
    assert len(mesh.entities()) == 7  # 3 points + 3 curves + 1 surface
    assert mesh.bbox == (0.0, 0.0, 0.0, 1.0, 1.0, 0.0)

    group = mesh.physical_group("tri", 2)
    assert group.entities == ((2, 1),)
    blocks = mesh.elements_in_group("tri", 2)
    assert len(blocks) == 1
    assert blocks[0].type == ElementType.TRI3
    assert blocks[0].connectivity.tolist() == [[1, 2, 3]]


def test_repr_summaries(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    assert repr(mesh) == "Mesh(dim=2, nodes=3, elements=1, groups=1, bbox=(0, 0, 0, 1, 1, 0))"
    assert repr(mesh.nodes) == "NodeTable(nodes=3)"
    assert repr(mesh.physical_group("tri", 2)) == (
        "PhysicalGroup(name='tri', dim=2, tag=1, entities=1)"
    )
    assert repr(mesh.element_blocks[0]) == "ElementBlock(entity=(2, 1), type=TRI3, elements=1)"
    assert repr(mesh.entities()[0]) == "EntityRecord(dim=0, tag=1)"
    assert repr(mesh.metadata).startswith("MeshMetadata(dim=2, msh_version='4.1', ")

    numpy_mesh = mesh.to_numpy()
    assert repr(numpy_mesh) == "NumpyMesh(nodes=3, blocks=[TRI3])"
    assert repr(numpy_mesh.blocks[ElementType.TRI3]) == (
        "NumpyElementBlock(elements=1, nodes_per_element=3)"
    )


def _assert_mesh_fidelity(original, reread):
    assert reread.num_nodes == original.num_nodes
    assert reread.num_elements == original.num_elements
    assert reread.metadata.dim == original.metadata.dim
    assert len(reread.entities()) == len(original.entities())
    assert np.array_equal(reread.nodes.tags, original.nodes.tags)
    assert np.allclose(reread.nodes.coords, original.nodes.coords)

    def block_key(block):
        return (block.entity_dim, block.entity_tag, int(block.type))

    orig_blocks = {block_key(b): b for b in original.element_blocks}
    reread_blocks = {block_key(b): b for b in reread.element_blocks}
    assert set(reread_blocks) == set(orig_blocks)
    for key, orig in orig_blocks.items():
        rb = reread_blocks[key]
        assert rb.connectivity.shape == orig.connectivity.shape
        assert np.array_equal(rb.connectivity, orig.connectivity)
        assert np.array_equal(rb.tags, orig.tags)


def test_roundtrip(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    out = tmp_path / "out.msh"
    mesh.write_msh(out)
    mesh2 = Mesh.read_msh(out)

    _assert_mesh_fidelity(mesh, mesh2)
    assert mesh2.physical_group("tri", 2).entities == mesh.physical_group("tri", 2).entities


@pytest.mark.parametrize(
    "name",
    ["single\\slash", "double\\\\slash", "trailing\\"],
    ids=["single", "repeated", "trailing"],
)
def test_physical_group_backslashes_roundtrip(tmp_path, name):
    mesh = build(Point((0, 0), label=name), MeshSettings())
    path = tmp_path / "backslash.msh"
    mesh.write_msh(path)

    reread = Mesh.read_msh(path)
    assert reread.physical_group(name, 0).name == name

    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        groups = gmsh.model.getPhysicalGroups()
        assert len(groups) == 1
        assert gmsh.model.getPhysicalName(*groups[0]) == name
    finally:
        gmsh.finalize()


@pytest.mark.parametrize(
    "entity",
    [(2, 999), (1, 1)],
    ids=["missing-entity", "wrong-dimension"],
)
def test_writer_rejects_invalid_group_membership(tmp_path, entity):
    base = _read_minimal_mesh(tmp_path)
    group = PhysicalGroup("invalid", 2, 99, (entity,))
    mesh = Mesh(base.nodes, base.element_blocks, base.entities(), (group,), base.metadata)

    with pytest.raises(ValueError, match="physical group"):
        mesh.write_msh(tmp_path / "out.msh")


def test_writer_rejects_unknown_element_type(tmp_path):
    base = _read_minimal_mesh(tmp_path)
    block = ElementBlock(2, 1, 999, np.array([1]), np.array([[1, 2, 3]]))
    mesh = Mesh(base.nodes, (block,), base.entities(), (), base.metadata)

    with pytest.raises(ValueError, match="unsupported element type"):
        mesh.write_msh(tmp_path / "out.msh")


@pytest.mark.parametrize(
    ("tags", "connectivity"),
    [
        (np.array([1, 2]), np.array([[1, 2, 3]])),
        (np.array([1]), np.array([[1, 2]])),
    ],
    ids=["tag-count", "node-count"],
)
def test_writer_rejects_inconsistent_connectivity(tmp_path, tags, connectivity):
    base = _read_minimal_mesh(tmp_path)
    block = ElementBlock(2, 1, ElementType.TRI3, tags, connectivity)
    mesh = Mesh(base.nodes, (block,), base.entities(), (), base.metadata)

    with pytest.raises(ValueError, match="connectivity"):
        mesh.write_msh(tmp_path / "out.msh")


def test_to_numpy(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    result = mesh.to_numpy()
    assert isinstance(result, NumpyMesh)
    assert result.coords.shape == (3, 3)
    assert ElementType.TRI3 in result.blocks
    block = result.blocks[ElementType.TRI3]
    assert isinstance(block, NumpyElementBlock)
    assert block.connectivity.tolist() == [[0, 1, 2]]
    assert block.node_tags.tolist() == [[1, 2, 3]]


def test_unknown_element_type_skipped(tmp_path):
    path = tmp_path / "unknown.msh"
    path.write_text(UNKNOWN_TYPE_MSH, encoding="utf-8")
    with pytest.warns(UserWarning, match="999"):
        mesh = Mesh.read_msh(path)
    assert mesh.num_nodes == 4
    assert mesh.num_elements == 0


def test_parametric_nodes_rejected(tmp_path):
    path = tmp_path / "parametric.msh"
    path.write_text(PARAMETRIC_NODES_MSH, encoding="utf-8")

    with pytest.raises(MeshFormatError, match="parametric"):
        Mesh.read_msh(path)


@pytest.mark.parametrize(
    "text",
    [
        MINIMAL_MSH.replace("$PhysicalNames\n1\n", "$PhysicalNames\n2\n"),
        MINIMAL_MSH.replace("$EndPhysicalNames", '2 2 "extra"\n$EndPhysicalNames'),
    ],
    ids=["missing", "extra"],
)
def test_physical_names_count_validated(tmp_path, text):
    path = tmp_path / "invalid.msh"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MeshFormatError, match="PhysicalNames"):
        Mesh.read_msh(path)


@pytest.mark.parametrize(
    "text",
    [
        MINIMAL_MSH.replace("$Nodes\n1 3 1 3", "$Nodes\n1 4 1 3"),
        MINIMAL_MSH.replace("$Nodes\n1 3 1 3", "$Nodes\n1 3 0 3"),
        MINIMAL_MSH.replace("$Nodes\n1 3 1 3", "$Nodes\n1 3 1 4"),
    ],
    ids=["total", "minimum", "maximum"],
)
def test_node_tag_summary_validated(tmp_path, text):
    path = tmp_path / "invalid.msh"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MeshFormatError, match="Nodes"):
        Mesh.read_msh(path)


@pytest.mark.parametrize(
    "text",
    [
        MINIMAL_MSH.replace("$Elements\n1 1 1 1", "$Elements\n1 99 1 1"),
        MINIMAL_MSH.replace("$Elements\n1 1 1 1", "$Elements\n1 1 0 1"),
        MINIMAL_MSH.replace("$Elements\n1 1 1 1", "$Elements\n1 1 1 2"),
        MINIMAL_MSH.replace("$Elements\n1 1 1 1", "$Elements\n2 1 1 1"),
        MINIMAL_MSH.replace("$EndElements", "1 1 2 3\n$EndElements"),
    ],
    ids=["total", "minimum", "maximum", "block-count", "extra-data"],
)
def test_element_counts_and_tag_summary_validated(tmp_path, text):
    path = tmp_path / "invalid.msh"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MeshFormatError, match="Elements"):
        Mesh.read_msh(path)


@pytest.mark.parametrize(
    "text",
    [
        MINIMAL_MSH.replace("4.1 0 8", "4.1 nope 8"),
        MINIMAL_MSH.replace("$Nodes\n1 3 1 3", "$Nodes\n1 nope 1 3"),
        MINIMAL_MSH.replace("2 1 2 1\n1 1 2 3\n$EndElements", "2 1 2 1\n$EndElements"),
        MINIMAL_MSH.replace(
            "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements",
            "$Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 2\n$EndElements",
        ),
    ],
    ids=["format-integer", "node-integer", "truncated-elements", "ragged-elements"],
)
def test_malformed_msh_raises_format_error(tmp_path, text):
    path = tmp_path / "malformed.msh"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MeshFormatError):
        Mesh.read_msh(path)


def test_binary_msh_rejected_before_decoding(tmp_path):
    path = tmp_path / "binary.msh"
    path.write_bytes(b"$MeshFormat\n4.1 1 8\n$EndMeshFormat\n\x80\xf0")

    with pytest.raises(MeshFormatError, match="binary"):
        Mesh.read_msh(path)


def test_invalid_utf8_raises_format_error(tmp_path):
    path = tmp_path / "invalid-utf8.msh"
    path.write_bytes(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\xff")

    with pytest.raises(MeshFormatError):
        Mesh.read_msh(path)


def test_empty_element_block_roundtrip(tmp_path):
    path = tmp_path / "empty-block.msh"
    path.write_text(EMPTY_ELEMENTS_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)

    assert mesh.num_elements == 0
    assert len(mesh.element_blocks) == 1
    block = mesh.element_blocks[0]
    assert block.type is ElementType.POINT1
    assert block.tags.shape == (0,)
    assert block.connectivity.shape == (0, 1)

    numpy_block = mesh.to_numpy().blocks[ElementType.POINT1]
    assert numpy_block.connectivity.shape == (0, 1)
    assert numpy_block.node_tags.shape == (0, 1)
    assert repr(numpy_block) == "NumpyElementBlock(elements=0, nodes_per_element=1)"

    out = tmp_path / "empty-block-out.msh"
    mesh.write_msh(out)
    assert "$Elements\n1 0 0 0\n" in out.read_text(encoding="utf-8")
    reread = Mesh.read_msh(out)
    assert reread.element_blocks[0].connectivity.shape == (0, 1)


def test_public_arrays_are_read_only(tmp_path):
    path = tmp_path / "tri.msh"
    path.write_text(MINIMAL_MSH, encoding="utf-8")
    mesh = Mesh.read_msh(path)
    with pytest.raises(ValueError):
        mesh.nodes.coords[0, 0] = 999.0


@pytest.mark.parametrize("field", ["entity_dim", "entity_tag"])
def test_one_sided_node_classification_is_read_only(tmp_path, field):
    base = _read_minimal_mesh(tmp_path)
    if field == "entity_dim":
        nodes = NodeTable(base.nodes.tags, base.nodes.coords, base.nodes.entity_dim.copy(), None)
    else:
        nodes = NodeTable(base.nodes.tags, base.nodes.coords, None, base.nodes.entity_tag.copy())
    mesh = Mesh(nodes, base.element_blocks, base.entities(), base.groups, base.metadata)

    with pytest.raises(ValueError):
        getattr(mesh.nodes, field)[0] = 0


def test_gmsh_roundtrip(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)  # keep the test output quiet
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        path = tmp_path / "box.msh"
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    mesh = Mesh.read_msh(path)
    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert mesh.bbox == pytest.approx((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), abs=1e-6)

    out = tmp_path / "box2.msh"
    mesh.write_msh(out)
    mesh2 = Mesh.read_msh(out)
    _assert_mesh_fidelity(mesh, mesh2)
