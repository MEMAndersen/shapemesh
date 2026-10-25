import pytest

from shapemesh import (
    Cube,
    Cylinder,
    Line,
    MeshSettings,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    Sphere,
    build,
)

_CASES = [
    (Point((0, 0, 0)), 0, 1),
    (Line((0, 0, 0), (1, 0, 0)), 1, 1),
    (Polyline([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), 1, 2),
    (Polygon([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), 2, 1),
    (Rectangle((0, 0, 0), 1, 1), 2, 1),
    (Sphere((0, 0, 0), 1), 3, 1),
    (Cube((0, 0, 0), 1, 1, 1), 3, 1),
    (Cylinder((0, 0, 0), (0, 0, 1), 0.5), 3, 1),
]

_IDS = ["point", "line", "polyline", "polygon", "rectangle", "sphere", "cube", "cylinder"]


@pytest.mark.parametrize(("shape", "dim", "count"), _CASES, ids=_IDS)
def test_primitive_realizes_top_dim_entities(shape, dim, count):
    mesh = build(shape, MeshSettings())
    assert mesh.metadata.dim == dim
    assert len(mesh.entities(dim=dim)) == count
    assert mesh.num_elements > 0


def test_point_only_meshing():
    mesh = build(Point((0, 0, 0)), MeshSettings())
    assert mesh.num_nodes == 1
    assert mesh.num_elements == 1
    assert mesh.metadata.dim == 0


def test_primitive_group_labeling():
    mesh = build(Cube((0, 0, 0), 1, 1, 1, label="box"), MeshSettings())
    assert len(mesh.physical_group("box", 3).entities) == 1
