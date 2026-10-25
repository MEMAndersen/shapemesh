import pytest

from shapemesh import Algorithm2D, Algorithm3D, MeshSettings


def test_defaults():
    s = MeshSettings()
    assert s.characteristic_length is None
    assert s.order == 1
    assert s.optimize is True
    assert s.algorithm_2d is None
    assert s.algorithm_3d is None
    assert s.advanced == ()


def test_characteristic_length_validation():
    assert MeshSettings(characteristic_length=0.5).characteristic_length == 0.5
    with pytest.raises(ValueError):
        MeshSettings(characteristic_length=0)
    with pytest.raises(ValueError):
        MeshSettings(characteristic_length=-1)


def test_order_validation():
    assert MeshSettings(order=5).order == 5
    for bad in (0, 6, 2.5, True, "3"):
        with pytest.raises(ValueError):
            MeshSettings(order=bad)


def test_optimize_must_be_bool():
    with pytest.raises(ValueError):
        MeshSettings(optimize=1)


def test_verbose_defaults_false_and_must_be_bool():
    assert MeshSettings().verbose is False
    assert MeshSettings(verbose=True).verbose is True
    with pytest.raises(ValueError):
        MeshSettings(verbose=1)


def test_algorithm_enums():
    assert Algorithm2D.DELAUNAY == 5
    assert Algorithm2D.QUASI_STRUCTURED_QUAD == 11
    assert Algorithm3D.HXT == 10
    s = MeshSettings(algorithm_2d=Algorithm2D.DELAUNAY, algorithm_3d=Algorithm3D.HXT)
    assert s.algorithm_2d is Algorithm2D.DELAUNAY
    with pytest.raises(ValueError):
        MeshSettings(algorithm_2d=Algorithm3D.HXT)  # wrong enum type


def test_advanced_validation():
    s = MeshSettings(advanced=(("Mesh.MinimumCirclePoints", 8),))
    assert s.advanced == (("Mesh.MinimumCirclePoints", 8),)
    with pytest.raises(ValueError):
        MeshSettings(advanced=(("Mesh.X", True),))  # bool value rejected
    with pytest.raises(ValueError):
        MeshSettings(advanced=(("not a pair",),))


def test_with_revalidates():
    s = MeshSettings(order=2)
    s2 = s.with_(order=3)
    assert s.order == 2  # original is unchanged (frozen)
    assert s2.order == 3
    with pytest.raises(ValueError):
        s.with_(order=10)
    with pytest.raises(TypeError):
        s.with_(nonsense=1)  # unknown field


def test_frozen():
    s = MeshSettings()
    with pytest.raises(AttributeError):
        s.order = 3
