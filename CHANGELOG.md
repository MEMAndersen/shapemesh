# Changelog

## [0.2.0] - 2026-11-01

### Changed

- **Breaking:** Replace `Mesh.to_numpy()`, `NumpyMesh`, and `NumpyElementBlock` with cached, read-only element-type block views from `Mesh.blocks_by_type()` and group row selection through `Mesh.element_indices_in_group()` and `Mesh.node_indices_in_group()` ([`#4`](https://github.com/MEMAndersen/shapemesh/pull/4)) (Mads Kruse)

### Added

- Add cached node tag lookups through `NodeTable.indices_for()` and `NodeTable.coords_for()`, with contiguous, direct-address, and sorted-search strategies ([`#3`](https://github.com/MEMAndersen/shapemesh/pull/3)) (Mads Kruse)

## [0.1.0] - 2026-10-27

_First release._

### Added

- Add 12 shape primitives and n-ary `Union`, `Intersection`, and `Difference` with operator overloads ([`821888c`](https://github.com/MEMAndersen/shapemesh/commit/821888c), [`f315cf6`](https://github.com/MEMAndersen/shapemesh/commit/f315cf6)) (Mads Kruse)
- Add `build()` with label-preserving physical groups and `MeshSettings` for element size, element order, 2D and 3D algorithms, and advanced Gmsh options ([`821888c`](https://github.com/MEMAndersen/shapemesh/commit/821888c), [`647c5e5`](https://github.com/MEMAndersen/shapemesh/commit/647c5e5)) (Mads Kruse)
- Add MSH 4.1 ASCII I/O and the `Mesh` query and NumPy API ([`41efc2b`](https://github.com/MEMAndersen/shapemesh/commit/41efc2b), [`d02d866`](https://github.com/MEMAndersen/shapemesh/commit/d02d866)) (Mads Kruse)
- Add a typed public API with `py.typed` and a structured exception hierarchy ([`821888c`](https://github.com/MEMAndersen/shapemesh/commit/821888c), [`4e30f0e`](https://github.com/MEMAndersen/shapemesh/commit/4e30f0e)) (Mads Kruse)
- Add 1D, 2D, and 3D examples and documentation for labels, threading, and the Gmsh version policy ([`cde6dc0`](https://github.com/MEMAndersen/shapemesh/commit/cde6dc0), [`413dc02`](https://github.com/MEMAndersen/shapemesh/commit/413dc02)) (Mads Kruse)
- Support Python 3.12 through 3.15 on Linux, macOS, and Windows with Gmsh 4.0 or later ([`4f941e1`](https://github.com/MEMAndersen/shapemesh/commit/4f941e1), [`b605815`](https://github.com/MEMAndersen/shapemesh/commit/b605815)) (Mads Kruse)

[0.2.0]: https://github.com/MEMAndersen/shapemesh/releases/tag/v0.2.0
[0.1.0]: https://github.com/MEMAndersen/shapemesh/releases/tag/v0.1.0
