# shapemesh

A Gmsh-powered geometry and meshing library: a closed set of 0–3D shapes,
n-ary booleans, and entity-level MSH 4.1 I/O.

> **Status:** design locked, not yet implemented.

## Features

- 8 shape primitives: Point, Line, Polyline, Polygon, Rectangle, Sphere, Cube, Cylinder
- n-ary boolean `Union` / `Intersection` / `Difference`
- Overlapping physical groups via fragment remap
- MSH 4.1 ASCII read/write (no live Gmsh session required for I/O)

## Install

```bash
pip install shapemesh
```

## Quick start

<!-- TODO: add once the public API is implemented. -->

## License

LGPL-2.1-or-later. Gmsh — a runtime dependency — is GPLv2-or-later.
