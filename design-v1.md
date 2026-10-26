# ShapeMesh v1 — Design Plan

**Status:** Design locked; no implementation yet  
**Date:** 2026-10-24  
**Package:** `shapemesh`  
**Source:** Python port of the discontinued Rust `geommesh`

## 1. Purpose

`shapemesh` is a Python geometry and meshing library powered by Gmsh. It provides a small closed set of 0–3D shapes, n-ary booleans, a validated build pipeline, and an entity-level mesh that can be written to and read from Gmsh MSH 4.1 without requiring a live Gmsh session for I/O.

The design keeps the useful part of the Rust pipeline while dropping the Rust FFI layer and its failure modes.

## 2. Goals

- Port the shape primitives (Point, Line, Polyline, Circle, Ellipse, Polygon, Rectangle, Disk, EllipticalDisk, Sphere, Box, Cylinder) and n-ary booleans.
- Validate before touching Gmsh.
- Keep the mesh entity-classified, matching Gmsh's `.msh` model.
- Provide deterministic label → physical group semantics.
- Use the official `gmsh` Python API; no C FFI.
- Serialize to and from MSH 4.1 ASCII, including overlapping physical groups.
- Serialize builds; document thread safety.
- Provide a strict, typed public API.

## 3. Non-goals (v1)

- Transformations (translate/rotate/scale)
- Placement frames (arbitrary orientation) for planar shapes
- Cone, Torus
- Extrude/Revolve
- Fillet/Chamfer
- STEP import
- Mesh size fields
- meshio
- MSH 2.2 read/write
- Binary MSH
- Mesh mutation methods
- Entity/face-level labels
- Detailed mesh size control (per-shape sizes, min/max bounds, curvature-based refinement)

## 4. Locked decisions

| Area | Decision | Reason |
|---|---|---|
| Name | `shapemesh` | Matches repo; PyPI name free as of 2026-10-24 |
| License | LGPL-3.0-or-later | Copyleft on the library itself without blocking non-GPL downstream users |
| Python | `>=3.12` | Modern typing and dataclasses |
| Shapes | Frozen dataclasses | Positional geometry, keyword-only label; `kw_only=True` also dodges the default-ordering error |
| Point | 0D shape plus `Vec3` coordinate type | Gmsh 0D entities contain type-15 point elements; avoids label leakage |
| Labels | Merge duplicate `(label, dim)` | Matches Gmsh physical groups and multi-entity groups |
| Overlaps | Fragment remap | A shared fragment entity can belong to multiple label groups |
| Boolean labels | Operand labels consumed; result label survives | Rust parity |
| Build | Required settings; single shape or sequence | Explicitness |
| Mesh | Hybrid; numpy required | Performance; zero-copy views |
| Node model | Global `NodeTable` plus optional entity tags | MSH4 writer needs blocks; global view needs coords |
| Entity topology | `EntityRecord` | MSH4 `$Entities` writer/reader fidelity |
| MSH | 4.1 ASCII only | Overlapping groups require 4.1 |
| Element types | Static curated enum | Only supported fixed-node types; unknown codes tolerated on read |
| Settings copy | Validated `with_` | `dataclasses.replace` re-runs `__post_init__` |
| Algorithms | Separate 2D/3D enums | Gmsh 2D and 3D algorithm tables differ |
| Gmsh session | Process-wide `RLock`; detect existing; finalize only if owned | Thread safety and coexistence |
| Errors | Small hierarchy | Builtin-compatible, low ceremony |
| Tests | Gmsh required | Integration-first |

## 5. Package layout

```text
shapemesh/
├── pyproject.toml
├── LICENSE
├── README.md
├── AGENTS.md
├── design-v1.md
├── src/
│   └── shapemesh/
│       ├── __init__.py
│       ├── constants.py
│       ├── shapes.py
│       ├── settings.py
│       ├── mesh.py        # Mesh + MSH 4.1 reader/writer
│       ├── builder.py
│       └── errors.py
└── tests/
    ├── test_shapes.py
    ├── test_settings.py
    ├── test_mesh_io.py
    ├── test_primitives.py
    ├── test_booleans.py
    ├── test_pipeline.py
    └── test_threading.py
```

`__init__.py` re-exports the public API (`Point`, `build`, `Mesh`, `MeshSettings`, the `ShapeMeshError` family, `ElementType`, etc.) via an explicit `__all__`.

If MSH I/O grows too large, split `mesh.py` into `mesh.py` and `msh.py`.

## 6. Public API

### 6.1 Shapes

```python
Point(coord, *, label=None)
Line(start, end, *, label=None)
Polyline(points, *, label=None)             # closed iff points[0] == points[-1]
Circle(center, radius, *, label=None)       # 1D boundary curve; any constant z
Ellipse(center, rx, ry, *, label=None)      # 1D boundary curve; any constant z
Polygon(points, *, label=None)              # any plane
Rectangle(origin, dx, dy, *, label=None)    # axis-aligned; any constant z
Disk(center, radius, *, label=None)         # 2D filled circle; any constant z
EllipticalDisk(center, rx, ry, *, label=None)  # 2D filled ellipse; any constant z
Sphere(center, radius, *, label=None)
Box(origin, dx, dy, dz, *, label=None)
Cylinder(start, end, radius, *, label=None) # provisional signature; see §17
Union(*operands, *, label=None)
Intersection(*operands, *, label=None)
Difference(*operands, *, label=None)        # first operand minus union of the rest
```

Rules:

- `Vec3Like` accepts `Vec3`, tuple/list, or ndarray of 2–3 floats.
- 2D input pads `z=0`.
- Coordinates must be finite.
- `Point` is rejected as a vertex coordinate; use `.coord`.
- `label` is keyword-only, non-empty, stripped, at most 127 characters (a defensive cap; MSH 4.1 imposes no such limit), and must not contain quotes or newlines.
- Shapes are frozen and hashable.
- `a | b`, `a & b`, and `a - b` desugar to binary `Union`, `Intersection`, and `Difference`.
- N-ary booleans use the explicit classes, accepting either variadic shapes or a single sequence (`Union(a, b, c)` or `Union([a, b, c])`).
- Booleans require at least two operands of the same topological dimension.
- An empty boolean result raises `GeometryError`.
- `Difference(a, b, c)` means `a - (b ∪ c)`.

### 6.2 Build

```python
def build(shapes: Shape | Sequence[Shape], settings: MeshSettings) -> Mesh
```

- `settings` is required; no default.
- A single `Shape` is normalized to a one-element sequence.
- An empty sequence raises `GeometryError`.

### 6.3 Mesh

```python
mesh = Mesh.read_msh(path)
mesh.write_msh(path)
mesh.to_numpy()

mesh.physical_group(name, dim)
mesh.physical_groups_named(name)
mesh.elements_in_group(name, dim)
mesh.entities(dim=None)
mesh.elements_on_entity(dim, tag)

mesh.num_nodes
mesh.num_elements
mesh.bbox

mesh.nodes
mesh.element_blocks
mesh.groups
mesh.metadata
```

- Public numpy arrays are read-only views.
- v1 has no mutation methods.
- `physical_group(name)` alone is ambiguous across dimensions; require `dim` or use `physical_groups_named(name)`.
- `elements_in_group(name, dim)` may return the same `ElementBlock` for multiple groups when groups overlap.

### 6.4 Errors

```python
ShapeMeshError(Exception)
├── GeometryError(ShapeMeshError, ValueError)   # invalid shape, empty boolean result
├── LabelError(ShapeMeshError, ValueError)      # invalid or conflicting labels
├── MeshFormatError(ShapeMeshError, ValueError) # corrupt or unsupported MSH
└── BuildError(ShapeMeshError, RuntimeError)    # Gmsh failure; carries log text
```

- `BuildError` exposes `log: str`.
- MSH format problems raise `MeshFormatError`.
- Settings validation raises `ValueError` (from the dataclass `__post_init__`).
- Structured library errors (geometry, label, build, MSH-format) all derive from `ShapeMeshError`; settings validation and writer-side data-consistency checks raise plain `ValueError`.

## 7. Data model

### 7.1 Shape base

```python
type Vec3Like = Vec3 | Sequence[float] | np.ndarray


@dataclass(frozen=True, slots=True, kw_only=True)
class Shape:
    label: str | None = None


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class Point(Shape):
    coord: Vec3


@dataclass(frozen=True, slots=True)
class Line(Shape):
    start: Vec3
    end: Vec3
```

- `kw_only=True` on `Shape` keeps `label` keyword-only *and* lets subclass geometry fields be positional without a default; otherwise the base default field would trigger the "non-default argument follows default argument" dataclass error.
- Subclass geometry fields stay positional: `Line((0, 0, 0), (1, 0, 0), label="edge")`.
- `Vec3Like` includes `np.ndarray` explicitly because ndarray is not a `typing.Sequence`; `str` is a `Sequence` but is rejected at runtime (see §13).
- `__post_init__` normalizes `Vec3Like` via `Vec3.from_any()` and validates.
- `Polyline`/`Polygon` normalize `points` to `tuple[Vec3, ...]` (never a list or ndarray) so shapes stay hashable.
- Each shape exposes a `dimension` class variable/property.
- Booleans use custom `init=False` constructors to accept varargs:

```python
@dataclass(frozen=True, slots=True, init=False)
class Union(Shape):
    operands: tuple[Shape, ...]

    def __init__(
        self,
        *args: Shape,
        label: str | None = None,
        operands: tuple[Shape, ...] | None = None,
    ) -> None:
        if operands is not None:
            if args:
                raise TypeError("operands cannot be supplied both positionally and by keyword")
            args = (operands,)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operands", _coerce_operands(args))
        self.__post_init__()
```

The keyword-only `operands` path supports `dataclasses.replace`; positional and
keyword operand forms are mutually exclusive.

### 7.2 Labels and physical groups

```python
groups: dict[tuple[str, int], set[tuple[int, int]]] = {}
# ("wall", 2) -> {(2, 17), (2, 18)}
```

- Key is `(label, dim)`.
- Duplicate labels of the same dimension merge into one physical group.
- Different dimensions are separate groups.
- Fragment preserves labels: a shared output entity appears in every group whose input map contains it.
- Boolean operations consume operand labels; the result label survives.
- Labels lost in booleans disappear silently; this is documented.
- Remap rules after fragment:
  - Work per label.
  - Include all output entities from that label's input map.
  - Filter output entities by the shape's dimension.
  - Dedupe within a label.
  - Never dedupe across labels.
- "Deterministic" (§2) applies to the label → entity-set mapping (names are stable); Gmsh-assigned physical-group tag integers are not guaranteed stable across runs, so tests assert names, not tags.

### 7.3 Mesh structures

```python
@dataclass(frozen=True, slots=True)
class EntityRecord:
    dim: int
    tag: int
    bbox: tuple[float, float, float, float, float, float]
    boundary: tuple[tuple[int, int], ...] = ()  # signed tags encode orientation


@dataclass(frozen=True, slots=True, eq=False)
class NodeTable:
    tags: np.ndarray  # (N,) int
    coords: np.ndarray  # (N, 3) float
    entity_dim: np.ndarray | None = None  # (N,), -1 if unknown
    entity_tag: np.ndarray | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ElementBlock:
    entity_dim: int
    entity_tag: int
    type: ElementType
    tags: np.ndarray  # (M,) int
    connectivity: np.ndarray  # (M, k), global node tags


class Mesh:
    _nodes: NodeTable
    _elements: tuple[ElementBlock, ...]
    _entities: tuple[EntityRecord, ...]
    _groups: tuple[PhysicalGroup, ...]
    _metadata: MeshMetadata
```

- `PhysicalGroup` is a frozen dataclass:
  - `name: str`
  - `dim: int`
  - `tag: int`
  - `entities: tuple[tuple[int, int], ...]`
- `MeshMetadata` is a frozen dataclass:
  - `dim: int`
  - `msh_version: str = "4.1"`
  - `gmsh_version: str | None = None`
  - `source: str | None = None`
- `num_nodes`, `num_elements`, and `bbox` are derived `Mesh` properties.
- `ElementBlock` keys are `(entity_dim, entity_tag, ElementType)`; duplicate blocks on read are merged.
- `ElementBlock.connectivity` stores global node tags, not zero-based indices.
- All public arrays are set non-writeable and exposed read-only.
- `NodeTable`/`ElementBlock` use `eq=False` (and thus `__hash__ = None`): the default dataclass `__eq__` would compare ndarray fields with `==`, raising on ambiguous truth values.
- `EntityRecord.bbox` uses six values even for points; the writer emits three coordinates for dim 0.

### 7.4 `to_numpy()`

```python
@dataclass(frozen=True, slots=True, eq=False)
class NumpyElementBlock:
    connectivity: np.ndarray  # (M, k), 0-based indices into coords
    node_tags: np.ndarray  # (M, k), original Gmsh tags


@dataclass(frozen=True, slots=True, eq=False)
class NumpyMesh:
    coords: np.ndarray  # (N, 3), global
    blocks: dict[ElementType, NumpyElementBlock]
```

- Connectivity indices are zero-based into `coords`.
- `node_tags` preserves the original Gmsh tags.
- A missing node tag referenced by an element raises `ValueError`.
- Views are read-only; callers copy if they need mutable arrays.
- `blocks` is keyed by `ElementType` only, so blocks from distinct entities of the same type are merged and entity association is lost; this view is for numerics, not entity queries.

### 7.5 Element types

`ElementType` is a static, hand-curated `IntEnum` listing only the fixed-node element types `shapemesh` supports. It is *not* auto-generated from `getElementProperties`: a mechanical dump would include variable-node types (Polygon 34, Polyhedron 35, `num_nodes == 0`) that cannot fit the fixed-width `(M, k)` connectivity.

Common members:

| Code | Type | Nodes | Dim |
|---:|---|---:|---:|
| 15 | Point1 | 1 | 0 |
| 1 | Line2 | 2 | 1 |
| 8 | Line3 | 3 | 1 |
| 2 | Tri3 | 3 | 2 |
| 9 | Tri6 | 6 | 2 |
| 3 | Quad4 | 4 | 2 |
| 10 | Quad9 | 9 | 2 |
| 16 | Quad8 | 8 | 2 |
| 4 | Tet4 | 4 | 3 |
| 11 | Tet10 | 10 | 3 |
| 5 | Hex8 | 8 | 3 |
| 12 | Hex27 | 27 | 3 |
| 17 | Hex20 | 20 | 3 |
| 6 | Prism6 | 6 | 3 |
| 13 | Prism18 | 18 | 3 |
| 18 | Prism15 | 15 | 3 |
| 7 | Pyramid5 | 5 | 3 |
| 14 | Pyramid14 | 14 | 3 |
| 19 | Pyramid13 | 13 | 3 |

The table includes the higher-order fixed-node types (e.g. 92 = Hexahedron 64, 93 = Hexahedron 125). Variable-node types (Polygon/Polyhedron) are excluded. Unknown type codes are tolerated on read (see §9.3) but rejected by the writer and the `ElementType`-keyed API. A Gmsh update that introduces a new type requires a library update to the enum.

### 7.6 Settings

```python
@dataclass(frozen=True, slots=True)
class MeshSettings:
    element_size: float | None = None   # per-point size (mesh.setSize)
    order: int = 1                                # 1..5
    optimize: bool = True
    verbose: bool = False  # show Gmsh terminal output (silenced by default)
    algorithm_2d: Algorithm2D | None = None
    algorithm_3d: Algorithm3D | None = None
    advanced: tuple[tuple[str, float | int | str], ...] = ()

    def __post_init__(self) -> None:
        # validate ranges/types; raise ValueError on violation

    def with_(self, **changes: object) -> MeshSettings:
        return dataclasses.replace(self, **changes)  # re-runs __post_init__
```

- `MeshSettings` is a plain frozen dataclass, not a pydantic model: the generated `__init__` rejects unknown kwargs (the `extra="forbid"` equivalent), and there is no coercion (e.g. `order="3"` is a `TypeError`), matching the strict-typed goal. `with_` uses `dataclasses.replace`, which re-runs `__post_init__`.

Algorithm enums follow Gmsh 4.15.2:

| 2D `Mesh.Algorithm` | Value |
|---|---:|
| MeshAdapt | 1 |
| Automatic | 2 |
| Initial mesh only | 3 |
| Delaunay | 5 |
| Frontal-Delaunay | 6 |
| BAMG | 7 |
| Frontal-Delaunay for Quads | 8 |
| Packing of Parallelograms | 9 |
| Quasi-structured Quad | 11 |

| 3D `Mesh.Algorithm3D` | Value |
|---|---:|
| Delaunay | 1 |
| Initial mesh only | 3 |
| Frontal | 4 |
| MMG3D | 7 |
| R-tree | 9 |
| HXT | 10 |

- For a 2D model, only `algorithm_2d` applies.
- For a 3D model, both `algorithm_2d` and `algorithm_3d` apply.
- `None` leaves the Gmsh default.

## 8. Build pipeline

```text
validate shapes, labels, booleans
acquire process-wide RLock
initialize gmsh if needed
new model; apply settings
realize shapes/booleans -> live entities + (label, dim) -> [dimtags]
occ.synchronize()
if len(live) > 1: fragment(live, []) -> remap labels, filter by dim
occ.synchronize()
add one physical group per (label, dim)
generate(max(model_dim, 1))
read EntityRecords, NodeTable, ElementBlocks, PhysicalGroups, metadata
capture logs; clear model; finalize only if we initialized
release lock
```

### 8.1 Shape realization

| Shape | OCC realization | Tracked dimension |
|---|---|---|
| Point | `occ.addPoint` | 0 |
| Line | `occ.addLine` | 1 |
| Polyline | points + individual `occ.addLine` edges | 1 |
| Circle | `occ.addCircle` | 1 |
| Ellipse | `occ.addEllipse` | 1 |
| Polygon | points + lines + `occ.addCurveLoop` + `occ.addPlaneSurface` | 2 |
| Rectangle | `occ.addRectangle` | 2 |
| Disk | `occ.addDisk` (r, r) | 2 |
| EllipticalDisk | `occ.addDisk` (rx, ry) | 2 |
| Sphere | `occ.addSphere` | 3 |
| Box | `occ.addBox` | 3 |
| Cylinder | `occ.addCylinder` | 3 |

Notes:

- `addWire` and `addCurveLoop` are topology containers, not model entities. They are never tracked as meshable shapes.
- A `Polyline` labels its individual edges.
- A `Polygon` labels its surface, not its boundary curves.
- `occ.synchronize()` is still required in Python. The Rust FFI wrapper is dropped, not the Gmsh synchronization call.
- `fragment` is skipped when there are zero or one live entities; the label map is the identity in that case.
- `outDimTagsMap` may include lower-dimensional outputs; filter by the input shape dimension before assigning labels.
- `generate(max(model_dim, 1))` is deliberate: `generate(0)` on a point-only model reports success but emits no elements, whereas `generate(1)` emits the type-15 point elements.

### 8.2 Overlapping groups

Two overlapping blocks with labels `"a"` and `"b"`:

1. `fragment` splits them into three volume pieces.
2. The label map for `"a"` includes the A-only and intersection pieces.
3. The label map for `"b"` includes the B-only and intersection pieces.
4. The intersection piece is written into both physical groups.
5. MSH4 `$Entities` lists both physical tags for that entity.

This is the required overlap mechanism; no multi-label-per-shape API is needed.

## 9. MSH 4.1 I/O

### 9.1 Scope

- Read and write MSH 4.1 ASCII only.
- Reject 2.2, 4.0, binary, partitioned, periodic, ghost, and parametric sections with clear errors.
- `$Entities` is optional on read only when there are no physical names; otherwise its absence is an error.
- `gmsh_version` is `None` when reading, because MSH4 does not store the Gmsh version.

### 9.2 Writer

Section order:

1. `$MeshFormat` — `4.1 0 8`
2. `$PhysicalNames` — one line per group
3. `$Entities` — bbox, boundary entities, all physical tags per entity
4. `$Nodes` — blocks grouped by `(entity_dim, entity_tag)`
5. `$Elements` — blocks grouped by `(entity_dim, entity_tag, element type)`

Rules:

- `$Nodes` uses the 4.1 ordering: all node tags first, then all coordinates.
- Node classification comes from `NodeTable.entity_dim/entity_tag`.
- If classification is absent, reconstruct the lowest-dimensional entity that uses the node from element connectivity.
- Orphan nodes that cannot be classified raise `ValueError`.
- `$Entities` physical tags are derived from `PhysicalGroup` membership.
- A group that references an entity with no `EntityRecord` raises `ValueError`.
- Entity boundary tags come from `EntityRecord.boundary`; build populates them with `gmsh.model.getBoundary(..., oriented=True)`.
- Floats use a round-trip-safe representation.
- Missing `$Entities` is never written: always emit it when entity records exist.

### 9.3 Reader

- Parse `$PhysicalNames`, `$Entities`, `$Nodes`, `$Elements`.
- Build `EntityRecord`, global `NodeTable`, `ElementBlock`, and `PhysicalGroup` collections.
- Merge duplicate element blocks with the same key.
- Validate counts and connectivity widths.
- Unknown element types are tolerated: the `$Elements` block header (`entityDim entityTag elementType numElements`) carries no node width, but each element is one line, so the width is read from the first element line; such blocks are consumed and skipped (with a `UserWarning`) instead of failing. The writer never emits them.
- If `$Entities` is absent and no physical names exist, synthesize entity records from element blocks with empty boundaries.
- If `$PhysicalNames` exists without `$Entities`, raise `MeshFormatError`.
- Set `metadata.source` to the input path and `metadata.msh_version` to `"4.1"`.

### 9.4 Why MSH 2.2 is rejected

MSH2.2 elements carry one physical tag. It cannot represent an entity that belongs to multiple physical groups. Overlapping groups are a requirement, so 2.2 is not a valid v1 format. A future 2.2 export would be lossy for overlaps.

### 9.5 Roundtrip test

Build two overlapping labeled boxes, write 4.1, reopen with Gmsh, then reopen with `Mesh.read_msh`. Verify:

- three volume pieces;
- the intersection piece appears in both groups;
- the overlap entity has two physical tags in `$Entities`;
- element connectivity and group membership survive.

## 10. Settings semantics

- `element_size` sets the target mesh size at every model point via `gmsh.model.mesh.setSize` before meshing. `h` must be positive.
- `order` is `1..5`; applied via `gmsh.option.setNumber("Mesh.ElementOrder", order)`
  before `generate`, which is given the model dimension. Setting the option before
  generation is required; `gmsh.model.mesh.setOrder` only changes an existing mesh.
- `optimize=True` calls `optimize("")` for linear meshes and `optimize("HighOrder")`
  for order > 1. High-order optimization is skipped for 0D models.
- `verbose=True` restores Gmsh terminal output, which is silenced by default.
- `algorithm_2d` maps to `Mesh.Algorithm`; `algorithm_3d` maps to `Mesh.Algorithm3D`.
- `advanced` is a tuple of `(option_name, value)` pairs applied after explicit fields. Integer/float values use `gmsh.option.setNumber`; strings use `setString`. Keys that conflict with explicit fields are rejected.
- `with_` merges and revalidates via `dataclasses.replace`, which re-runs `__post_init__`; unknown kwargs raise `TypeError`.

## 11. Errors and diagnostics

- Shape validation raises `GeometryError` or `LabelError`.
- Gmsh call failures raise `BuildError` with the last Gmsh log text.
- Empty boolean results raise `GeometryError`.
- MSH format problems raise `MeshFormatError`.
- Before finalizing Gmsh, capture `gmsh.logger.get()` / `getLastError()` so the log survives cleanup.
- Error messages include the shape index, label, and operation where possible.

## 12. Threading and Gmsh lifecycle

- A process-wide `threading.RLock` wraps the entire build, not just `initialize`/`finalize`.
- `gmsh.isInitialized()` is checked before initialization.
- `shapemesh` finalizes Gmsh only if it initialized it.
- When a build uses an existing Gmsh session, every global option it touches is
  restored during cleanup, including after failures.
- `build()` must be called from the main thread to *initialize* Gmsh: Gmsh installs a SIGINT handler (kept so Ctrl-C stops meshing gracefully), which Python only permits on the main thread. Once Gmsh is initialized on the main thread, worker threads may call `build()` (serialized by the lock).
- A new model is created per build and cleared before release.
- `gmsh` is imported lazily inside `build()`; importing `shapemesh` does not initialize Gmsh.
- Builds serialize. Use multiprocessing for parallel meshing.
- Direct concurrent use of the Gmsh Python API outside `shapemesh` is unsupported and documented as such.

## 13. Validation rules

| Shape | Rules |
|---|---|
| Vec3 | Finite floats; 2D pads `z=0`; reject strings, `Point`, and non-finite values |
| Point | Valid `Vec3` |
| Line | Start ≠ end |
| Polyline | ≥2 points; consecutive points distinct; closed iff first == last; closed form needs ≥3 distinct points |
| Circle | Radius > 0 |
| Ellipse | `rx > 0`, `ry > 0` |
| Polygon | ≥3 points; all vertices distinct; no repeated final point; consecutive points distinct; non-collinear; planar within `1e-7 * scale` |
| Rectangle | `dx > 0`, `dy > 0` |
| Disk | Radius > 0 |
| EllipticalDisk | `rx > 0`, `ry > 0` |
| Sphere | Radius > 0 |
| Box | `dx > 0`, `dy > 0`, `dz > 0` |
| Cylinder | Start ≠ end; radius > 0 |
| Booleans | ≥2 operands; same dimension; non-empty result |
| Labels | Non-empty, stripped, ≤127 chars, no quotes/newlines |
| Build | Non-empty shape sequence; required settings |

`scale` for polygon planarity is the bounding-box diagonal of the polygon. Document the tolerance choice.

## 14. Tests

- Port Rust primitives and pipeline tests.
- Every primitive realizes the correct top-dimensional entity/entities and meshes with its group.
- Union/intersection/difference on overlapping boxes; volume checks.
- Box minus sphere ≈ 0.966.
- Two overlapping boxes fragment into three conformal pieces.
- Nested `(a - b) + c`.
- Line-in-box mixed-dimension fragment.
- Overlapping-group test described in §8.2.
- Label merge test: same label and dimension produce one group containing both entities.
- Point-only 0D meshing.
- Polyline/Polygon entity-count tests.
- Empty boolean → `GeometryError`.
- MSH4 roundtrip through Gmsh and through `Mesh.read_msh`.
- Unknown element type tolerance test (reader skips unknown types; writer rejects them).
- Read-only array test.
- Main-thread-only build guard (worker thread without Gmsh raises a clear error); worker-thread builds with Gmsh pre-initialized serialize.
- Settings validation and `with_` test.
- `.msh` re-open roundtrip.

## 15. Packaging and tooling

- Build backend: hatchling.
- Environment and lockfile: uv.
- Linting/formatting: ruff.
- Type checking: pyright.
- Tests: pytest.
- Python: `>=3.12`.
- Runtime dependencies: `numpy`, `gmsh>=4.15`.
- Dev dependencies: pytest, ruff, pyright.
- License: LGPL-3.0-or-later, with Gmsh attribution in the README.
- CI matrix: Python 3.12–3.15 on Linux, macOS, Windows; Gmsh 4.15.2 (set up after the feature branch is squash-merged).
- Gmsh policy: tested with 4.15.2; newer 4.x allowed. New element types require a `shapemesh` update.
- Version: `0.1.0`.

## 16. Roadmap (not v1)

- Transforms: translate/rotate/scale
- Placement frames (arbitrary orientation) for planar shapes
- Cone, Torus
- Extrude/Revolve
- Fillet/Chamfer
- STEP import
- Mesh size fields
- MSH2.2 export (lossy for overlapping groups)
- Binary MSH
- meshio bridge
- Mesh mutation: filter/renumber
- Entity/face-level labels
- Detailed mesh size control (per-shape sizes, min/max bounds, curvature-based refinement)

## 17. Open items and assumptions

1. **Cylinder signature** is provisional: `Cylinder(start, end, radius)`, with `axis = end - start`. Alternative: `Cylinder(base, axis, radius, height)`. Confirm before implementation.
2. **Primitive signatures assumed**: `Box(origin, dx, dy, dz)`, `Rectangle(origin, dx, dy)`, `Sphere(center, radius)`, `Polyline(points)` with closure inferred; `Circle(center, radius)`, `Ellipse(center, rx, ry)`, `Disk(center, radius)`, `EllipticalDisk(center, rx, ry)` (axis-aligned, any constant z).
3. **Boundary orientation**: `EntityRecord.boundary` stores signed tags. Build uses `getBoundary(..., oriented=True)`; the reader preserves signs. If orientation is unknown, empty boundary is allowed.
4. **Meshless entities**: the reader preserves entities even if they have no elements; the writer requires an `EntityRecord` for every entity referenced by a group.
5. **Gmsh version metadata**: `gmsh_version` is available only for builds, not for MSH4 reads.
6. **Label loss in booleans** is silent and documented.
7. **Static element enum**: only the curated `ElementType` members are supported; unknown type codes are tolerated on read (skipped by reading the per-element node width from the first element line) but rejected by the writer and the `ElementType`-keyed API.

## 18. Next steps

1. Scaffold the repository and packaging.
2. Implement `errors.py`, `shapes.py`, and `settings.py`.
3. Implement `mesh.py` plus MSH 4.1 reader/writer.
4. Implement `builder.py` with the pipeline in §8.
5. Port tests and add overlap/roundtrip tests.
6. Add README, CI, and documentation for labels, thread safety, and the Gmsh version policy.

No implementation code is part of this design phase.
