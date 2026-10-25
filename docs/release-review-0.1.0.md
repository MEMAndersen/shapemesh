# ShapeMesh 0.1.0 release review

**Review date:** 2026-10-25  
**Reviewed commit:** `cde6dc0` on `feat/shapes-settings`  
**Remediation branch:** `fix/review-findings`  
**Behavioral baseline:** `design-v1.md`

This document records the pre-release review and tracks remediation. The review
found the source layout and basic toolchain healthy, but identified correctness,
MSH I/O, packaging, licensing, documentation, and test-coverage issues that must
be addressed or consciously accepted before tagging 0.1.0.

## Status summary

- **H1 — High:** element order is not applied during generation. **Fixed.**
- **H2 — High:** parametric node blocks silently corrupt coordinates. **Fixed.**
- **H3 — High release hazard:** stale distributions. **Resolved externally:**
  the user deleted the old `dist/`; clean release artifacts still need to be built
  from the final release commit.
- All other findings remain open unless their status is changed below.

## High-severity findings

### H1. `MeshSettings.order` does not produce higher-order meshes

**Status:** Fixed  
**References:** `src/shapemesh/builder.py:174-180`; `design-v1.md:525-526`

`gmsh.model.mesh.setOrder()` is called before generation, when no mesh exists.
It does not configure the subsequent `generate()` call. An order-3 box produced
only `POINT1`, `LINE2`, `TRI3`, and `TET4`, all linear elements.

Required remediation:

- Set `Mesh.ElementOrder` before generation instead of calling `setOrder()` on
  the empty mesh.
- Add an integration regression test that asserts higher-order element types.
- Correct the implementation direction in design section 10.

### H2. Parametric MSH nodes silently corrupt coordinates

**Status:** Fixed  
**Reference:** `src/shapemesh/mesh.py:461-466`

The reader discards the parametric flag and always consumes three coordinates
per node. A valid parametric curve block that should produce coordinates
`[[0, 0, 0], [1, 0, 0]]` was read as `[[0, 0, 0], [0.25, 1, 0]]`.
Design section 9.1 requires parametric input to be rejected.

Required remediation:

- Retain the block's parametric flag and raise `MeshFormatError` when nonzero.
- Reject the block before consuming any parametric coordinates.
- Add a regression test using the corrupting curve-block example.

### H3. Existing distributions were stale and unsafe to publish

**Status:** Resolved externally; final artifact verification remains a release
step  
**References:** former `dist/shapemesh-0.1.0-py3-none-any.whl` and
`dist/shapemesh-0.1.0.tar.gz`

The old archives exported `Cube` rather than `Box`, contained an obsolete README
and primitive set, omitted newer examples, and included only the old `LICENSE`
instead of `COPYING` and `COPYING.LESSER`.

The user deleted these artifacts after the review. A fresh build from the
reviewed source contained the current API and both license files, so this was a
stale-artifact problem rather than a current Hatchling license-configuration
problem. Build both distributions from the exact final release commit into an
empty `dist/` and inspect them before upload.

## Medium-severity geometry and settings findings

### M1. Valid ellipses with `rx < ry` cannot be built

**Status:** Fixed

**References:** `src/shapemesh/shapes.py:443-446`, `:605-607`

Both `Ellipse((0, 0), 1, 2)` and `EllipticalDisk((0, 0), 1, 2)` raise Gmsh's
`BuildError: Major radius rx should be larger than minor radius ry`, although
the public contract permits independent positive x/y radii. Swap the OCC radii
and rotate or explicitly select the major axis while preserving the documented
x/y extents. Do not merely add an `rx >= ry` restriction.

### M2. Gmsh options leak between builds

**Status:** Fixed

**References:** `src/shapemesh/builder.py:95-98`, `:132-145`

Gmsh options are process-global. In a preinitialized session, a build using
`Algorithm2D.DELAUNAY` changed `Mesh.Algorithm` to 5; a subsequent default build
retained 5 instead of the fresh default 6 and reproduced the previous mesh.
Advanced options also persist. Snapshot and restore every touched option, or
otherwise guarantee the documented per-build behavior. Include every option
introduced by the H1 fix.

### M3. `advanced` can override explicit settings

**Status:** Fixed

**References:** `src/shapemesh/builder.py:53`, `:126-145`

Only the two algorithm options are reserved. `Mesh.ElementOrder` conflicts with
`order`, and `General.Terminal` conflicts with `verbose`, yet both are accepted.
Reserve every Gmsh key owned by an explicit setting and reject conflicts before
touching Gmsh.

### M4. Point-only high-order builds fail with default optimization

**Status:** Fixed

**Reference:** `src/shapemesh/builder.py:174-180`

`build(Point((0, 0)), MeshSettings(order=2))` raises
`BuildError: Unknown element type for ordered monomials: 1`; disabling
optimization succeeds. Skip high-order optimization for a 0D model.

### M5. Numeric validation occurs before float conversion

**Status:** Fixed

**References:** `src/shapemesh/shapes.py:80-87`, `:156-165`;
`src/shapemesh/settings.py:60-67`

`Fraction(1, 10**400)` passes validation and is then stored as `0.0` for shape
radii and `MeshSettings.element_size`. Huge integers such as `10**1000` leak
`OverflowError`. Convert to float with appropriate exception handling, then
validate finiteness and positivity on the converted value.

### M6. Repeated non-consecutive polygon vertices bypass validation

**Status:** Fixed

**References:** `src/shapemesh/shapes.py:452-455`, `:757-766`

`Polygon([(0,0), (1,0), (0,1), (1,0)])` is accepted despite the distinct-vertex
contract and later fails in Gmsh with `Curve loop is not closed`. Require all
normalized vertices to be unique before touching Gmsh.

## Medium-severity MSH and NumPy findings

### M7. Legal backslashes in physical-group names do not round-trip

**Status:** Open  
**References:** `src/shapemesh/mesh.py:381-393`, `:822-826`;
`src/shapemesh/shapes.py:64-77`

The writer emits backslashes literally, while POSIX `shlex.split` treats them as
escapes. Two backslashes collapse to one, and a trailing backslash makes
shapemesh's own output unreadable with `ValueError: No closing quotation`.
Implement symmetric MSH-compliant name handling and test through both shapemesh
and Gmsh.

### M8. Declared counts and tag extrema are not consistently validated

**Status:** Open  
**References:** `src/shapemesh/mesh.py:381-393`, `:450-529`

An `$Elements` header declaring 99 elements but containing one is accepted, as
is a `$PhysicalNames` count of two with one record. Node/element totals and
extrema are partly ignored. Validate all declared counts and min/max tags
against actual records.

### M9. Malformed or unsupported input leaks unintended exceptions

**Status:** Open  
**References:** `src/shapemesh/mesh.py:319-323`, `:481-517`, `:769-777`

Truncated elements can raise `IndexError`; inconsistent connectivity rows can
raise NumPy's ragged-array `ValueError`; malformed integers can leak conversion
errors; and a real binary MSH file can raise `UnicodeDecodeError` before the
intended binary-format rejection. Translate these to contextual
`MeshFormatError` and inspect the ASCII format header before decoding a binary
payload.

### M10. Valid zero-element blocks break reading, writing, and NumPy conversion

**Status:** Open  
**References:** `src/shapemesh/mesh.py:282-285`, `:505-517`, `:735-750`,
`:868-876`

A Gmsh-accepted empty `POINT1` block fails during reading. Empty blocks also
break writer tag extrema, produce `(0,)` instead of `(0,k)` NumPy connectivity,
and break the NumPy block repr. Preserve known connectivity widths for empty
arrays and handle empty tag extrema explicitly.

### M11. Required writer-side validation is missing

**Status:** Open  
**References:** `src/shapemesh/mesh.py:822-834`, `:858-863`, `:902-918`

A physical group referencing an entity without an `EntityRecord` is silently
written and reread with empty membership instead of raising `ValueError`.
Runtime element code 999 is also emitted and then skipped by shapemesh's own
reader. Validate entity memberships, membership dimensions, connectivity
consistency, and supported element types before writing.

## Repository setup, licensing, and structure findings

### M12. The typed distribution lacks its PEP 561 marker

**Status:** Open  
**Reference:** `pyproject.toml:23`; missing `src/shapemesh/py.typed`

The package declares `Typing :: Typed`, but neither the source tree nor the
fresh wheel contains `py.typed`. Add the marker and verify it in the built
wheel.

### M13. Three examples violate the mandatory source-header policy

**Status:** Open  
**References:** `AGENTS.md:18-38`; `examples/beam_with_hole.py:1`,
`examples/frame.py:1`, `examples/pilecap.py:1`

All three examples begin with a module docstring instead of the required LGPL
notice. The seven package modules and seven test modules had the required
notice.

### M14. The specified CI matrix is absent

**Status:** Open release gate; intentionally deferred by the design until after
the feature-branch merge  
**Reference:** `design-v1.md:596-608`

No CI workflow is tracked. Before tagging, run the specified Python 3.12-3.15
matrix on Linux, macOS, and Windows with Gmsh 4.15.2. Native Gmsh behavior makes
the platform coverage particularly important.

### M15. The stated LGPL licensing goal is misleading with mandatory Gmsh

**Status:** Open project/licensing question  
**References:** `design-v1.md:46`, `README.md:43`

The design says LGPL was chosen without blocking non-GPL downstream users.
That is not a safe general claim when the required backend is Gmsh, which is
GPL-2.0-or-later. Gmsh's official licensing policy requires a different or
commercial license for integration into distributed closed-source software.
Clarify downstream obligations and decide whether that goal is actually
supported. This finding does not challenge the LGPL licensing of shapemesh's
own code.

### L1. The design source of truth and release documentation are stale

**Status:** Open  
**References:** `AGENTS.md:7-8`; `design-v1.md:3`, `:70`, `:629-630`,
`:637-646`; `README.md:6`, `:32-33`

The design still says no implementation exists, lists the old single-`LICENSE`
layout, calls the implemented cylinder signature provisional, and retains
implementation as future work. The README says examples are coming soon and
must lose its development status for the release.

### L2. `COPYING` omits the canonical GPL application appendix

**Status:** Open  
**Reference:** `COPYING:621`

All operative sections 0-17 are present, but the canonical 53-line "How to
Apply These Terms to Your New Programs" appendix is absent. Restore the full
verbatim FSF text for license-file completeness. This is distinct from H3's
complete omission of the GPL text in the stale artifacts.

## Lower-priority API findings

### L3. One-sided node classification leaves an array writable

**Status:** Open  
**Reference:** `src/shapemesh/mesh.py:579-588`

If only one of `entity_dim` and `entity_tag` is supplied, it is not frozen and
can be mutated through the public array. Freeze each present array
independently and preferably reject mismatched classification-array state.

### L4. `BuildError.log` omits Gmsh's last-error diagnostic

**Status:** Open  
**Reference:** `src/shapemesh/builder.py:148-152`

The log can be empty even when `gmsh.logger.getLastError()` contains the actual
failure. Include that diagnostic before model removal or finalization.

### L5. Container boundaries are inconsistent with the declared API

**Status:** Open  
**References:** `src/shapemesh/shapes.py:189-197`, `:469-472`

Undeclared sets and generators are accepted as coordinates, while
`Polyline(None)` and `Polygon(None)` leak raw `TypeError`. Validate outer
containers and translate invalid input to `GeometryError`, or deliberately
widen the declared API.

### L6. Standard dataclass replacement is broken for booleans

**Status:** Open  
**Reference:** `src/shapemesh/shapes.py:817-826`

`dataclasses.replace(Union(a, b), operands=(a, a))` fails because the custom
varargs constructor cannot reconstruct the `operands` field. Support dataclass
reconstruction or explicitly document that replacement is unsupported.

### L7. NumPy result types are omitted from top-level exports

**Status:** Open API decision  
**References:** `src/shapemesh/mesh.py:38-48`;
`src/shapemesh/__init__.py:30-38`, `:63-98`

`NumpyMesh` and `NumpyElementBlock` are public in `shapemesh.mesh` but are not
re-exported with the other mesh data-model types. Re-export them or make the
intended narrower API explicit.

### L8. Some public docstrings violate the repository contract

**Status:** Open  
**References:** `src/shapemesh/shapes.py:142-180`, `:270-272`, `:828-830`

Examples include missing `Raises:` sections on `Vec3` and `Vec3.from_any()` and
missing docstrings on concrete `dimension` properties. Audit all public
classes, functions, methods, and properties against `AGENTS.md`.

## Important missing tests

1. Add the design section 9.5 overlapping-label round-trip: two overlapping
   labeled boxes, shapemesh write, Gmsh reopen, then shapemesh reopen. Verify
   both memberships, multiple physical tags, connectivity, coordinates, and
   signed boundaries.
2. Strengthen existing round-trips, which mostly compare counts rather than
   full mesh and entity fidelity.
3. Add regressions for every corrected finding, including higher-order
   generation, option isolation and conflicts, `rx < ry`, empty blocks,
   malformed input, and numeric boundaries.
4. Complete boolean volume assertions, true n-ary geometric semantics, and
   labeled-primitive coverage required by the design. Targeted checks during
   the review produced the expected union/intersection/difference volumes, but
   those results are not protected by the suite.

## Open questions before freezing the API

1. `design-v1.md:183` specifies `ValueError` for settings validation, while
   `:381` says `order="3"` raises `TypeError`. Code and tests currently use
   `ValueError`. Resolve the contradiction explicitly.
2. `pyproject.toml:26` permits Gmsh 5.x through `gmsh>=4.15`, while
   `docs/version-policy.md:3-7` only claims support for newer 4.x. Either use a
   `<5` bound or broaden and validate the policy.
3. Decide whether top-level NumPy result exports and dataclass replacement for
   booleans are supported API, as noted in L6 and L7.

## Verification during the original review

- Source suite: 70 passed.
- Fresh built-wheel suite: 70 passed.
- Warnings-as-errors suite: 70 passed.
- Ruff lint and formatting: passed.
- Pyright through `uv`: 0 errors.
- `uv lock --check`: passed.
- Offline wheel/sdist build: passed.
- All three examples ran successfully.
- Runtime verification used Windows, Python 3.13.2, Gmsh 4.15.2, and NumPy
  2.5.4. Other support-matrix combinations remain unverified.

## Remediation log

- 2026-10-25: Created `fix/review-findings` from `feat/shapes-settings` at
  `cde6dc0`.
- 2026-10-25: The user deleted the stale `dist/` artifacts (H3).
- 2026-10-25: Persisted this review and started H1/H2 remediation.
- 2026-10-25: Fixed H1 by setting `Mesh.ElementOrder` before generation and
  added an order-3 box regression covering point, curve, surface, and volume
  element types. Corrected design section 10 accordingly.
- 2026-10-25: Fixed H2 by rejecting every nonzero parametric flag in `$Nodes`
  blocks with `MeshFormatError`; added the previously corrupting curve block as
  a regression.
- 2026-10-25: H1/H2 verification: 72 tests passed, Ruff lint and formatting
  passed, and Pyright reported 0 errors. An order-3 box also retained
  `LINE4`/`TRI10`/`TET20` with default optimization enabled.
- 2026-10-25: Fixed M1 by passing the larger ellipse radius to OCC first and
  selecting a y major axis when `ry > rx`. Added curve and disk regressions with
  a nonzero center that verify the resulting x/y/z bounding extents. Verification:
  74 tests passed, Ruff lint and formatting passed, and Pyright reported 0 errors.
- 2026-10-25: Fixed M2 by centralizing configured Gmsh options and restoring them
  in preinitialized sessions after successful and failed builds. Added numeric and
  string option regressions, including a subsequent default build.
- 2026-10-25: Fixed M3 by reserving `General.Terminal` and `Mesh.ElementOrder` in
  addition to the algorithm options; added conflict tests for all four keys.
- 2026-10-25: Fixed M4 by skipping high-order optimization for 0D models and
  added a point-only order-2 regression. Updated the design and threading docs.
  M2-M4 verification: 80 tests passed, Ruff lint and formatting passed, and
  Pyright reported 0 errors.
- 2026-10-25: Fixed M5 by converting numeric inputs to float before finiteness
  and positivity checks, translating conversion failures to `GeometryError` or
  `ValueError` as appropriate. Added underflow and overflow regressions and
  verified that ordinary `Fraction` inputs remain valid.
- 2026-10-25: Fixed M6 by requiring every polygon vertex to be distinct and
  added a repeated non-consecutive-vertex regression. Clarified the design rule.
  M5-M6 verification: 86 tests passed, Ruff lint and formatting passed, and
  Pyright reported 0 errors.
