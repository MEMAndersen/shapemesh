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

**Status:** Fixed

**References:** `src/shapemesh/mesh.py:381-393`, `:822-826`;
`src/shapemesh/shapes.py:64-77`

The writer emits backslashes literally, while POSIX `shlex.split` treats them as
escapes. Two backslashes collapse to one, and a trailing backslash makes
shapemesh's own output unreadable with `ValueError: No closing quotation`.
Implement symmetric MSH-compliant name handling and test through both shapemesh
and Gmsh.

### M8. Declared counts and tag extrema are not consistently validated

**Status:** Fixed

**References:** `src/shapemesh/mesh.py:381-393`, `:450-529`

An `$Elements` header declaring 99 elements but containing one is accepted, as
is a `$PhysicalNames` count of two with one record. Node/element totals and
extrema are partly ignored. Validate all declared counts and min/max tags
against actual records.

### M9. Malformed or unsupported input leaks unintended exceptions

**Status:** Fixed

**References:** `src/shapemesh/mesh.py:319-323`, `:481-517`, `:769-777`

Truncated elements can raise `IndexError`; inconsistent connectivity rows can
raise NumPy's ragged-array `ValueError`; malformed integers can leak conversion
errors; and a real binary MSH file can raise `UnicodeDecodeError` before the
intended binary-format rejection. Translate these to contextual
`MeshFormatError` and inspect the ASCII format header before decoding a binary
payload.

### M10. Valid zero-element blocks break reading, writing, and NumPy conversion

**Status:** Fixed

**References:** `src/shapemesh/mesh.py:282-285`, `:505-517`, `:735-750`,
`:868-876`

A Gmsh-accepted empty `POINT1` block fails during reading. Empty blocks also
break writer tag extrema, produce `(0,)` instead of `(0,k)` NumPy connectivity,
and break the NumPy block repr. Preserve known connectivity widths for empty
arrays and handle empty tag extrema explicitly.

### M11. Required writer-side validation is missing

**Status:** Fixed

**References:** `src/shapemesh/mesh.py:822-834`, `:858-863`, `:902-918`

A physical group referencing an entity without an `EntityRecord` is silently
written and reread with empty membership instead of raising `ValueError`.
Runtime element code 999 is also emitted and then skipped by shapemesh's own
reader. Validate entity memberships, membership dimensions, connectivity
consistency, and supported element types before writing.

## Repository setup, licensing, and structure findings

### M12. The typed distribution lacks its PEP 561 marker

**Status:** Fixed

**Reference:** `pyproject.toml:23`; missing `src/shapemesh/py.typed`

The package declares `Typing :: Typed`, but neither the source tree nor the
fresh wheel contains `py.typed`. Add the marker and verify it in the built
wheel.

### M13. Three examples violate the mandatory source-header policy

**Status:** Fixed

**References:** `AGENTS.md:18-38`; `examples/beam_with_hole.py:1`,
`examples/frame.py:1`, `examples/pilecap.py:1`

All three examples begin with a module docstring instead of the required LGPL
notice. The seven package modules and seven test modules had the required
notice.

### M14. The specified CI matrix is absent

**Status:** Fixed

**References:** `.github/workflows/ci.yml`; `design-v1.md` §15

Added a GitHub Actions workflow that runs the suite on Python 3.12–3.15 across
Linux, macOS, and Windows with Gmsh 4.15.2 pinned by `uv.lock`. The workflow
also enforces the lockfile, Ruff lint and formatting, Pyright, warnings-as-errors
tests, and a package gate that builds both distributions, verifies their
contents and metadata, runs the suite against the installed wheel, and
smoke-tests the examples.

### M15. The stated LGPL licensing goal is misleading with mandatory Gmsh

**Status:** Fixed  
**References:** `design-v1.md:46`, `README.md:68-77`

The design said LGPL was chosen without blocking non-GPL downstream users.
That is not a safe general claim when the required backend is Gmsh, which is
GPL-2.0-or-later. Clarified the downstream obligations:
- Updated `design-v1.md:46` to note that the mandatory Gmsh backend is
  GPL-2.0-or-later and affects distributed downstream works.
- Added a dedicated Gmsh licensing note to `README.md` explaining that
  shapemesh itself is LGPL-3.0-or-later, while using it in a distributed
  application brings Gmsh's copyleft obligations into scope.

### L1. The design source of truth and release documentation are stale

**Status:** In progress  
**References:** `AGENTS.md:7-8`; `design-v1.md:3`, `:70`, `:629-630`,
`:637-646`; `README.md:6`, `:32-33`

The design still says no implementation exists, lists the old single-`LICENSE`
layout, calls the implemented cylinder signature provisional, and retains
implementation as future work. The README says examples are coming soon and
must lose its development status for the release.

**Done so far:**
- Updated `design-v1.md` status to "Implemented; source of truth for v0.1.0".
- Replaced `LICENSE` with `COPYING`/`COPYING.LESSER` in the package layout.
- Updated the Gmsh dependency and version-policy lines to `gmsh>=4.0`.
- Marked the cylinder signature and primitive signatures as implemented.
- Rewrote `README.md` with a release-ready structure and a TODO placeholder for
  the author's personal primer.

**Still to do:**
- Author fills in the personal primer in `README.md`.
- Final review of the README tone and accuracy before release.

### L2. `COPYING` omits the canonical GPL application appendix

**Status:** Fixed

**Reference:** `COPYING:621`

All operative sections 0-17 are present, but the canonical 53-line "How to
Apply These Terms to Your New Programs" appendix is absent. Restore the full
verbatim FSF text for license-file completeness. This is distinct from H3's
complete omission of the GPL text in the stale artifacts.

## Lower-priority API findings

### L3. One-sided node classification leaves an array writable

**Status:** Fixed

**Reference:** `src/shapemesh/mesh.py:579-588`

If only one of `entity_dim` and `entity_tag` is supplied, it is not frozen and
can be mutated through the public array. Freeze each present array
independently and preferably reject mismatched classification-array state.

### L4. `BuildError.log` omits Gmsh's last-error diagnostic

**Status:** Fixed

**Reference:** `src/shapemesh/builder.py:148-152`

The log can be empty even when `gmsh.logger.getLastError()` contains the actual
failure. Include that diagnostic before model removal or finalization.

### L5. Container boundaries are inconsistent with the declared API

**Status:** Fixed

**References:** `src/shapemesh/shapes.py:189-197`, `:469-472`

Undeclared sets and generators are accepted as coordinates, while
`Polyline(None)` and `Polygon(None)` leak raw `TypeError`. Validate outer
containers and translate invalid input to `GeometryError`, or deliberately
widen the declared API.

### L6. Standard dataclass replacement is broken for booleans

**Status:** Fixed

**Reference:** `src/shapemesh/shapes.py:817-826`

`dataclasses.replace(Union(a, b), operands=(a, a))` fails because the custom
varargs constructor cannot reconstruct the `operands` field. Support dataclass
reconstruction or explicitly document that replacement is unsupported.

### L7. NumPy result types are omitted from top-level exports

**Status:** Fixed

**References:** `src/shapemesh/mesh.py:38-48`;
`src/shapemesh/__init__.py:30-38`, `:63-98`

`NumpyMesh` and `NumpyElementBlock` are public in `shapemesh.mesh` but are not
re-exported with the other mesh data-model types. Re-export them or make the
intended narrower API explicit.

### L8. Some public docstrings violate the repository contract

**Status:** Fixed

**References:** `src/shapemesh/shapes.py:142-180`, `:270-272`, `:828-830`

Examples include missing `Raises:` sections on `Vec3` and `Vec3.from_any()` and
missing docstrings on concrete `dimension` properties. Audit all public
classes, functions, methods, and properties against `AGENTS.md`.

## Important missing tests

1. ~~Add the design section 9.5 overlapping-label round-trip: two overlapping
   labeled boxes, shapemesh write, Gmsh reopen, then shapemesh reopen. Verify
   both memberships, multiple physical tags, connectivity, coordinates, and
   signed boundaries.~~ Done.
2. ~~Strengthen existing round-trips, which mostly compare counts rather than
   full mesh and entity fidelity.~~ Done for the build-to-MSH, minimal-MSH, and
   Gmsh-generated round-trips.
3. ~~Add regressions for every corrected finding, including higher-order
   generation, option isolation and conflicts, `rx < ry`, empty blocks,
   malformed input, and numeric boundaries.~~ Done. Each H/M finding now has an
   explicit regression test or is covered by a dedicated test module:
   - H1: `test_order_controls_generated_element_types`
   - H2: `test_parametric_nodes_rejected`
   - M1: `test_ellipse_with_y_major_axis`
   - M2: `test_preinitialized_gmsh_options_are_restored`
   - M3: `test_advanced_options_cannot_override_explicit_settings`
   - M4: `test_point_only_high_order_meshing`
   - M5: `test_positive_shape_dimensions_validate_converted_float`,
     `test_element_size_validates_converted_float`
   - M6: `test_polygon_rejects_repeated_nonconsecutive_vertices`
   - M7: `test_physical_group_backslashes_roundtrip`
   - M8: `test_physical_names_count_validated`,
     `test_node_tag_summary_validated`,
     `test_element_counts_and_tag_summary_validated`
   - M9: `test_malformed_msh_raises_format_error`,
     `test_binary_msh_rejected_before_decoding`,
     `test_invalid_utf8_raises_format_error`
   - M10: `test_empty_element_block_roundtrip`
   - M11: `test_writer_rejects_invalid_group_membership`,
     `test_writer_rejects_unknown_element_type`,
     `test_writer_rejects_inconsistent_connectivity`
4. ~~Complete boolean volume assertions, true n-ary geometric semantics, and
   labeled-primitive coverage required by the design. Targeted checks during
   the review produced the expected union/intersection/difference volumes, but
   those results are not protected by the suite.~~ Done.

## Open questions before freezing the API

None remaining. The settings-validation exception type and the Gmsh version
bound were resolved in this block.

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
- 2026-10-25: Fixed M7 by parsing physical-name quotes without POSIX escape
  processing, preserving Gmsh's literal backslash behavior. Added single,
  repeated, and trailing backslash round-trips through shapemesh and Gmsh.
- 2026-10-25: Fixed M11 by validating group memberships, membership dimensions,
  element types, and tag/connectivity shapes before writing. Added invalid-group,
  unknown-type, and inconsistent-connectivity regressions. M7/M11 verification:
  94 tests passed, Ruff lint and formatting passed, and Pyright reported 0 errors.
- 2026-10-26: Fixed M8 by validating physical-name, node, and element counts,
  node/element tag ranges, and missing or extra section data.
- 2026-10-26: Fixed M9 by translating malformed fields, truncated and ragged
  elements, and invalid UTF-8 to `MeshFormatError`. Binary input is detected from
  its ASCII header before payload decoding; a Gmsh-generated binary file was also
  verified to raise the intended error.
- 2026-10-26: Fixed M10 by preserving `(0, k)` connectivity through reading,
  writing, and NumPy conversion and by returning zero tag extrema for empty
  blocks. Added an empty-block round-trip and repr regression. M8-M10
  verification: 111 tests passed, Ruff lint and formatting passed, and Pyright
  reported 0 errors.
- 2026-10-26: Fixed M12 by adding `src/shapemesh/py.typed`; verified the marker
  in freshly built wheel and sdist artifacts.
- 2026-10-26: Fixed M13 by adding two-line copyright/SPDX notices to all three
  examples and updating `AGENTS.md` to permit that short form explicitly for
  documentation-focused examples. Verified the headers in the sdist and reran
  all examples successfully.
- 2026-10-26: Fixed L2 by appending the canonical GPL application appendix.
  Verified that `COPYING` exactly matches the canonical 674-line FSF text and
  that both archives contain the completed license files. Verification for this
  block: 111 tests passed, Ruff lint and formatting passed, and Pyright reported
  0 errors.
- 2026-10-26: Fixed L3 by freezing `entity_dim` and `entity_tag` independently;
  added one-sided classification-array mutation regressions.
- 2026-10-26: Fixed L4 by including `gmsh.logger.getLastError()` in captured
  build diagnostics without duplicating existing log lines. Added a real
  invalid-option build regression.
- 2026-10-26: Fixed L5 by enforcing declared coordinate-sequence containers and
  normalizing polyline/polygon points through one helper. Added set, generator,
  and `None` regressions. L3-L5 verification: 117 tests passed, Ruff lint and
  formatting passed, and Pyright reported 0 errors.
- 2026-10-26: Fixed L6 by adding a keyword-only boolean `operands` path for
  `dataclasses.replace`, while preserving variadic and single-sequence forms and
  rejecting mixed positional/keyword operands. Added operand and label
  replacement regressions and updated the design.
- 2026-10-26: Fixed L7 by re-exporting `NumpyMesh` and `NumpyElementBlock` at
  package level and using those exports in NumPy-view tests.
- 2026-10-26: Fixed L8 by documenting `Vec3` exceptions and all 13 concrete
  `dimension` properties. The public-method docstring check now passes. L6-L8
  verification: 118 tests passed, Ruff lint and formatting passed, and Pyright
  reported 0 errors.
- 2026-10-27: Added the design §9.5 overlapping-group round-trip: build two
  overlapping labeled boxes, write MSH 4.1, reopen with Gmsh to verify three
  volume pieces and shared physical-group membership, then reopen with
  `Mesh.read_msh` and verify entity count, group intersection, element-block
  keys, and physical-group membership of the overlap entity.
- 2026-10-27: Added boolean geometric-semantics tests: union/intersection/
  difference volumes on overlapping boxes, box-minus-sphere volume ≈ 0.966, and
  a nested `(a - b) + c` union. Added labeled-primitive coverage for point,
  polyline, polygon, disk, sphere, and cylinder. Test-coverage verification:
  128 tests passed, Ruff lint and formatting passed, and Pyright reported 0
  errors.
- 2026-10-27: Resolved the settings-validation contradiction by updating
  `design-v1.md` to state that `order="3"` raises `ValueError` from
  `__post_init__`, matching code and tests. Broadened the Gmsh dependency to
  `gmsh>=4.0` in `pyproject.toml` and updated `docs/version-policy.md` to
  require 4.0+ while noting testing against 4.15.2. Strengthened MSH round-trip
  tests to compare node tags/coordinates and element-block keys, tags, and
  connectivity. Policy-and-roundtrip verification: 128 tests passed, Ruff lint
  and formatting passed, and Pyright reported 0 errors.
- 2026-10-27: Audited and closed test-gap item 3. Every H/M finding now has an
  explicit regression test mapping, documented in the release review.
- 2026-10-27: Fixed M14 by adding `.github/workflows/ci.yml` with the design §15
  matrix (Python 3.12–3.15 on Linux, macOS, and Windows), Gmsh 4.15.2 pinned
  through `uv.lock`, lockfile/Ruff/Pyright gates, warnings-as-errors tests, and a
  distribution build gate that checks archive contents and metadata, runs the
  suite against the installed wheel, and smoke-tests the examples. Added the
  Python 3.15 classifier and a CI badge. The first run on `main` passed all 14
  jobs: <https://github.com/MEMAndersen/shapemesh/actions/runs/40567585907>.
