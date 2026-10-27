# Gmsh version policy

- `shapemesh` requires **Gmsh 4.0 or later**.
- It is tested against **Gmsh 4.15.2**.
- Newer 4.x releases are expected to work.
- The supported element types are a static, curated enum. A Gmsh release that
  introduces a **new element type** requires a `shapemesh` update; unknown types
  are tolerated on read (skipped) but rejected by the writer.
