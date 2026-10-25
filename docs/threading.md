# Thread safety

`shapemesh` wraps every build in a process-wide lock, but Gmsh itself has one
hard constraint: **initialization must happen on the main thread**.

## The rule

- `build()` must be called from the **main thread** to initialize Gmsh, because
  Gmsh installs a SIGINT handler (so Ctrl-C can stop meshing gracefully), and
  Python only permits signal handlers on the main thread.
- Calling `build()` from a worker thread *before* Gmsh is initialized raises a
  clear `RuntimeError`.
- **Once Gmsh is initialized on the main thread**, worker threads may call
  `build()`; the process-wide lock serializes them.

## Parallel meshing

Use `multiprocessing`, not threads, for parallel meshing — each process has its
own main thread and its own Gmsh session.

## Lifecycle

- Gmsh is imported lazily inside `build()`; importing `shapemesh` does not
  initialize Gmsh.
- `shapemesh` finalizes Gmsh only if it initialized it.
- When a build uses an existing Gmsh session, every global option touched by the
  build is restored during cleanup, including after failures.
- A new model is created per build and removed before release.
