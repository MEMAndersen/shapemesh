"""shapemesh — a Python geometry and meshing library powered by Gmsh.

See ``.plans/design-v1.md`` for the design. The public API is re-exported from
this package as the submodules are implemented; each submodule adds a
``from .module import ...`` here and extends ``__all__`` (design §5).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__: list[str] = []
