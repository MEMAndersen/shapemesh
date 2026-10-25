"""Central numeric constants for shapemesh."""

# Polygon vertices are considered coplanar when their perpendicular distance
# from the fitted plane is below this fraction of the bounding-box diagonal.
PLANARITY_TOLERANCE = 1e-7

# Two polygon edges are considered non-collinear when the sine of the angle
# between them exceeds this value.
COLLINEARITY_TOLERANCE = 1e-7

# `General.Terminal` is a gmsh verbosity level (0 = silent, higher = more
# verbose), not a binary toggle. shapemesh is silent by default and uses
# gmsh's normal level when `verbose` is set.
GMSH_TERMINAL_SILENT = 0
GMSH_TERMINAL_VERBOSE = 1
