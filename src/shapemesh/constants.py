# Copyright (C) 2026 Mads Kruse
#
# This file is part of shapemesh.
#
# shapemesh is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# shapemesh is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with shapemesh. If not, see <https://www.gnu.org/licenses/>.

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
