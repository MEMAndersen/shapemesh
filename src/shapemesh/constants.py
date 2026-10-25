"""Central numeric constants for shapemesh."""

# Polygon vertices are considered coplanar when their perpendicular distance
# from the fitted plane is below this fraction of the bounding-box diagonal.
PLANARITY_TOLERANCE = 1e-7

# Two polygon edges are considered non-collinear when the sine of the angle
# between them exceeds this value.
COLLINEARITY_TOLERANCE = 1e-7
