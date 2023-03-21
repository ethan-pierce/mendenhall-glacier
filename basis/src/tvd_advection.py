"""Total Variation Diminishing scheme for advection problems.

Provides utility functions to implement a total variation diminishing (TVD)
scheme for solving both homogenous and inhomogenous advection problems on a
Landlab grid. The spatial terms are cast as fluxes between nodes, and then
approximated as a weighted average of an upwind differencing method and a
Lax-Wendroff method, based on the value of a Van Leer flux limiter. 
This then allows us to use a forward Euler time-stepping algorithm with a 
standard CFL condition for the time discretization.

"""
import numpy as np
from landlab import RasterModelGrid

class AdvectTVD:
    """Implement the TVD advection scheme."""

    def __init__(self, grid, field: str):
        """Initialize the advector with a Landlab grid and indicate the field to advect."""
        self._grid = grid

        if field in self._grid.at_node.keys():
            self._field = field
        else:
            raise ValueError("Input field " + str(field) + " not found at grid nodes.")

    