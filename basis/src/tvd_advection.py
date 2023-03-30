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

    def __init__(self, grid, field: str, velocity: str):
        """Initialize the advector with a Landlab grid and indicate the field to advect."""
        self._grid = grid
        self.time_elapsed = 0.0

        if field in self._grid.at_node.keys():
            self._field = field
        else:
            raise ValueError("Input field " + str(field) + " not found at grid nodes.")

        if velocity in self._grid.at_link.keys():
            self._vel = velocity
        else:
            raise ValueError("Velocity field " + str(velocity) + " not found at grid links.")

        self._parallel_links = self.set_parallel_links_at_link(self._grid)
        self._upwind_links = self.find_upwind_link_at_link(self._grid, self._vel)

        if str(field) + '_flux' not in self._grid.at_link.keys():
            self._grid.add_zeros(str(field) + '_flux', at = 'link')
        if str(field) + '_flux_div' not in self._grid.at_node.keys():
            self._grid.add_zeros(str(field) + '_flux_div', at = 'node')

    def set_parallel_links_at_link(self, grid: RasterModelGrid) -> np.ndarray:
        """Map each link to its neighboring links that are parallel and directly adjacent."""
        parallel_links = -np.ones((grid.number_of_links, 2), dtype=int)

        parallel_links[grid.vertical_links, 0] = (
            grid.links_at_node[grid.node_at_link_tail[grid.vertical_links], 3]
        )
        parallel_links[grid.vertical_links, 1] = (
            grid.links_at_node[grid.node_at_link_head[grid.vertical_links], 1]
        )
        parallel_links[grid.horizontal_links, 0] = (
            grid.links_at_node[grid.node_at_link_tail[grid.horizontal_links], 2]
        )
        parallel_links[grid.horizontal_links, 1] = (
            grid.links_at_node[grid.node_at_link_head[grid.horizontal_links], 0]
        )

        boundary_tail = grid.status_at_node[grid.node_at_link_tail] != grid.BC_NODE_IS_CORE
        parallel_links[boundary_tail, 0] = -1
        boundary_head = grid.status_at_node[grid.node_at_link_head] != grid.BC_NODE_IS_CORE
        parallel_links[boundary_head, 1] = -1

        return parallel_links

    def find_upwind_link_at_link(self, grid: RasterModelGrid, velocity: str) -> np.ndarray:
        """Identify extension of each link in the upwind direction."""
        upwind_links = np.zeros(grid.number_of_links, dtype = int)

        upwind_links[:] = np.where(
            grid.at_link[velocity][:] > 0,
            self._parallel_links[:, 0],
            self._parallel_links[:, 1]
        )

        return upwind_links

    def calc_courant(self, grid: RasterModelGrid, velocity: str, dt: float) -> np.ndarray:
        """Return the local Courant number at each link."""

        courant = np.zeros(grid.number_of_links)

        courant[grid.horizontal_links] = (
            grid.at_link[velocity][grid.horizontal_links] * dt 
            / grid.spacing[0]
        )
        
        courant[grid.vertical_links] = (
            grid.at_link[velocity][grid.vertical_links] * dt 
            / grid.spacing[1]
        )

        return courant

    def map_value_to_links_linear_upwind(self, grid: RasterModelGrid, field: str, velocity: str) -> np.ndarray:
        """Map node values to links with a linear upwind method."""
        value_at_links = np.zeros(grid.number_of_links)
        velocity_is_positive = grid.at_link[velocity] > 0

        value_at_links[velocity_is_positive] = (
            grid.at_node[field][grid.node_at_link_tail[velocity_is_positive]]
        )

        value_at_links[~velocity_is_positive] = (
            grid.at_node[field][grid.node_at_link_head[~velocity_is_positive]]
        )

        return value_at_links

    def map_value_to_links_lax_wendroff(self, grid: RasterModelGrid, field: str, velocity: str, dt: float) -> np.ndarray:
        """Map node values to links with a Lax-Wendroff method."""
        courant = self.calc_courant(grid, velocity, dt)

        value_at_links = 0.5 * (
            (1 + courant) * grid.at_node[field][grid.node_at_link_tail] +
            (1 - courant) * grid.at_node[field][grid.node_at_link_head]
        )

        return value_at_links

    def calc_upwind_to_local_grad_ratio(self, grid: RasterModelGrid, field: str, velocity: str, dt: float) -> np.ndarray:
        """Return the ratio of upwind to local gradients."""
        courant = self.calc_courant(grid, velocity, dt)
        heads = grid.node_at_link_head
        tails = grid.node_at_link_tail
        local_diff = grid.at_node[field][heads] - grid.at_node[field][tails]

        ratio = -np.ones(grid.number_of_links)

        well_defined = (self._upwind_links != -1) & (local_diff != 0.0)

        ratio[well_defined] = (
            local_diff[self._upwind_links[well_defined]] /
            local_diff[well_defined]
        )

        return ratio

    def calc_flux_limiter(self, ratio: np.ndarray) -> np.ndarray:
        """Return the value of the Van Leer flux limiter at links."""
        flux_limiter = (ratio + np.abs(ratio)) / (1.0 + np.abs(ratio))
        return flux_limiter

    def calc_rate_of_change(self, grid: RasterModelGrid, field: str, velocity: str, dt: float):
        """Return the rate of change due to advection at grid nodes."""
        low_value_at_links = self.map_value_to_links_linear_upwind(grid, field, velocity)
        high_value_at_links = self.map_value_to_links_lax_wendroff(grid, field, velocity, dt)
        ratio = self.calc_upwind_to_local_grad_ratio(grid, field, velocity, dt)
        flux_limiter = self.calc_flux_limiter(ratio)
        
        field_at_links = flux_limiter * high_value_at_links + (1.0 - flux_limiter) * low_value_at_links
        
        grid.at_link[str(field) + '_flux'][grid.active_links] = (
            grid.at_link[velocity][grid.active_links] * field_at_links[grid.active_links]
        )
        
        grid.at_node[str(field) + '_flux_div'][:] = grid.calc_flux_div_at_node(str(field) + '_flux')

        return grid.at_node[str(field) + '_flux_div']

    def update(self, dt: float):
        """Run one forward time step of (dt) seconds."""
        rate_of_change = self.calc_rate_of_change(self._grid, self._field, self._vel, dt)
        self._grid.at_node[self._field][:] -= rate_of_change[:] * dt
        self.time_elapsed += dt