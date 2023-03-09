"""Basal Ice Stratigraphy (BasIS) model.

Coupled model of erosion and sediment entrainment beneath an ice sheet or glacier, using the Landlab
framework. The model estimates erosion rates using a power law relationship for bulk erosion
(Herman et al., 2021). Sediment is entrained as part of a frozen fringe (Rempel et al., 2008) at the
ice-till interface, and is then transported within the ice column by vertical regelation
(Worster and Wettlaufer, 2006). Basal melt removes material at the slip interface, which is assumed to
sit below the frozen fringe layer.

Required input fields, defined on grid nodes:
    ice_thickness: the total thickness of glacier ice (m)
    sliding_velocity_x: the x-directed component of basal ice velocity (m / a)
    sliding_velocity_y: the y-directed component of basal ice velocity (m / a)
    basal_water_pressure: the water pressure beneath the glacier (Pa)

Example usage:
    basis = BasalIceStratigrapher()
    basis.initialize('my_config_file.toml')
    basis.run_one_step(0.1)
    basis.write_output('my_output_file.nc')

"""

import numpy as np
import tomli
from netCDF4 import Dataset
from landlab import RasterModelGrid, NodeStatus

class BasalIceStratigrapher:
    """Model subglacial sediment entrainment processes."""

##############
# Initialize #
##############

    def __init__(self):
        """Initialize the model with empty fields."""
        self.config = ''
        self.grid = None
        self.params = {}
        self.time_elapsed = 0.0

    def initialize(self, config: str):
        """Using a configuration file, construct a grid and populate input fields."""
        with open(config, 'rb') as f:
            cfg = tomli.load(f)

        self.grid = RasterModelGrid(cfg['grid']['shape'], cfg['grid']['spacing'])
        
        self.params = cfg['parameters']

        for variable, info in cfg['inputs'].items():

            if len(info['file']) > 0:
                data = Dataset(info['file'])
                field = data[info['varname']]

                if len(field.shape) == 3:
                    field = field[0]

                try:
                    grid.add_field(variable, data, at = 'node')
                except:
                    raise ValueError('Could not add ' + str(variable) + ' to grid. Check names, shapes, etc. and try again.')
            
            else:
                try:
                    value = np.full_like(self.grid.shape, info['value'])
                    grid.add_field(variable, value, at = 'node')
                except:
                    raise ValueError('Missing input file or scalar value for ' + str(variable) + '.')

        self.grid.status_at_node[self.grid.at_node['ice_thickness'][:] <= 0.5] = NodeStatus.CLOSED

    def set_value(self, var: str, value: np.ndarray):
        """Set the value of a variable on the model grid."""
        self.grid.at_node[var][:] = value

###################
# Model processes #
###################

    def calc_shear_stress(self):
        """Calculate the basal shear stress beneath the glacier."""
        pass

    def calc_erosion_rate(self):
        """Calculate the erosion rate beneath the glacier."""
        pass

    def calc_melt_rate(self):
        """Calculate the melt rate beneath the glacier."""
        pass

    def calc_thermal_gradients(self):
        """Calculate the thermal gradient through the frozen fringe layer."""
        pass 

    def calc_fringe_growth_rate(self):
        """Calculate the growth rate of the frozen fringe."""
        pass

    def calc_regelation_rate(self):
        """Calculate the vertical velocity of sediment particles above the fringe."""
        pass

##########
# Update #
##########

    def erode_bedrock(self, t: float):
        """Run one step, only eroding bedrock beneath the glacier."""
        pass

    def entrain_sediment(self, t: float):
        """Run one step, only entraining sediment (if available) in basal ice layers."""
        pass

    def advect_sediment(self, t: float):
        """Run one step, only advecting sediment packages via basal sliding."""
        pass

    def run_one_step(self, t: float):
        """Run one step with all process models."""
        pass

############
# Finalize #
############

    def write_output(self, path_to_file: str):
        """Write output to a netcdf file."""
        pass