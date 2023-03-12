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
                    self.grid.add_field(variable, data, at = 'node')
                except:
                    raise ValueError('Could not add ' + str(variable) + ' to grid. Check names, shapes, etc. and try again.')
            
            else:
                value = np.full(self.grid.shape, info['value'])
                self.grid.add_field(variable, value, at = 'node')

        for required in ['ice_thickness', 'sliding_velocity_x', 'sliding_velocity_y', 'basal_water_pressure']:
            if required not in self.grid.at_node.keys():
                raise AttributeError(required + ' missing at grid nodes.')

        self.grid.status_at_node[self.grid.at_node['ice_thickness'][:] <= 0.5] = NodeStatus.CLOSED

        self.grid.add_zeros('sliding_velocity_magnitude', at = 'node')
        self.grid.at_node['sliding_velocity_magnitude'][:] = np.abs(
            np.sqrt(self.grid.at_node['sliding_velocity_x'][:]**2 + self.grid.at_node['sliding_velocity_y'][:]**2 )
        )

        self.grid.add_zeros('effective_pressure', at = 'node')
        self.grid.add_zeros('basal_shear_stress', at = 'node')
        self.grid.add_zeros('erosion_rate', at = 'node')
        self.grid.add_zeros('basal_melt_rate', at = 'node')

    def set_value(self, var: str, value: np.ndarray):
        """Set the value of a variable on the model grid."""
        self.grid.at_node[var][:] = value

###################
# Model processes #
###################

    def calc_effective_pressure(self):
        """Calculate the effective pressure at the ice-bed interface."""
        rho = self.params['ice_density']
        g = self.params['gravity']
        H = self.grid.at_node['ice_thickness'][:]
        Pw = self.grid.at_node['basal_water_pressure'][:]

        Pi = rho * g * H
        self.grid.at_node['effective_pressure'][:] = Pi - Pw

    def calc_shear_stress(self):
        """Calculate the basal shear stress beneath the glacier (Zoet and Iverson, 2021)."""
        C = self.params['slip_law_coefficient']
        p = self.params['shear_exponent']
        theta = np.deg2rad(self.params['friction_angle'])
        N = self.grid.at_node['effective_pressure'][:]
        Us = self.grid.at_node['sliding_velocity_magnitude'][:]

        N_reg = np.where(N != 0, N, np.nan)
        Ut = C * N_reg
        tau_b = N * np.tan(theta) * np.float_power(Us / (Us + Ut), (1 / p))
        
        self.grid.at_node['basal_shear_stress'][:] = tau_b[:]

    def calc_erosion_rate(self):
        """Calculate the erosion rate beneath the glacier (Herman et al., 2021)."""
        Ks = self.params['erosion_coefficient']
        m = self.params['erosion_exponent']
        Us = self.grid.at_node['sliding_velocity_magnitude'][:]

        self.grid.at_node['erosion_rate'][:] = Ks * Us**m

    def calc_melt_rate(self):
        """Calculate the melt rate beneath the glacier."""
        rho = self.params['ice_density']
        L = self.params['ice_latent_heat']
        Us = self.grid.at_node['sliding_velocity_magnitude'][:]
        tau_b = self.grid.at_node['basal_shear_stress'][:]

        frictional_heat_flux = Us * tau_b
        geothermal_heat_flux = self.params['geothermal_heat_flux']

        self.grid.at_node['basal_melt_rate'][:] = (
            (frictional_heat_flux + geothermal_heat_flux) / (rho * L)
        )

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