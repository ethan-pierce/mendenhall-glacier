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
# TODO plot dt at a single point
# TODO check fringe under icefalls


import numpy as np
import tomli
from netCDF4 import Dataset
from landlab import RasterModelGrid, NodeStatus
import matplotlib.pyplot as plt

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
        self.sec_per_a = 31556926

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

                self.grid.add_field(variable, field, at = 'node')
            
            else:
                value = np.full(self.grid.shape, info['value'])
                self.grid.add_field(variable, value, at = 'node')

        for required in ['ice_thickness', 'surface_elevation', 'sliding_velocity_x', 'sliding_velocity_y', 'basal_water_pressure']:
            if required not in self.grid.at_node.keys():
                raise AttributeError(required + ' missing at grid nodes.')

        self.grid.status_at_node[self.grid.at_node['ice_thickness'][:] <= 0.5] = NodeStatus.CLOSED
        self.grid.at_node['sliding_velocity_x'][:] *= 1 / self.sec_per_a
        self.grid.at_node['sliding_velocity_y'][:] *= 1 / self.sec_per_a

        self.grid.add_zeros('sliding_velocity_magnitude', at = 'node')
        self.grid.at_node['sliding_velocity_magnitude'][:] = np.abs(
            np.sqrt(self.grid.at_node['sliding_velocity_x'][:]**2 + self.grid.at_node['sliding_velocity_y'][:]**2 )
        )

        self.grid.add_zeros('effective_pressure', at = 'node')

        self.grid.add_zeros('basal_shear_stress', at = 'node')

        self.grid.add_zeros('erosion_rate', at = 'node')

        self.grid.add_zeros('basal_melt_rate', at = 'node')

        self.grid.add_zeros('frictional_heat_flux', at = 'node')
        self.grid.add_zeros('fringe_thermal_gradient', at = 'node')
        self.grid.add_zeros('transition_temperature', at = 'node')

        self.grid.add_zeros('fringe_undercooling', at = 'node')
        self.grid.add_zeros('fringe_saturation', at = 'node')
        self.grid.add_zeros('nominal_heave_rate', at = 'node')
        self.grid.add_zeros('flow_resistance', at = 'node')
        self.grid.add_zeros('fringe_heave_rate', at = 'node')
        self.grid.add_zeros('fringe_growth_rate', at = 'node')

        self.grid.add_zeros('dispersed_layer_gradient', at = 'node')
        self.grid.add_zeros('dispersed_layer_growth_rate', at = 'node')

        self.grid.add_zeros('till_thickness', at = 'node')
        self.grid.add_zeros('fringe_thickness', at = 'node')
        self.grid.add_zeros('dispersed_layer_thickness', at = 'node')
        self.grid.add_full('dispersed_concentration', self.params['initial_dispersed_concentration'], at = 'node')

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

        N_reg = np.where(
            self.grid.at_node['ice_thickness'] > 0.5, N, 1e-9
        )
        Ut = C * N_reg
        tau_b = N * np.tan(theta) * np.float_power(Us / (Us + Ut), (1 / p))
        
        self.grid.at_node['basal_shear_stress'][:] = tau_b[:]
        
    def calc_erosion_rate(self):
        """Calculate the erosion rate beneath the glacier (Herman et al., 2021)."""
        Ks = self.params['erosion_coefficient']
        m = self.params['erosion_exponent']
        Us = self.grid.at_node['sliding_velocity_magnitude'][:]

        self.grid.at_node['erosion_rate'][:] = (Ks * Us**m) / self.sec_per_a 

    def calc_melt_rate(self):
        """Calculate the melt rate beneath the glacier."""
        rho = self.params['ice_density']
        L = self.params['ice_latent_heat']
        Us = self.grid.at_node['sliding_velocity_magnitude'][:]
        tau_b = self.grid.at_node['basal_shear_stress'][:]

        frictional_heat_flux = Us * tau_b
        self.grid.at_node['frictional_heat_flux'][:] = frictional_heat_flux

        geothermal_heat_flux = self.params['geothermal_heat_flux']

        self.grid.at_node['basal_melt_rate'][:] = (
            (frictional_heat_flux + geothermal_heat_flux) / (rho * L)
        )

    def calc_thermal_gradients(self):
        """Calculate the thermal gradient through the frozen fringe layer."""
        gamma = self.params['surface_energy']
        rp = self.params['pore_throat_radius']
        self.params['entry_pressure'] = (2 * gamma) / rp

        pf = self.params['entry_pressure']
        Tm = self.params['melt_temperature']
        rho = self.params['ice_density']
        L = self.params['ice_latent_heat']
        self.params['fringe_base_temperature'] = Tm - ((pf * Tm) / (rho * L))

        ki = self.params['ice_thermal_conductivity']
        ks = self.params['sediment_thermal_conductivity']
        phi = self.params['frozen_fringe_porosity']
        self.params['fringe_conductivity'] = (1 - phi) * ks + phi * ki

        K = self.params['fringe_conductivity']
        Qg = self.params['geothermal_heat_flux']
        Qf = self.grid.at_node['frictional_heat_flux'][:]
        self.grid.at_node['fringe_thermal_gradient'][:] = -(Qg + Qf) / K

        G = self.grid.at_node['fringe_thermal_gradient'][:]
        hf = self.grid.at_node['fringe_thickness'][:]
        Tf = self.params['fringe_base_temperature']

        self.grid.at_node['transition_temperature'][:] = G * hf + Tf

    def calc_fringe_growth_rate(self):
        """Calculate the growth rate of the frozen fringe."""
        G = self.grid.at_node['fringe_thermal_gradient'][:]
        h = self.grid.at_node['fringe_thickness'][:]
        Tm = self.params['melt_temperature']
        Tf = self.params['fringe_base_temperature']
        self.grid.at_node['fringe_undercooling'][:] = 1 - ((G * h) / (Tm - Tf))

        alpha = self.params['fringe_alpha']
        beta = self.params['fringe_beta']
        theta = self.grid.at_node['fringe_undercooling'][:]
        self.grid.at_node['fringe_saturation'][:] = 1 - theta**(-beta)

        rho_w = self.params['water_density']
        rho_i = self.params['ice_density']
        L = self.params['ice_latent_heat']
        k0 = self.params['permeability']
        eta = self.params['water_viscosity']
        self.grid.at_node['nominal_heave_rate'][:] = -(rho_w**2 * L * G * k0) / (rho_i * Tm * eta)

        d = self.params['film_thickness']
        R = self.params['till_grain_radius']
        self.grid.at_node['flow_resistance'][:] = -(rho_w**2 * k0 * G * R**2) / (rho_i**2 * (Tm - Tf) * d**3)

        phi = self.params['frozen_fringe_porosity']
        Vs = self.grid.at_node['nominal_heave_rate'][:]
        Pi = self.grid.at_node['flow_resistance'][:]
        N = self.grid.at_node['effective_pressure'][:]
        pf = self.params['entry_pressure']

        # Throwaway variables for long coefficients
        A = theta + phi * (1 - theta + (1 / (1 - beta)) * (theta**(1 - beta) - 1))
        B = ((1 - phi)**2 / (alpha + 1)) * (theta**(alpha + 1) - 1)
        C = ((2 * (1 - phi) * phi) / (alpha - beta + 1)) * (theta**(alpha - beta + 1) - 1)
        D = (phi**2 / (alpha - 2 * beta + 1)) * (theta**(alpha - 2 * beta + 1) - 1)

        self.grid.at_node['fringe_heave_rate'][:] = Vs * (A - (N / pf)) / (B + C + D + Pi)
        V = self.grid.at_node['fringe_heave_rate'][:]
        m = self.grid.at_node['basal_melt_rate'][:]
        S = self.grid.at_node['fringe_saturation'][:]

        self.grid.at_node['fringe_growth_rate'] = np.where(
            S > 0,
            (-m - V) / (phi * S),
            0.0
        )

    def calc_regelation_rate(self):
        """Calculate the vertical velocity of sediment particles above the fringe."""
        Tm = self.params['melt_temperature']
        Tf = self.params['fringe_base_temperature']
        rho = self.params['ice_density']
        g = self.params['gravity']
        L = self.params['ice_latent_heat']
        r = self.params['till_grain_radius']
        z0 = self.params['critical_depth']
        gamma = self.params['ice_clapeyron_slope']
        theta = self.grid.at_node['fringe_undercooling'][:]

        # Meyer et al. 2018, eq (12)
        temp_at_top_of_fringe = Tm - (Tm - Tf) * theta

        # The temperature gradient depends on the supercooling at the top of the fringe and the pressure-melting-point
        G_premelting = (Tm - temp_at_top_of_fringe) / z0
        G_pressure_melting = gamma * rho * g
        G = np.maximum(G_premelting, G_pressure_melting)
        self.grid.at_node['dispersed_layer_gradient'][:] = G

        # Kozeny-Carmen equation for permeability
        phi = self.params['cluster_volume_fraction']
        K = (r**2 * (1 - phi)**3) / (45 * phi**2)

        mu = self.params['water_viscosity']
        ki = self.params['ice_thermal_conductivity']
        kp = self.params['sediment_thermal_conductivity']
        coeff = (K * rho * L) / (mu * Tm * (2 * ki + kp))

        # The dispersed layer thickness grows with the velocity of the fastest particles
        self.grid.at_node['dispersed_layer_growth_rate'][:] = (coeff * 3 * ki) / (1 + coeff * rho * L) * G

##########
# Update #
##########

    def erode_bedrock(self, dt: float):
        """Run one step, only eroding bedrock beneath the glacier."""
        self.calc_effective_pressure()
        self.calc_shear_stress()
        self.calc_erosion_rate()

        self.grid.at_node['till_thickness'][:] += self.grid.at_node['erosion_rate'][:] * dt

    def entrain_sediment(self, dt: float, already_eroded = False, clamp = True):
        """Run one step, only entraining sediment (if available) in basal ice layers."""
        if not already_eroded:
            self.calc_effective_pressure()
            self.calc_shear_stress()

        self.calc_melt_rate()
        self.calc_thermal_gradients()
        self.calc_fringe_growth_rate()
        self.calc_regelation_rate()

        Ht = self.grid.at_node['till_thickness'][:]
        Hf = self.grid.at_node['fringe_thickness'][:]
        Cf = self.params['frozen_fringe_porosity']
        Hd = self.grid.at_node['dispersed_layer_thickness'][:]

        fringe_max_dh = self.grid.at_node['fringe_growth_rate'] * dt
        fringe_dh = np.minimum(fringe_max_dh, Ht / Cf)
        dispersed_dh = self.grid.at_node['dispersed_layer_growth_rate'] * dt

        Ht[:] -= np.where(fringe_dh > 0, Cf * fringe_dh, 0)
        Hf[:] += fringe_dh[:]
        Hd[:] += dispersed_dh[:]

        Ht[:] = np.where(Ht < 0, 0, Ht)
        Hf[:] = np.where(Hf <= 1e-6, 1e-6, Hf)

        if clamp:
            Hf[:] = np.where(Hf > np.percentile(Hf, 99), np.percentile(Hf, 99), Hf)

    def advect_sediment(self, dt: float):
        """Run one step, only advecting sediment packages via basal sliding."""
        ux = self.grid.at_node['sliding_velocity_x'][:]
        uy = self.grid.at_node['sliding_velocity_y'][:]
        Hf = self.grid.at_node['fringe_thickness'][:]
        Hd = self.grid.at_node['dispersed_layer_thickness'][:]
        Cd = self.grid.at_node['dispersed_concentration'][:]
        
        ux_links = self.grid.map_value_at_max_node_to_link('surface_elevation', 'sliding_velocity_x'),
        uy_links = self.grid.map_value_at_max_node_to_link('surface_elevation', 'sliding_velocity_y'),
        Hf_links = self.grid.map_value_at_max_node_to_link('surface_elevation', 'fringe_thickness'),
        Hd_links = self.grid.map_value_at_max_node_to_link('surface_elevation', 'dispersed_layer_thickness')
        Cd_links = self.grid.map_value_at_max_node_to_link('surface_elevation', 'dispersed_concentration')

        ux_dx = self.grid.calc_diff_at_link(ux) / self.grid.dx
        uy_dy = self.grid.calc_diff_at_link(uy) / self.grid.dy

        Hf_dx = self.grid.calc_diff_at_link(Hf) / self.grid.dx
        Hf_dy = self.grid.calc_diff_at_link(Hf) / self.grid.dy

        Hd_dx = self.grid.calc_diff_at_link(Hd) / self.grid.dx
        Hd_dy = self.grid.calc_diff_at_link(Hd) / self.grid.dy

        Cd_dx = self.grid.calc_diff_at_link(Cd) / self.grid.dx
        Cd_dy = self.grid.calc_diff_at_link(Cd) / self.grid.dy

        advect_Hf = self.grid.map_mean_of_links_to_node(
            (Hf_links * ux_dx + Hf_links * uy_dy + ux_links * Hf_dx + uy_links * Hf_dy) * dt
        )

        advect_Hd = self.grid.map_mean_of_links_to_node(
            (Hd_links * ux_dx + Hd_links * uy_dy + ux_links * Hd_dx + uy_links * Hd_dy) * dt
        )

        advect_Cd = self.grid.map_mean_of_links_to_node(
            (Cd_links * ux_dx + Cd_links * uy_dy + ux_links * Cd_dx + uy_links * Cd_dy) * dt
        )

        Hf[:] -= advect_Hf[:]
        Hd[:] -= advect_Hd[:]
        Cd[:] -= advect_Cd[:]
        
        Hf[:] = np.where(Hf <= 0, 1e-6, Hf[:])
        Hd[:] = np.where(Hd <= 0, 0.0, Hd[:])
        Cd[:] = np.where(Cd <= 0, 1e-3, Cd[:])

    def run_one_step(self, dt: float, advect = True):
        """Run one step with all process models."""
        self.erode_bedrock(dt)
        self.entrain_sediment(dt)

        if advect:
            self.advect_sediment(dt)

        self.time_elapsed += dt

############
# Finalize #
############

    def write_output(self, path_to_file: str):
        """Write output to a netcdf file."""
        pass

    def plot_var(self, 
                 var: str, 
                 path_to_file: str = False,
                 scalar: float = 1.0,
                 units_label: str = 'None', 
                 imshow_args: dict = {}, 
                 cbar_args: dict = {}, 
                 savefig_args: dict = {}):
        """Plot a raster field and optionally save to a file."""
        field = np.where(
            self.grid.at_node['ice_thickness'][:] > 0.5,
            self.grid.at_node[var][:] * scalar,
            np.nan
        )
        field = np.reshape(field, [self.grid.shape[1], self.grid.shape[0]])

        im = plt.imshow(field, **imshow_args)
        plt.colorbar(im, **cbar_args)
        plt.title(var.replace('_', ' ') + ' (' + units_label + ')')

        if path_to_file:
            plt.savefig(path_to_file, **savefig_args)

        plt.close('all')