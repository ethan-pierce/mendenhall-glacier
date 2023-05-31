import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({'font.size': 22})

working_dir = './experiments/sensitivity/'
df = pd.read_csv(working_dir + 'outputs/results.csv')

vars_list = [
    'effective_pressure',
    'sliding_velocity_x',
    'basal_shear_stress',
    'geothermal_heat_flux',
    'till_grain_radius',
    'pore_throat_radius',
    'critical_depth',
    'cluster_volume_fraction',
    'friction_angle',
    'sediment_thermal_conductivity',
    'frozen_fringe_porosity',
    'permeability'
]
defaults = {
    'effective_pressure': 100,
    'sliding_velocity_x': 50,
    'basal_shear_stress': 56.3,
    'geothermal_heat_flux': 0.06,
    'till_grain_radius': 40,
    'pore_throat_radius': 1,
    'critical_depth': 10.0,
    'cluster_volume_fraction': 0.64,
    'friction_angle': 32,
    'sediment_thermal_conductivity': 6.27,
    'frozen_fringe_porosity': 0.4,
    'permeability': 4.1e-17
}
scalars = {
    'effective_pressure': 1e-3,
    'sliding_velocity_x': 1,
    'basal_shear_stress': 1e-3,
    'geothermal_heat_flux': 1,
    'till_grain_radius': 1e6,
    'pore_throat_radius': 1e6,
    'critical_depth': 1,
    'cluster_volume_fraction': 1,
    'friction_angle': 1,
    'sediment_thermal_conductivity': 1,
    'frozen_fringe_porosity': 1,
    'permeability': 1
}
names = {
    'effective_pressure': 'Effective pressure',
    'sliding_velocity_x': 'Sliding velocity',
    'basal_shear_stress': 'Basal shear stress',
    'geothermal_heat_flux': 'Geothermal heat flux',
    'till_grain_radius': 'Till grain radius',
    'pore_throat_radius': 'Pore throat radius',
    'critical_depth': 'Critical depth',
    'cluster_volume_fraction': 'Cluster volume fraction',
    'friction_angle': 'Till friction angle',
    'sediment_thermal_conductivity': 'Thermal conductivity',
    'frozen_fringe_porosity': 'Fringe porosity',
    'permeability': 'Fringe permeability'
}
units = {
    'effective_pressure': '(kPa)',
    'sliding_velocity_x': '(m a$^{-1}$)',
    'basal_shear_stress': '(kPa)',
    'geothermal_heat_flux': '(W m$^{-2}$)',
    'till_grain_radius': '($\mu$m)',
    'pore_throat_radius': '($\mu$m)',
    'critical_depth': '(m)',
    'cluster_volume_fraction': '',
    'friction_angle': '(degrees)',
    'sediment_thermal_conductivity': '(W m$^{-1}$ K$^{-1}$)',
    'frozen_fringe_porosity': '',
    'permeability': '(m$^2$)'
}
logs = ['sliding_velocity_x', 'till_grain_radius', 'pore_throat_radius']

colors = ['tab:blue', 'tab:orange']

fig, ax = plt.subplots(3, 4, figsize = (32, 16))
plt.suptitle('Sensitivity experiments', fontsize = 36)

for i in range(len(vars_list)):
    var = vars_list[i]
    default = defaults[var]
    scalar = scalars[var]
    name = names[var]
    unit = units[var]

    axis = np.ravel(ax)[i]
    axis2 = axis.twinx()

    axis.set_title(name)
    axis.set_xlabel(name + ' ' + unit)
    axis.set_ylabel('Fringe flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0], fontsize = 22)
    axis2.set_ylabel('Dispersed flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1], fontsize = 22)

    axis.tick_params(axis = 'y', colors = colors[0])
    axis2.tick_params(axis = 'y', colors = colors[1])
    axis2.spines['left'].set_color(colors[0])
    axis2.spines['right'].set_color(colors[1])

    axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axis2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if var in logs:
        axis.set_xscale('log')
        axis2.set_xscale('log')

    if var == 'permeability':
        axis.xaxis.set_label_coords(0.35, -0.125)

    axis.plot(
        df[df.variable == var].value * scalar,
        df[df.variable == var].fringe_sedflux * 1e-3,
        color = colors[0]
    )
    axis.scatter(default, df[df.variable == 'default'].fringe_sedflux * 1e-3, color = colors[0], s = 50)

    axis2.plot(
        df[df.variable == var].value * scalar,
        df[df.variable == var].dispersed_sedflux * 1e-3,
        color = colors[1]
    )
    axis2.scatter(default, df[df.variable == 'default'].dispersed_sedflux * 1e-3, color = colors[1], s = 50)

plt.tight_layout()
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.3)
plt.savefig(working_dir + '/outputs/sensitivity.png')
