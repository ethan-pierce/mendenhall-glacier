import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

working_dir = './experiments/sensitivity/'
df = pd.read_csv(working_dir + 'outputs/results.csv')

colors = ['tab:blue', 'tab:orange']

fig, ax = plt.subplots(3, 4, figsize = (32, 16))

# ax[0, 0].set_title('Effective pressure')
ax[0, 0].set_xlabel('$log$ Effective pressure (kPa)')
ax[0, 0].set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
ax[0, 0].tick_params(axis = 'y', colors = colors[0])
ax[0, 0].plot(
    df[df.variable == 'effective_pressure'].value / 1e3,
    df[df.variable == 'effective_pressure'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = ax[0, 0].twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'effective_pressure'].value / 1e3,
    df[df.variable == 'effective_pressure'].dispersed_sedflux / 1e3,
    color = colors[1]
)
ax[0, 0].scatter(100, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(100, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

# ax[0, 0].set_title('Sliding velocity (m a$^{-1}$)')
ax[0, 1].set_xlabel('$log$ Sliding velocity (m a$^{-1}$)')
ax[0, 1].set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
ax[0, 1].tick_params(axis = 'y', colors = colors[0])
ax[0, 1].set_xscale('log')
ax[0, 1].plot(
    df[df.variable == 'sliding_velocity_x'].value,
    df[df.variable == 'sliding_velocity_x'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = ax[0, 1].twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'sliding_velocity_x'].value,
    df[df.variable == 'sliding_velocity_x'].dispersed_sedflux / 1e3,
    color = colors[1]
)
ax[0, 1].scatter(50, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(50, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

# ax[0, 0].set_title('Basal shear stress (kPa)')
ax[0, 2].set_xlabel('Basal shear stress (kPa)')
ax[0, 2].set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
ax[0, 2].tick_params(axis = 'y', colors = colors[0])
ax[0, 2].plot(
    df[df.variable == 'basal_shear_stress'].value / 1e3,
    df[df.variable == 'basal_shear_stress'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = ax[0, 2].twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'basal_shear_stress'].value / 1e3,
    df[df.variable == 'basal_shear_stress'].dispersed_sedflux / 1e3,
    color = colors[1]
)
ax[0, 2].scatter(56.3, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(56.3, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

# ax[0, 0].set_title('Geothermal heat flux (W m${-2}$)')
ax[0, 3].set_xlabel('Geothermal heat flux (W m${-2}$)')
ax[0, 3].set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
ax[0, 3].tick_params(axis = 'y', colors = colors[0])
ax[0, 3].plot(
    df[df.variable == 'geothermal_heat_flux'].value,
    df[df.variable == 'geothermal_heat_flux'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = ax[0, 3].twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'geothermal_heat_flux'].value,
    df[df.variable == 'geothermal_heat_flux'].dispersed_sedflux / 1e3,
    color = colors[1]
)
ax[0, 3].scatter(0.06, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(0.06, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

label = '$log$ Till grain radius ($\mu m$)'
plot = ax[1, 0]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.set_xscale('log')
plot.plot(
    df[df.variable == 'till_grain_radius'].value * 1e6,
    df[df.variable == 'till_grain_radius'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'till_grain_radius'].value * 1e6,
    df[df.variable == 'till_grain_radius'].dispersed_sedflux / 1e3,
    color = colors[1],
    linestyle = ':',
    lw = 3
)
plot.scatter(4e-5 * 1e6, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(4e-5 * 1e6, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

label = '$log$ Pore throat radius ($\mu m$)'
plot = ax[1, 1]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.set_xscale('log')
plot.plot(
    df[df.variable == 'pore_throat_radius'].value * 1e6,
    df[df.variable == 'pore_throat_radius'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'pore_throat_radius'].value * 1e6,
    df[df.variable == 'pore_throat_radius'].dispersed_sedflux / 1e3,
    color = colors[1]
)
plot.scatter(1.0, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(1.0, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

label = '$log$ Coupled grain & pore throat radius ($\mu m$)'
plot = ax[1, 2]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.set_xscale('log')
plot.plot(
    df[df.variable == 'coupled_grain_size'].value * 1e6,
    df[df.variable == 'coupled_grain_size'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'coupled_grain_size'].value * 1e6,
    df[df.variable == 'coupled_grain_size'].dispersed_sedflux / 1e3,
    color = colors[1]
)

label = 'Cluster volume fraction'
plot = ax[1, 3]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.plot(
    df[df.variable == 'cluster_volume_fraction'].value,
    df[df.variable == 'cluster_volume_fraction'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'cluster_volume_fraction'].value,
    df[df.variable == 'cluster_volume_fraction'].dispersed_sedflux / 1e3,
    color = colors[1]
)
plot.scatter(0.64, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(0.64, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

label = 'Till friction angle (degrees)'
plot = ax[2, 0]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.plot(
    df[df.variable == 'friction_angle'].value,
    df[df.variable == 'friction_angle'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'friction_angle'].value,
    df[df.variable == 'friction_angle'].dispersed_sedflux / 1e3,
    color = colors[1]
)
plot.scatter(32, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(32, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

label = 'Sediment thermal conductivity (W $(mK)^{-1}$)'
plot = ax[2, 1]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.plot(
    df[df.variable == 'sediment_thermal_conductivity'].value,
    df[df.variable == 'sediment_thermal_conductivity'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'sediment_thermal_conductivity'].value,
    df[df.variable == 'sediment_thermal_conductivity'].dispersed_sedflux / 1e3,
    color = colors[1]
)
plot.scatter(6.27, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(6.27, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

label = 'Frozen fringe porosity'
plot = ax[2, 2]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.plot(
    df[df.variable == 'frozen_fringe_porosity'].value,
    df[df.variable == 'frozen_fringe_porosity'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'frozen_fringe_porosity'].value,
    df[df.variable == 'frozen_fringe_porosity'].dispersed_sedflux / 1e3,
    color = colors[1]
)
plot.scatter(0.4, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(0.4, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

label = 'Till permeability (m$^2$)'
plot = ax[2, 3]
plot.set_xlabel(label)
plot.set_ylabel('Fringe sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[0])
plot.tick_params(axis = 'y', colors = colors[0])
plot.plot(
    df[df.variable == 'permeability'].value,
    df[df.variable == 'permeability'].fringe_sedflux / 1e3,
    color = colors[0]
)

ax2 = plot.twinx()
ax2.set_ylabel('Dispersed sed. flux (kg m$^{-1}$ a$^{-1}$)', color = colors[1])
ax2.tick_params(axis = 'y', colors = colors[1])
ax2.spines['left'].set_color(colors[0])
ax2.spines['right'].set_color(colors[1])
ax2.plot(
    df[df.variable == 'permeability'].value,
    df[df.variable == 'permeability'].dispersed_sedflux / 1e3,
    color = colors[1],
    linestyle = ':',
    lw = 3
)
plot.scatter(4.1e-17, df[df.variable == 'default'].fringe_sedflux / 1e3, color = colors[0], s = 50)
ax2.scatter(4.1e-17, df[df.variable == 'default'].dispersed_sedflux / 1e3, color = colors[1], s = 50)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.3)
plt.savefig(working_dir + '/outputs/sensitivity.png')