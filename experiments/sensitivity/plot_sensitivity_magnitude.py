import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 24})
plt.style.use('tableau-colorblind10')

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
    'till_grain_radius': 45,
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

fbars = []
dbars = []

for var in vars_list:
    values = df[df.variable == var].value * scalars[var]
    fringeflux = df[df.variable == var].fringe_sedflux / 1e3
    dispflux = df[df.variable == var].dispersed_sedflux / 1e3

    idx = np.argmin(np.abs(values - defaults[var]))

    fringe_slope = np.gradient(fringeflux, values / np.array(values)[idx])[idx] 
    disp_slope = np.gradient(dispflux, values / np.array(values)[idx])[idx] 

    fbars.append(fringe_slope)
    dbars.append(disp_slope)


width = 0.6
xs = np.arange(len(vars_list))
colors = ['C4', 'C5']

fig, ax = plt.subplots(2, 1, figsize=(22, 24))

fidx = np.flip(np.argsort(fbars))
fringe_bars = ax[0].bar(xs, np.array(fbars)[fidx], width, label='Frozen fringe sensitivity', 
                        color=colors[0], edgecolor = 'darkblue', linewidth = 2)
ax[0].bar_label(fringe_bars, fmt='%.2f', padding=3, fontsize = 28)

didx= np.flip(np.argsort(dbars))
disp_bars = ax[1].bar(xs, np.array(dbars)[didx], width, label='Dispersed layer sensitivity', 
                      color=colors[1], edgecolor = 'orange', linewidth = 2)
ax[1].bar_label(disp_bars, fmt='%.2f', padding=3, fontsize = 28)

ax[0].axhline(0, color = 'k', linestyle = ':')
ax[1].axhline(0, color = 'k', linestyle = ':')

ax[0].set_xticks(xs)
ax[0].set_xticklabels(np.array(list(names.values()))[fidx], rotation=45, ha='right', fontsize = 28)

ax[1].set_xticks(xs)
ax[1].set_xticklabels(np.array(list(names.values()))[didx], rotation=45, ha='right', fontsize = 28)

ax[0].yaxis.grid(True, which = 'both', linestyle = ':', linewidth = 0.5, alpha = 0.7, color = 'gray')
ax[1].yaxis.grid(True, which = 'both', linestyle = ':', linewidth = 0.5, alpha = 0.7, color = 'gray')

ax[0].set_ylim([-50, 100])
ax[1].set_ylim([-13, 11])

ax[0].set_ylabel('Sediment flux (kg m$^{-1}$ a$^{-1}$)', fontsize = 30)
ax[1].set_ylabel('Sediment flux (kg m$^{-1}$ a$^{-1}$)', fontsize = 30)

ax[0].legend(loc = 'upper right')
ax[1].legend(loc = 'upper right')

for axis in ax:
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

plt.suptitle('Sensitivity Magnitude', fontsize = 40)
plt.subplots_adjust(top = 0.9)
plt.tight_layout()
plt.savefig('./figures/sensitivity_magnitude.png', dpi = 300)