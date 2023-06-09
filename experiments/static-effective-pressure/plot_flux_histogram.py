import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tomli

from basis.src.basis import BasalIceStratigrapher

plt.rcParams.update({'font.size': 14})

Ns = [60, 65, 70, 75, 80, 85, 90, 95]
fboxes = []
dboxes = []

ffluxes = []
dfluxes = []

for N in Ns:
    fringe = np.loadtxt('./experiments/static-effective-pressure/outputs/fringe_Pw_' + str(N) + '_pct.txt')
    disp = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_Pw_' + str(N) + '_pct.txt')

    BIS = BasalIceStratigrapher()
    BIS.initialize('./experiments/static-effective-pressure/input_file.toml')

    BIS.set_value('fringe_thickness', fringe)
    BIS.set_value('dispersed_layer_thickness', disp)

    dx = BIS.grid.dx
    dy = BIS.grid.dy
    bounds = [50 * dx, 100 * dx, 0 * dy, 15 * dy]
    BIS.identify_terminus(bounds, depth = 2)
    depth1 = BIS.east_boundary + BIS.north_boundary + BIS.west_boundary + BIS.south_boundary
    depth2 = BIS.adjacent_to_terminus

    fboxes.append(BIS.grid.at_node['fringe_thickness'][depth2])
    dboxes.append(BIS.grid.at_node['dispersed_layer_thickness'][depth2])

    # fflux = np.loadtxt('./experiments/static-effective-pressure/outputs/fringe_flux_Pw_' + str(N) + '_pct.txt')
    # dflux = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_flux_Pw_' + str(N) + '_pct.txt')

    fflux = np.sum(
        BIS.grid.at_node['fringe_thickness'][depth2]
        * BIS.grid.dx
        * BIS.grid.at_node['sliding_velocity_magnitude'][depth2]
        * 0.6
        * BIS.sec_per_a
    )

    dflux = np.sum(
        BIS.grid.at_node['dispersed_layer_thickness'][depth2]
        * BIS.grid.dx 
        * BIS.grid.at_node['sliding_velocity_magnitude'][depth2]
        * 0.05
        * BIS.sec_per_a
    )

    ffluxes.append(fflux)
    dfluxes.append(dflux)

print(np.array(ffluxes) + np.array(dfluxes))

# # Boxplot
fig, ax = plt.subplots(figsize = (12, 6))

fplot = ax.boxplot(
    fboxes, 
    positions = np.arange(len(fboxes)), 
    widths = 0.4, 
    bootstrap = 1000,
    patch_artist = True,
    boxprops = dict(
        facecolor = 'lightblue'
    ),
    medianprops = dict(
        color = 'royalblue',
        linewidth = 2
    )
)

dplot = ax.boxplot(
    dboxes,
    positions = np.arange(len(fboxes)) + 0.425,
    widths = 0.4,
    bootstrap = 1000,
    patch_artist = True,
    boxprops = dict(
        facecolor = 'navajowhite'
    ),
    medianprops = dict(
        color = 'orange',
        linewidth = 2
    )
)

ax.yaxis.grid(True, which = 'both', linestyle = ':', linewidth = 0.5, alpha = 0.7, color = 'gray')
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))

ax.set_xticks(np.arange(len(fboxes)) + 0.2)
ax.set_xticklabels([str(100 - N) for N in Ns])

ax.set_xlabel('Effective pressure scenario (% of overburden)')
ax.set_ylabel('Layer thickness (m)')
plt.title('Modeled layer thicknesses at the terminus')

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='royalblue', label='Frozen fringe'),
                   plt.Rectangle((0, 0), 1, 1, facecolor='navajowhite', edgecolor='orange', label='Dispersed layer')]
ax.legend(handles=legend_elements, loc='upper right')

plt.savefig('./figures/stratigraphy_boxplot.png', dpi = 300)
plt.close('all')



# Line plot
fig, ax = plt.subplots(figsize = (12, 6))
ax2 = ax.twinx()

blue = 'royalblue'

fplot = ax.plot(Ns, ffluxes, color = blue, label = 'Frozen fringe')
dplot = ax2.plot(Ns, dfluxes, color = 'orange', label = 'Dispersed layer')
sumplot = ax.plot(
    Ns, np.array(ffluxes) + np.array(dfluxes), 
    color = blue, label = 'Sum of layers',
    linestyle = ':', linewidth = 2.5
)

ax.set_xticks(Ns)
ax.set_xticklabels([str(100 - N) for N in Ns])
ax.set_xlabel('Effective pressure scenario (% of overburden)')
ax.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
ax2.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')

# ax.set_ylim([0, 56000])
ax2.set_ylim([1500, 4500])

ax2.spines['left'].set_color(blue)
ax.tick_params(axis='y', colors=blue)
ax2.spines['right'].set_color('orange')
ax2.tick_params(axis='y', colors='orange')

lines, labs = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
plt.legend(lines + lines2, labs + labs2, loc = 'best')

plt.title('Modeled sediment flux at the terminus')
plt.savefig('./figures/fluxes_per_scenario.png', dpi = 300)
