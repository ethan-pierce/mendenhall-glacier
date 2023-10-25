import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tomli
from landlab.plot import imshow_grid
from basis.src.basis import BasalIceStratigrapher

plt.rcParams.update({'font.size': 14})

Ns = [60, 65, 70, 75, 80, 85, 90, 95]
boxplots = {
    'slow': {'fringe': [], 'dispersed': []},
    'fast': {'fringe': [], 'dispersed': []}
}

fluxes = {
    'slow': {'fringe': [], 'dispersed': []},
    'fast': {'fringe': [], 'dispersed': []}
}

for scenario in ['slow', 'fast']:
    model = BasalIceStratigrapher()
    model.initialize('./experiments/static-effective-pressure/' + scenario + '_input_file.toml')

    dx = model.grid.dx
    dy = model.grid.dy
    bounds = [50 * dx, 100 * dx, 0 * dy, 35 * dy]
    model.identify_terminus(bounds, depth = 3)

    idx = np.ravel(np.argwhere(model.grid.at_node['adjacent_to_terminus'] | model.grid.at_node['is_terminus']))

    for N in Ns:
        fringe_flux = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/flux/fringe_Pw_' + str(N) + '.txt')
        dispersed_flux = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/flux/disp_Pw_' + str(N) + '.txt')

        fluxes[scenario]['fringe'].append(fringe_flux[-1])
        fluxes[scenario]['dispersed'].append(dispersed_flux[-1])

        fringe_2d = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/fringe_Pw_' + str(N) + '.txt')
        dispersed_2d = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/disp_Pw_' + str(N) + '.txt')

        boxplots[scenario]['fringe'].append(fringe_2d[idx][fringe_2d[idx] != 1e-6])
        boxplots[scenario]['dispersed'].append(dispersed_2d[idx])

# # Frozen fringe fluxes
# fig, ax = plt.subplots(figsize = (12 , 6))

# ax.scatter(Ns, fluxes['slow']['fringe'], label = 'SLOW scenario')
# ax.plot(Ns, fluxes['slow']['fringe'], linestyle = ':', color = 'C0')

# ax.scatter(Ns, fluxes['fast']['fringe'], label = 'FAST scenario')
# ax.plot(Ns, fluxes['fast']['fringe'], linestyle = ':', color = 'C1')

# ax.fill_between(Ns, fluxes['slow']['fringe'], np.array(fluxes['slow']['fringe']) + np.array(fluxes['slow']['dispersed']), alpha = 0.25, linestyle = ':-', color = 'C0')
# ax.fill_between(Ns, fluxes['fast']['fringe'], np.array(fluxes['fast']['fringe']) + np.array(fluxes['fast']['dispersed']), alpha = 0.25, linestyle = ':-', color = 'C1')

# plt.legend()
# plt.show()

# # Dispersed layer fluxes
# fig, ax = plt.subplots(figsize = (12 , 6))

# ax.scatter(Ns, fluxes['slow']['dispersed'], label = 'SLOW scenario')
# ax.plot(Ns, fluxes['slow']['dispersed'], linestyle = ':', color = 'C0')

# ax.scatter(Ns, fluxes['fast']['dispersed'], label = 'FAST scenario')
# ax.plot(Ns, fluxes['fast']['dispersed'], linestyle = ':', color = 'C1')

# plt.legend()
# plt.show()

# Layer thickness boxplots
fig, ax = plt.subplots(figsize = (12, 6))

fplot = ax.boxplot(
    boxplots['slow']['fringe'], 
    positions = np.arange(len(Ns)), 
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
    boxplots['slow']['dispersed'],
    positions = np.arange(len(Ns)) + 0.425,
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

ax.set_xticks(np.arange(len(Ns)) + 0.2)
ax.set_xticklabels([str(100 - N) for N in Ns])

ax.set_xlabel('Effective pressure scenario (% of overburden)')
ax.set_ylabel('Layer thickness (m)')
plt.title('Modeled layer thicknesses at the terminus')

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='royalblue', label='Frozen fringe'),
                   plt.Rectangle((0, 0), 1, 1, facecolor='navajowhite', edgecolor='orange', label='Dispersed layer')]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()

# Layer thickness boxplots
fig, ax = plt.subplots(figsize = (12, 6))

fplot = ax.boxplot(
    boxplots['fast']['fringe'], 
    positions = np.arange(len(Ns)), 
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
    boxplots['fast']['dispersed'],
    positions = np.arange(len(Ns)) + 0.425,
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

ax.set_xticks(np.arange(len(Ns)) + 0.2)
ax.set_xticklabels([str(100 - N) for N in Ns])

ax.set_xlabel('Effective pressure scenario (% of overburden)')
ax.set_ylabel('Layer thickness (m)')
plt.title('Modeled layer thicknesses at the terminus')

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='royalblue', label='Frozen fringe'),
                   plt.Rectangle((0, 0), 1, 1, facecolor='navajowhite', edgecolor='orange', label='Dispersed layer')]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()
