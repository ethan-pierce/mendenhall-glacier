import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmcrameri as cmc

import tomli
from landlab.plot import imshow_grid
from basis.src.basis import BasalIceStratigrapher

plt.rcParams.update({'font.size': 14})
plt.style.use('tableau-colorblind10')
plt.rcParams.update({'image.cmap': 'cmc.bilbaoS'})

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
    bounds = [50 * dx, 70 * dx, 0 * dy, 40 * dy]
    model.identify_terminus(bounds, depth = 2)

    idx = np.ravel(np.argwhere(model.grid.at_node['adjacent_to_terminus'] | model.grid.at_node['is_terminus']))

    for N in Ns:
        fringe_2d = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/fringe_Pw_' + str(N) + '.txt')
        dispersed_2d = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/disp_Pw_' + str(N) + '.txt')

        boxplots[scenario]['fringe'].append(fringe_2d[idx][fringe_2d[idx] != 1e-6])
        boxplots[scenario]['dispersed'].append(dispersed_2d[idx])

        ffluxes = fringe_2d[idx] * model.grid.dx * model.grid.at_node['sliding_velocity_magnitude'][idx] * model.sec_per_a * 0.6
        dfluxes = dispersed_2d[idx] * model.grid.dx * model.grid.at_node['sliding_velocity_magnitude'][idx] * model.sec_per_a * 0.05
        
        fluxes[scenario]['fringe'].append(ffluxes)
        fluxes[scenario]['dispersed'].append(dfluxes)

print(np.sum(fluxes['slow']['fringe'], axis = 1))
print(np.sum(fluxes['slow']['dispersed'], axis = 1))
print(np.sum(fluxes['fast']['fringe'], axis = 1))
print(np.sum(fluxes['fast']['dispersed'], axis = 1))


cf = 'C4'
cd = 'C5'

cslow = 'tab:purple'
cfast = 'tab:red'

Nlabels = [str(100 - i) for i in Ns]

# Frozen fringe fluxes
# fig, ax = plt.subplots(figsize = (12 , 6))

# ax.scatter(Nlabels, np.sum(fluxes['slow']['fringe'], axis = 1), label = 'SLOW scenario', color = cslow)
# ax.plot(Nlabels, np.sum(fluxes['slow']['fringe'], axis = 1), linestyle = ':', color = cslow)

# ax.scatter(Nlabels, np.sum(fluxes['fast']['fringe'], axis = 1), label = 'FAST scenario', color = cfast)
# ax.plot(Nlabels, np.sum(fluxes['fast']['fringe'], axis = 1), linestyle = ':', color = cfast)

# ax.set_xlabel('Effective pressure (% of overburden)')
# ax.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
# plt.title('Frozen fringe flux', size = 18)
# plt.legend()
# plt.savefig('./figures/fringe_fluxes.png', dpi = 300)

# # Dispersed layer fluxes
# fig, ax = plt.subplots(figsize = (12 , 6))

# ax.scatter(Nlabels, np.sum(fluxes['slow']['dispersed'], axis = 1), label = 'SLOW scenario', color = cslow)
# ax.plot(Nlabels, np.sum(fluxes['slow']['dispersed'], axis = 1), linestyle = ':', color = cslow)

# ax.scatter(Nlabels, np.sum(fluxes['fast']['dispersed'], axis = 1), label = 'FAST scenario', color = cfast)
# ax.plot(Nlabels, np.sum(fluxes['fast']['dispersed'], axis = 1), linestyle = ':', color = cfast)

# ax.set_xlabel('Effective pressure (% of overburden)')
# ax.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
# plt.title('Dispersed layer flux', size = 18)
# plt.legend()
# plt.savefig('./figures/dispersed_fluxes.png', dpi = 300)

# All fluxes
# fastmark = 'o'
# slowmark = 'x'
# fringecol = 'C4'
# dispcol = 'C5'

# fig, ax = plt.subplots(figsize = (12 , 6))
# ax2 = ax.twinx()

# ax.scatter(Nlabels, np.sum(fluxes['slow']['fringe'], axis = 1), label = 'SLOW, Frozen fringe', color = fringecol, marker = slowmark)
# ax.plot(Nlabels, np.sum(fluxes['slow']['fringe'], axis = 1), linestyle = ':', color = fringecol)

# ax.scatter(Nlabels, np.sum(fluxes['fast']['fringe'], axis = 1), label = 'FAST, Frozen fringe', color = fringecol, marker = fastmark)
# ax.plot(Nlabels, np.sum(fluxes['fast']['fringe'], axis = 1), linestyle = ':', color = fringecol)

# ax2.scatter(Nlabels, np.sum(fluxes['slow']['dispersed'], axis = 1), label = 'SLOW, Dispersed layer', color = dispcol, marker = slowmark)
# ax2.plot(Nlabels, np.sum(fluxes['slow']['dispersed'], axis = 1), linestyle = ':', color = dispcol)

# ax2.scatter(Nlabels, np.sum(fluxes['fast']['dispersed'], axis = 1), label = 'FAST, Dispersed layer', color = dispcol, marker = fastmark)
# ax2.plot(Nlabels, np.sum(fluxes['fast']['dispersed'], axis = 1), linestyle = ':', color = dispcol)

# ax.set_xlabel('Effective pressure (% of overburden)')
# ax.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
# ax.set_ylim([-500, 50000])

# ax2.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
# ax2.set_ylim([-35, 3500])

# ax.spines['left'].set_color(fringecol)
# ax.yaxis.label.set_color(fringecol)
# ax.tick_params(axis='y', colors=fringecol)

# ax2.spines['right'].set_color(dispcol)
# ax2.yaxis.label.set_color(dispcol)
# ax2.tick_params(axis='y', colors=dispcol)

# plt.title('Sediment flux results', size = 18)
# ax.legend(loc = 'upper left')
# ax2.legend(loc = 'upper right')
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

ax.set_ylim([0, 5.5])

ax.set_xlabel('Effective pressure scenario (% of overburden)')
ax.set_ylabel('Layer thickness (m)')
plt.title('Layer heights at the terminus ($\mathtt{SLOW}$ scenario)', size = 20)

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='royalblue', label='Frozen fringe'),
                   plt.Rectangle((0, 0), 1, 1, facecolor='navajowhite', edgecolor='orange', label='Dispersed layer')]
ax.legend(handles=legend_elements, loc='upper right')

plt.savefig('./figures/slow_scenario_layers.png', dpi = 300)

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

ax.set_ylim([0, 5.5])

ax.set_xlabel('Effective pressure scenario (% of overburden)')
ax.set_ylabel('Layer thickness (m)')
plt.title('Layer heights at the terminus ($\mathtt{FAST}$ scenario)', size = 20)

legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='royalblue', label='Frozen fringe'),
                   plt.Rectangle((0, 0), 1, 1, facecolor='navajowhite', edgecolor='orange', label='Dispersed layer')]
ax.legend(handles=legend_elements, loc='upper right')

plt.savefig('./figures/fast_scenario_layers.png', dpi = 300)
