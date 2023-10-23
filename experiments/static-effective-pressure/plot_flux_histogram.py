import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tomli
from landlab.plot import imshow_grid
from basis.src.basis import BasalIceStratigrapher

plt.rcParams.update({'font.size': 14})

Ns = [60, 80, 90, 95]
boxplots = {
    'slow': {N: [] for N in Ns},
    'fast': {N: [] for N in Ns}
}

fluxes = {
    'slow': {'fringe': [], 'dispersed': []},
    'fast': {'fringe': [], 'dispersed': []}
}

alt_fluxes = {
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

    for N in Ns:
        fringe_flux = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/flux/fringe_Pw_' + str(N) + '.txt')
        dispersed_flux = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/flux/disp_Pw_' + str(N) + '.txt')

        fluxes[scenario]['fringe'].append(fringe_flux)
        fluxes[scenario]['dispersed'].append(dispersed_flux)

        fringe_2d = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/history/fringe_Pw_' + str(N) + '.txt')
        dispersed_2d = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/history/disp_Pw_' + str(N) + '.txt')

        idx = np.ravel(np.argwhere(model.grid.at_node['adjacent_to_terminus'] | model.grid.at_node['is_terminus']))
        dispersed_at_terminus = dispersed_2d[:, idx]

        fringe_spinup = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spinup/fringe_Pw_' + str(N) + '.txt')
        dispersed_spinup = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spinup/disp_Pw_' + str(N) + '.txt')


        print(fringe_2d[1, idx] - fringe_spinup[idx])
        quit()

quit()

fig, ax = plt.subplots(figsize = (12 , 6))

ax.scatter(Ns, [fluxes['slow']['fringe'][i][-1] for i in range(len(Ns))])
ax.scatter(Ns, [fluxes['fast']['fringe'][i][-1] for i in range(len(Ns))])

ax.scatter(Ns, [fluxes['slow']['dispersed'][i][-1] for i in range(len(Ns))])
ax.scatter(Ns, [fluxes['fast']['dispersed'][i][-1] for i in range(len(Ns))])

plt.show()


# # Boxplot
# fig, ax = plt.subplots(figsize = (12, 6))

# fplot = ax.boxplot(
#     fboxes, 
#     positions = np.arange(len(fboxes)), 
#     widths = 0.4, 
#     bootstrap = 1000,
#     patch_artist = True,
#     boxprops = dict(
#         facecolor = 'lightblue'
#     ),
#     medianprops = dict(
#         color = 'royalblue',
#         linewidth = 2
#     )
# )

# dplot = ax.boxplot(
#     dboxes,
#     positions = np.arange(len(fboxes)) + 0.425,
#     widths = 0.4,
#     bootstrap = 1000,
#     patch_artist = True,
#     boxprops = dict(
#         facecolor = 'navajowhite'
#     ),
#     medianprops = dict(
#         color = 'orange',
#         linewidth = 2
#     )
# )

# ax.yaxis.grid(True, which = 'both', linestyle = ':', linewidth = 0.5, alpha = 0.7, color = 'gray')
# ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))

# ax.set_xticks(np.arange(len(fboxes)) + 0.2)
# ax.set_xticklabels([str(100 - N) for N in Ns])

# ax.set_xlabel('Effective pressure scenario (% of overburden)')
# ax.set_ylabel('Layer thickness (m)')
# plt.title('Modeled layer thicknesses at the terminus')

# legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='royalblue', label='Frozen fringe'),
#                    plt.Rectangle((0, 0), 1, 1, facecolor='navajowhite', edgecolor='orange', label='Dispersed layer')]
# ax.legend(handles=legend_elements, loc='upper right')

# plt.savefig('./figures/stratigraphy_boxplot.png', dpi = 300)
# plt.close('all')



# # Line plot
# fig, ax = plt.subplots(figsize = (12, 6))
# ax2 = ax.twinx()

# blue = 'royalblue'

# fplot = ax.plot(Ns, ffluxes, color = blue, label = 'Frozen fringe')
# dplot = ax2.plot(Ns, dfluxes, color = 'orange', label = 'Dispersed layer')
# sumplot = ax.plot(
#     Ns, np.array(ffluxes) + np.array(dfluxes), 
#     color = blue, label = 'Sum of layers',
#     linestyle = ':', linewidth = 2.5
# )

# ax.set_xticks(Ns)
# ax.set_xticklabels([str(100 - N) for N in Ns])
# ax.set_xlabel('Effective pressure scenario (% of overburden)')
# ax.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
# ax2.set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')

# # ax.set_ylim([0, 56000])
# ax2.set_ylim([1500, 4500])

# ax2.spines['left'].set_color(blue)
# ax.tick_params(axis='y', colors=blue)
# ax2.spines['right'].set_color('orange')
# ax2.tick_params(axis='y', colors='orange')

# lines, labs = ax.get_legend_handles_labels()
# lines2, labs2 = ax2.get_legend_handles_labels()
# plt.legend(lines + lines2, labs + labs2, loc = 'best')

# plt.title('Modeled sediment flux at the terminus')
# plt.savefig('./figures/fluxes_per_scenario.png', dpi = 300)
