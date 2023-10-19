import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import tomli

from basis.src.basis import BasalIceStratigrapher

plt.rcParams.update({'font.size': 12})

model = BasalIceStratigrapher()
model.initialize('./experiments/static-effective-pressure/slow_input_file.toml')

mask = np.where(
    model.grid.at_node['ice_thickness'] > 0.5,
    1,
    0
)

def regrid(field):
    return np.flip(np.reshape(field, model.grid.shape), axis = 0)

for scenario in ['slow', 'fast']:
    input_dir = './experiments/static-effective-pressure/outputs/' + scenario + '/spatial/'

    fig, axes = plt.subplots(2, 4, figsize = (30, 30))
    a = 0

    for N in [60, 80, 90, 95]:
        fringe = np.loadtxt(input_dir + 'fringe_Pw_' + str(N) + '.txt')
        disp = np.loadtxt(input_dir + 'disp_Pw_' + str(N) + '.txt')

        axf = axes[1, a]
        axd = axes[0, a]

        axf.imshow(regrid(mask), cmap = 'Greys_r')

        field = regrid(fringe)
        toplot = np.where(
            field > 1e-3,
            field,
            np.nan
        )

        fmin = 0
        fmax = 8.5
        dmin = 0
        dmax = 4.5

        im = axf.imshow(toplot, cmap = 'pink_r')
        cbar = plt.colorbar(im, ax = axf, fraction = 0.0543, pad = 0.04)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # axf.set_title('N = ' + str(100 - N) + '% P$_i$')
        axf.set_title('Frozen fringe (m)')
        axf.set_xlabel('Grid x')
        axf.set_ylabel('Grid y')

        axd.imshow(regrid(mask), cmap = 'Greys_r')

        field = regrid(disp)
        icemask = regrid(model.grid.at_node['ice_thickness'])

        toplot = np.where(
            icemask > 0.5,
            field,
            np.nan
        )

        im = axd.imshow(toplot, cmap = 'pink_r')
        cbar = plt.colorbar(im, ax = axd, fraction = 0.0543, pad = 0.04)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # axd.set_title('N = ' + str(100 - N) + '% P$_i$')
        axd.set_title('Dispersed layer (m)')
        axd.set_xlabel('Grid x')
        axd.set_ylabel('Grid y')

        a += 1

    # plt.annotate('Frozen fringe thickness (m) at end of simulation', [0.355, 0.475], xycoords = 'figure fraction', fontsize = 22)
    # plt.suptitle('Dispersed layer thickness (m) at end of simulation')
    plt.tight_layout()
    plt.show()

    # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.35, hspace=0.1)
    # plt.savefig('./figures/slow_scenario_results.png', dpi = 300)

# -------------------------------------------------

# fig, ax = plt.subplots(1, 2, figsize = (24, 8))

# for N in [60, 65, 70, 75, 80, 85, 90, 95]:
#     fringe = np.loadtxt('./experiments/static-effective-pressure/outputs/fringe_flux_Pw_' + str(N) + '_pct.txt')
#     disp = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_flux_Pw_' + str(N) + '_pct.txt')

#     years = np.linspace(0, 250, len(fringe))

#     ax[0].plot(years, fringe, label = 'N = ' + str(100-N) +'% P$_i$')

#     ax[1].plot(years, disp, label = 'N = ' + str(100-N) +'% P$_i$')
    
    
# ax[0].set_xlabel('Year of simulation')
# ax[0].set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
# # ax[0].legend(loc = 'center right')
# ax[0].annotate('Flux from the frozen fringe', [0.2, 0.92], xycoords = 'figure fraction')

# ax[1].set_xlabel('Year of simulation')
# ax[1].set_ylabel('Sediment flux (m$^3$ a$^{-1}$)')
# ax[1].legend(loc = 'upper left')
# ax[1].annotate('Flux from the dispersed layer', [0.65, 0.92], xycoords = 'figure fraction')

# plt.suptitle('Sediment fluxes at the terminus (m$^3$ a$^{-1}$)')
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.1)
# plt.savefig('./figures/flux_results.png')

# -------------------------------------------------

# fig, axes = plt.subplots(2, 4, figsize = (24, 14))
# a = 0

# for N in [60, 65, 70, 75, 80, 85, 90, 95]:
#     disp = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_Pw_' + str(N) + '_pct.txt')

#     ax = np.ravel(axes)[a]
#     a += 1

#     ax.imshow(np.flip(np.reshape(mask, model.grid.shape), axis = 0), cmap = 'Greys')

#     field = np.flip(np.reshape(disp, model.grid.shape), axis = 0)
#     icemask = np.flip(np.reshape(model.grid.at_node['ice_thickness'], model.grid.shape), axis = 0)

#     toplot = np.where(
#         icemask > 0.5,
#         field,
#         np.nan
#     )

#     im = ax.imshow(toplot, cmap = 'pink_r', vmin = 0, vmax = 4)
#     plt.colorbar(im, ax = ax, fraction = 0.0543, pad = 0.04)

#     ax.set_title('N = ' + str(100 - N) + '% P$_i$')
#     ax.set_xlabel('Grid x')
#     ax.set_ylabel('Grid y')

# plt.suptitle('Dispersed layer thickness (m) at end of simulation')
# plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.35, hspace=0.1)
# plt.savefig('./figures/dispersed_advection.png')

# fflux = np.loadtxt('./experiments/static-effective-pressure/outputs/fringe_flux_Pw_' + str(N) + '_pct.txt')
# dflux = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_flux_Pw_' + str(N) + '_pct.txt')
