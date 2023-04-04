import numpy as np
import matplotlib.pyplot as plt
import tomli

from basis.src.basis import BasalIceStratigrapher

plt.rcParams.update({'font.size': 18})

BIS = BasalIceStratigrapher()
BIS.initialize('./experiments/static-effective-pressure/input_file.toml')

mask = np.where(
    BIS.grid.at_node['ice_thickness'] > 0.5,
    1,
    0
)

fig, axes = plt.subplots(2, 4, figsize = (24, 14))
a = 0

for N in [60, 65, 70, 75, 80, 85, 90, 95]:
    fringe = np.loadtxt('./experiments/static-effective-pressure/outputs/fringe_Pw_' + str(N) + '_pct.txt')
    
    ax = np.ravel(axes)[a]
    a += 1

    ax.imshow(np.flip(np.reshape(mask, BIS.grid.shape), axis = 0), cmap = 'Greys')

    field = np.flip(np.reshape(fringe, BIS.grid.shape), axis = 0)
    toplot = np.where(
        field > 1e-3,
        field,
        np.nan
    )

    im = ax.imshow(toplot, cmap = 'pink_r', vmin = 0, vmax = 8)
    plt.colorbar(im, ax = ax, fraction = 0.0543, pad = 0.04)

    ax.set_title('N = ' + str(100 - N) + '% P$_i$')
    ax.set_xlabel('Grid x')
    ax.set_ylabel('Grid y')

plt.suptitle('Frozen fringe thickness (m) at end of simulation')
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
plt.savefig('./figures/fringe_advection.png')

# -------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize = (24, 14))
a = 0

for N in [60, 65, 70, 75, 80, 85, 90, 95]:
    disp = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_Pw_' + str(N) + '_pct.txt')

    ax = np.ravel(axes)[a]
    a += 1

    ax.imshow(np.flip(np.reshape(mask, BIS.grid.shape), axis = 0), cmap = 'Greys')

    field = np.flip(np.reshape(disp, BIS.grid.shape), axis = 0)
    icemask = np.flip(np.reshape(BIS.grid.at_node['ice_thickness'], BIS.grid.shape), axis = 0)

    toplot = np.where(
        icemask > 0.5,
        field,
        np.nan
    )

    im = ax.imshow(toplot, cmap = 'pink_r', vmin = 0, vmax = 4)
    plt.colorbar(im, ax = ax, fraction = 0.0543, pad = 0.04)

    ax.set_title('N = ' + str(100 - N) + '% P$_i$')
    ax.set_xlabel('Grid x')
    ax.set_ylabel('Grid y')

plt.suptitle('Dispersed layer thickness (m) at end of simulation')
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.35, hspace=0.1)
plt.savefig('./figures/dispersed_advection.png')

fflux = np.loadtxt('./experiments/static-effective-pressure/outputs/fringe_flux_Pw_' + str(N) + '_pct.txt')
dflux = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_flux_Pw_' + str(N) + '_pct.txt')
