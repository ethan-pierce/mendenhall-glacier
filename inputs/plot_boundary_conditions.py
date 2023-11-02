import numpy as np
from netCDF4 import Dataset

import matplotlib.pyplot as plt
import cmcrameri as cmc

plt.style.use('tableau-colorblind10')
plt.rcParams['image.cmap'] = 'cmc.bilbao_r'
plt.rcParams['font.size'] = 14

from basis.src.basis import BasalIceStratigrapher

inputs = Dataset('./inputs/igm-results/input_saved.nc')
fast = Dataset('./inputs/igm-results/fast-output.nc')
slow = Dataset('./inputs/igm-results/slow-output.nc')

def regrid(field):
    return np.flip(field, axis = 0)

mask = np.where(inputs.variables['thk'][:] > 0.1, np.nan, 1)
xlim = [25, 175]
ylim = [200, 25]

fig, ax = plt.subplots(2, 2, figsize = (12, 12))

# select = inputs.variables['thk'][:] > 0.1
# L1_vel = np.mean(np.abs(fast.variables['velsurf_mag'][0][select] - np.sqrt(inputs.variables['vvelsurfobs'][:]**2 + inputs.variables['uvelsurfobs'][:]**2)[select]))
# L1_thk = np.mean(np.abs(fast.variables['thk'][0][select] - inputs.variables['thk'][:][select]))

# L1_vel = np.mean(np.abs(slow.variables['velsurf_mag'][0][select] - np.sqrt(inputs.variables['vvelsurfobs'][:]**2 + inputs.variables['uvelsurfobs'][:]**2)[select]))
# L1_thk = np.mean(np.abs(slow.variables['thk'][0][select] - inputs.variables['thk'][:][select]))

im0 = ax[0, 0].imshow(regrid(fast.variables['thk'][0]), cmap = 'cmc.devon')
plt.colorbar(im0, ax = ax[0, 0])
ax[0, 0].imshow(regrid(mask), cmap = 'Greys_r', vmin = 0, vmax = 1)
ax[0, 0].set_title('Ice thickness (m)')
ax[0, 0].set_xlabel('Grid x')
ax[0, 0].set_ylabel('Grid y')
ax[0, 0].set_xlim(xlim)
ax[0, 0].set_ylim(ylim)
ax[0, 0].tick_params(axis = 'x', which = 'both', size = 8)
ax[0, 0].tick_params(axis = 'y', which = 'both', size = 8)
ax[0, 0].annotate('Std. Dev. = 54.59 m', (90, 195))

im1 = ax[0, 1].imshow(regrid(np.sqrt(fast.variables['uvelbase'][0]**2 + fast.variables['vvelbase'][0]**2)), cmap = 'cmc.lajolla')
plt.colorbar(im1, ax = ax[0, 1])
ax[0, 1].imshow(regrid(mask), cmap = 'Greys_r', vmin = 0, vmax = 1)
ax[0, 1].set_title('Sliding velocity magnitude (m a$^{-1}$)')
ax[0, 1].set_xlabel('Grid x')
ax[0, 1].set_ylabel('Grid y')
ax[0, 1].set_xlim(xlim)
ax[0, 1].set_ylim(ylim)
ax[0, 1].tick_params(axis = 'x', which = 'both', size = 8)
ax[0, 1].tick_params(axis = 'y', which = 'both', size = 8)
ax[0, 1].annotate('Std. Dev. = 11.01 m a$^{-1}$', (80, 195))

im2 = ax[1, 0].imshow(regrid(slow.variables['thk'][0]), cmap = 'cmc.devon')
plt.colorbar(im0, ax = ax[1, 0])
ax[1, 0].imshow(regrid(mask), cmap = 'Greys_r', vmin = 0, vmax = 1)
ax[1, 0].set_title('Ice thickness (m)')
ax[1, 0].set_xlabel('Grid x')
ax[1, 0].set_ylabel('Grid y')
ax[1, 0].set_xlim(xlim)
ax[1, 0].set_ylim(ylim)
ax[1, 0].tick_params(axis = 'x', which = 'both', size = 8)
ax[1, 0].tick_params(axis = 'y', which = 'both', size = 8)
ax[1, 0].annotate('Std. Dev. = 62.06 m', (90, 195))

im3 = ax[1, 1].imshow(regrid(np.sqrt(slow.variables['uvelbase'][0]**2 + slow.variables['vvelbase'][0]**2)), cmap = 'cmc.lajolla')
plt.colorbar(im1, ax = ax[1, 1])
ax[1, 1].imshow(regrid(mask), cmap = 'Greys_r', vmin = 0, vmax = 1)
ax[1, 1].set_title('Sliding velocity magnitude (m a$^{-1}$)')
ax[1, 1].set_xlabel('Grid x')
ax[1, 1].set_ylabel('Grid y')
ax[1, 1].set_xlim(xlim)
ax[1, 1].set_ylim(ylim)
ax[1, 1].tick_params(axis = 'x', which = 'both', size = 8)
ax[1, 1].tick_params(axis = 'y', which = 'both', size = 8)
ax[1, 1].annotate('Std. Dev. = 9.99 m a$^{-1}$', (80, 195))

plt.annotate('$\mathtt{FAST}$ scenario', (0.4, 0.97), xycoords = 'figure fraction', size = 22)
plt.annotate('$\mathtt{SLOW}$ scenario', (0.4, 0.5), xycoords = 'figure fraction', size = 22)

plt.subplots_adjust(left=0.075, bottom=0.075, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
plt.savefig('./figures/igm_results.png', dpi = 300)