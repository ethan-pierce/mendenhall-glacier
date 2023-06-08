import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.size': 20})

geology = Dataset('./inputs/igm-results/geology-optimized.nc')
glacier = Dataset('./inputs/igm-results/ex.nc')

fields = [
    geology['thkobs'][:],
    geology['uvelsurfobs'][:],
    geology['vvelsurfobs'][:],
    geology['thk'][:],
    glacier['uvelbase'][0,:,:],
    glacier['vvelbase'][0,:,:] * -1
]

titles = [
    'Observed ice thickness (m)',
    'Observed (x) surface velocity (m a$^{-1}$)',
    'Observed (y) surface velocity (m a$^{-1}$)',
    'Modeled ice thickness (m)',
    'Modeled (x) sliding velocity (m a$^{-1}$)',
    'Modeled (y) sliding velocity (m a$^{-1}$)'
]

mask = geology['icemask'][:]

fig, ax = plt.subplots(2, 3, figsize = (20, 12))

for i in range(len(np.ravel(ax))):
    axis = np.ravel(ax)[i]
    plot = np.where(
        mask,
        fields[i],
        np.nan
    )
    
    mn = np.nanmin(plot)
    mx = np.nanmax(plot)

    axis.imshow(mask + 0.65, cmap = 'Greys_r', vmin = 0, vmax = 1)

    if i in [0, 3]:
        cmap = 'Blues_r'
        im = axis.imshow(plot, cmap = cmap)
    else:
        divnorm = TwoSlopeNorm(vmin = mn, vcenter = 0, vmax = mx)
        cmap = 'RdBu_r'
        im = axis.imshow(plot, cmap = cmap, norm = divnorm)
    
    axis.set_title(titles[i], y = 1.05)
    
    if i in [0, 3]:
        cbar_ticks = [0, 0.25 * mx, 0.5 * mx, 0.75 * mx, mx]
    else:
        cbar_ticks = [mn, 0.66 * mn, 0.33 * mn, 0, 0.33 * mx, 0.66 * mx, mx]

    cbar = plt.colorbar(im, ax = axis, fraction = 0.0543, pad = 0.04, ticks = cbar_ticks)
    cbar.ax.set_yticklabels([str(int(5 * round(i / 5))) for i in cbar_ticks])

    if i in [3, 4, 5]:
        axis.set_xlabel('Grid x')
    if i in [0, 3]:
        axis.set_ylabel('Grid y')

plt.tight_layout()
plt.savefig('./figures/boundary_conditions.png', dpi = 300)
