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

minmax = [
    [0, 550],
    [-400, 200],
    [-200, 250]
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
    if i < 3:
        mn, mx = minmax[i]
    else:
        mn, mx = minmax[i - 3]

    axis.imshow(mask + 0.75, cmap = 'Greys_r', vmin = 0, vmax = 1)
    im = axis.imshow(plot, cmap = 'viridis', vmin = mn, vmax = mx)
    
    axis.set_title(titles[i], y = 1.05)
    plt.colorbar(im, ax = axis, fraction = 0.0543, pad = 0.04)

    if i in [3, 4, 5]:
        axis.set_xlabel('Grid x')
    if i in [0, 3]:
        axis.set_ylabel('Grid y')

plt.tight_layout()
plt.savefig('./figures/boundary_conditions.png', dpi = 300)
