import numpy as np
from netCDF4 import Dataset

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.size': 18})

geology = Dataset('./inputs/igm-results/geology-optimized.nc')
glacier = Dataset('./inputs/igm-results/ex.nc')

fig, ax = plt.subplots(2, 3, figsize = (30, 18))

cmap = 'viridis'

plot0 = geology['usurf'][:]
im0 = ax[0, 0].imshow(plot0, cmap = cmap)
ax[0, 0].set_title('Surface elevation (m)')
plt.colorbar(im0, ax = ax[0, 0], fraction = 0.0543, pad = 0.04)

plot1 = geology['thk'][:]
im1 = ax[0, 1].imshow(plot1, cmap = cmap)
ax[0, 1].set_title('Ice thickness (m)')
plt.colorbar(im1, ax = ax[0, 1], fraction = 0.0543, pad = 0.04)

plot2 = glacier['topg'][0]
im2 = ax[0, 2].imshow(plot2, cmap = cmap)
ax[0, 2].set_title('Bedrock elevation (m)')
plt.colorbar(im2, ax = ax[0, 2], fraction = 0.0543, pad = 0.04)

plot3 = np.where(plot1 > 0.5, geology['slidingco'][:], np.nan)
im3 = ax[1, 0].imshow(plot3, cmap = cmap)
ax[1, 0].set_title('Sliding coefficient')
plt.colorbar(im3, ax = ax[1, 0], fraction = 0.0543, pad = 0.04)

plot4 = np.where(plot1 > 0.5, np.sqrt(glacier['uvelbase'][0]**2 + glacier['vvelbase'][0]**2), np.nan)
im4 = ax[1, 1].imshow(plot4, cmap = cmap)
ax[1, 1].set_title('Sliding velocity (m a$^{-1}$)')
plt.colorbar(im4, ax = ax[1, 1], fraction = 0.0543, pad = 0.04)

plot5 = np.where(plot1 > 0.5, glacier['velbar_mag'][0], np.nan)
im5 = ax[1, 2].imshow(plot5, cmap = cmap)
ax[1, 2].set_title('Depth-averaged velocity m a$^{-1}$)')
plt.colorbar(im5, ax = ax[1, 2], fraction = 0.0543, pad = 0.04)

xticks = [i for i in range(plot0.shape[1]) if i % 35 == 0]
xticklabels = [int(np.round(geology['x'][i] / 1e3, 0)) for i in range(len(geology['x'][:])) if i % 35 == 0]

yticks = [i for i in range(plot0.shape[0]) if i % 41 == 0]
yticklabels = [int(np.round(geology['y'][i] / 1e3, 0)) for i in range(len(geology['y'][:])) if i % 41 == 0]

for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i, j].set_xticks(xticks, xticklabels, rotation = 45)
        ax[i, j].set_yticks(yticks, yticklabels, rotation = 45)

        ax[i, j].set_xlabel('UTM zone 8N (km)')
        ax[i, j].set_ylabel('UTM zone 8N (km)')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig('./figures/boundary-conditions.png', dpi = 300)