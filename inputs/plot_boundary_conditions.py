import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.size': 20})

from basis.src.basis import BasalIceStratigrapher

geology = Dataset('./inputs/igm-results/geology-optimized.nc')
glacier = Dataset('./inputs/igm-results/ex.nc')

# velsurf_mag = np.sqrt(glacier['uvelsurf'][0,:,:]**2 + glacier['vvelsurf'][0,:,:]**2)
# velsurfobs_mag = np.sqrt(geology['uvelsurfobs'][:]**2 + geology['vvelsurfobs'][:]**2)

# fields = [
#     geology['thkobs'][:],
#     geology['uvelsurfobs'][:],
#     geology['vvelsurfobs'][:],
#     np.abs(geology['thk'][:] - geology['thkobs'][:]) / geology['thkobs'][:],
#     np.abs(geology['usurf'][:] - geology['usurfobs'][:]) / geology['usurfobs'][:],
#     np.abs(velsurf_mag - velsurfobs_mag) / velsurfobs_mag
# ]

# titles = [
#     'Observed ice thickness (m)',
#     'Observed (x) surface velocity (m a$^{-1}$)',
#     'Observed (y) surface velocity (m a$^{-1}$)',
#     'Relative error in ice thickness',
#     'Relative error in surface elevation',
#     'Relative error in surface velocity'
# ]

# mask = geology['icemask'][:]

# fig, ax = plt.subplots(2, 3, figsize = (20, 12))

# for i in range(len(np.ravel(ax))):
#     axis = np.ravel(ax)[i]
#     plot = np.where(
#         mask,
#         fields[i],
#         np.nan
#     )
    
#     mn = np.nanmin(plot)

#     if i < 3:
#         mx = np.nanmax(plot)
#     elif i == 5:
#         mx = np.nanpercentile(plot, 75)
#     else:
#         mx = np.nanpercentile(plot, 95)

#     axis.imshow(mask + 0.65, cmap = 'Greys_r', vmin = 0, vmax = 1)

#     if i in [0]:
#         cmap = 'Blues_r'
#         im = axis.imshow(plot, cmap = cmap)
#     elif i in [1, 2]:
#         divnorm = TwoSlopeNorm(vmin = mn, vcenter = 0, vmax = mx)
#         cmap = 'RdBu_r'
#         im = axis.imshow(plot, cmap = cmap, norm = divnorm)
#     else:
#         cmap = 'RdYlBu_r'
#         im = axis.imshow(plot, cmap = cmap, vmin = mn, vmax = mx)
    
#     axis.set_title(titles[i], y = 1.05)
    
#     if i in [0]:
#         cbar_ticks = [0, 0.25 * mx, 0.5 * mx, 0.75 * mx, mx]
#     elif i in [1, 2]:
#         cbar_ticks = [mn, 0.66 * mn, 0.33 * mn, 0, 0.33 * mx, 0.66 * mx, mx]
#     else:
#         cbar_ticks = np.linspace(mn, mx, 5)
    
#     cbar = plt.colorbar(im, ax = axis, fraction = 0.0543, pad = 0.04, ticks = cbar_ticks)

#     if i < 3:
#         cbar.ax.set_yticklabels([str(int(10 * round(i / 10))) for i in cbar_ticks])
#     elif i in [3, 5]:
#         cbar.ax.set_yticklabels(["{:0.2f}".format(0.1 * round(i / 0.1)) for i in cbar_ticks])
#     else:
#         cbar.ax.set_yticklabels([str(0.01 * round(i / 0.01)) for i in cbar_ticks])

#     if i in [3, 4, 5]:
#         axis.set_xlabel('Grid x')
#     if i in [0, 3]:
#         axis.set_ylabel('Grid y')

# plt.tight_layout()
# plt.savefig('./figures/inversion_results.png', dpi = 400)


fields = [
    glacier['thk'][0],
    np.sqrt(glacier['uvelbase'][0]**2 + glacier['vvelbase'][0]**2)
]

titles = [
    'Modeled ice thickness (m)',
    'Modeled sliding velocity (m a$^{-1}$)'
]

cmaps = [
    'Blues_r',
    'RdYlBu_r',
]

mask = geology['icemask'][:]

fig, ax = plt.subplots(1, 2, figsize = (18, 9))

for i in range(len(np.ravel(ax))):
    axis = np.ravel(ax)[i]
    plot = np.where(
        mask,
        fields[i],
        np.nan
    )
    
    mn = np.nanmin(plot)
    if i == 0:
        mx = np.nanmax(plot)
    else:
        mx = np.nanpercentile(plot, 99)

    axis.imshow(mask + 0.65, cmap = 'Greys_r', vmin = 0, vmax = 1)

    cmap = cmaps[i]
    im = axis.imshow(plot, cmap = cmap, vmin = mn, vmax = mx)
    
    axis.set_title(titles[i], y = 1.05)

    cbar_ticks = np.linspace(mn, mx, 5)    
    
    if i == 0:
        cbar = plt.colorbar(im, ax = axis, fraction = 0.0543, pad = 0.04, ticks = cbar_ticks)
    if i == 1:
        cbar = plt.colorbar(im, ax = axis, fraction = 0.0543, pad = 0.04, ticks = cbar_ticks, extend = 'max')
    cbar.ax.set_yticklabels([str(int(10 * round(i / 10))) for i in cbar_ticks])

    axis.set_xlabel('Grid x')
    axis.set_ylabel('Grid y')

plt.tight_layout()
plt.savefig('./figures/boundary_conditions.png', dpi = 300)