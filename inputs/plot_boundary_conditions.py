import numpy as np
from netCDF4 import Dataset

import matplotlib.pyplot as plt
import cmcrameri as cmc

plt.style.use('tableau-colorblind10')
plt.rcParams['image.cmap'] = 'cmc.bilbao_r'
plt.rcParams['font.size'] = 18

from basis.src.basis import BasalIceStratigrapher

inputs = Dataset('./inputs/igm-results/input_saved.nc')
fast = Dataset('./inputs/igm-results/fast-output.nc')
slow = Dataset('./inputs/igm-results/slow-output.nc')

print(inputs.variables.keys())
print(fast.variables.keys())

def regrid(field):
    return np.flip(field, axis = 0)

mask = np.where(inputs.variables['thk'][:] > 0.1, np.nan, 0.25)

fig, ax = plt.subplots(1, 2, figsize = (12, 6))

im0 = ax[0].imshow(regrid(fast.variables['thk'][0]), cmap = 'cmc.batlowK')
plt.colorbar(im0, ax = ax[0])
ax[0].imshow(regrid(mask), cmap = 'Greys_r', vmin = 0, vmax = 1)

im1 = ax[1].imshow(regrid(np.sqrt(fast.variables['uvelbase'][0]**2 + fast.variables['vvelbase'][0]**2)), cmap = 'cmc.lajolla')
plt.colorbar(im1, ax = ax[1])
ax[1].imshow(regrid(mask), cmap = 'Greys_r', vmin = 0, vmax = 1)

plt.show()