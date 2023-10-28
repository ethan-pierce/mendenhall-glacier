"""Plot the centerlines of static experiments."""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
import cmcrameri as cmc
import copy
import skimage.graph
from scipy.ndimage import gaussian_filter1d
from basis.src.basis import BasalIceStratigrapher
from landlab.plot import imshow_grid

plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'image.cmap': 'cmc.bilbaoS'})

centerline = (rxr.open_rasterio('./inputs/centerline-binary.tif')[0].values).astype(int)

Ns = [60, 65, 70, 75, 80, 85, 90, 95]

def regrid(field, grid):
    return np.flip(np.reshape(field, grid.shape), axis = 0)

for scenario in ['fast', 'slow']:
    results = {N: {'fvals': None, 'dvals': None} for N in Ns}

    model = BasalIceStratigrapher()
    model.initialize('./experiments/static-effective-pressure/' + scenario + '_input_file.toml')
    glacier = np.flip(np.reshape(model.grid.at_node['ice_thickness'][:], model.grid.shape), axis = 0)

    model.grid.add_field('centerline', np.flip(centerline, axis = 0), at = 'node')

    costs = np.where(centerline, 100, 1e6)
    outlet = (60, 195)
    origin = (156, 30)

    pather = skimage.graph.MCP(costs, fully_connected = True)
    distance, _ = pather.find_costs([(outlet[1], outlet[0])])

    along_centerline = np.where(centerline, distance, -1)
    model.grid.add_field('distance_along_centerline', np.flip(along_centerline, axis = 0), at = 'node')

    for i in range(len(Ns)):
        N = Ns[i]

        fringe = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/fringe_Pw_' + str(N) + '.txt')
        dispersed = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/disp_Pw_' + str(N) + '.txt')

        D = model.grid.at_node['distance_along_centerline'][:]
        Hf = model.grid.add_field('fringe_thickness', fringe, at = 'node', clobber = True)
        Hd = model.grid.add_field('dispersed_layer_thickness', dispersed, at = 'node', clobber = True)

        distance = D[np.nonzero(np.where(model.grid.at_node['centerline'][:] == 1, D, 0))]
        fringe = Hf[np.nonzero(np.where(model.grid.at_node['centerline'][:] == 1, Hf, 0))]
        dispersed = Hd[np.nonzero(np.where(model.grid.at_node['centerline'][:] == 1, Hd, 0))]

        i = 0
        sort = np.argsort(distance)
        sorted_distance = distance[sort][i:] * 1e-3
        sorted_fringe = gaussian_filter1d(fringe[sort], sigma = 1)[i:]
        sorted_dispersed = gaussian_filter1d(fringe[sort], sigma = 1)[i:] + gaussian_filter1d(dispersed[sort], sigma = 1)[i:]

        fig, ax = plt.subplots(figsize = (32, 6))
        fcol = 'C4'
        dcol = 'C5'

        plt.plot(sorted_distance, sorted_fringe, color = fcol, label = 'Frozen fringe')
        plt.plot(sorted_distance, sorted_dispersed, color = dcol, label = 'Dispersed layer')

        # plt.vlines(sorted_distance[0], sorted_fringe[0], sorted_dispersed[0], color = dcol, linestyle = ':')
        # plt.vlines(sorted_distance[0], 0, sorted_fringe[0], color = fcol, linestyle = ':')

        # plt.annotate(str(round(sorted_fringe[0], 2)) + ' m', [-0.75, 0.0], size = 14, color = fcol)
        # plt.annotate(str(round(sorted_dispersed[0] - sorted_fringe[0], 2)) + ' m', [-0.75, 1.0], size = 14, color = dcol)

        if scenario == 'fast':
            ax.set_ylim([-0.1, 4.0])
        elif scenario == 'slow':
            ax.set_ylim([-0.1, 8.0])
        else:
            pass

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc = 'upper left')

        plt.xlabel('Distance from terminus (km)')
        plt.ylabel('Height in the ice column (m)')
        plt.title('Effective pressure = ' + str(100-N) + '% overburden')

        plt.savefig('./figures/centerlines/' + scenario + '_centerline_Pw_' + str(N) + '.png', dpi = 300)
