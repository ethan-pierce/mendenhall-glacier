import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cmcrameri.cm as cmc
from basis.src.basis import BasalIceStratigrapher

plt.style.use('tableau-colorblind10')
plt.rcParams['image.cmap'] = 'cmc.bilbao_r'
plt.rcParams['font.size'] = 18

model = BasalIceStratigrapher()
model.initialize('./experiments/static-effective-pressure/slow_input_file.toml')

mask = np.where(
    model.grid.at_node['ice_thickness'] > 0.5,
    1,
    0
)

xmin = np.nanmin(model.grid.node_x[model.grid.at_node['ice_thickness'] > 0.5]) / model.grid.dx
xmax = np.nanmax(model.grid.node_x[model.grid.at_node['ice_thickness'] > 0.5]) / model.grid.dx
ymin = np.nanmin(model.grid.node_y[model.grid.at_node['ice_thickness'] > 0.5]) / model.grid.dy
ymax = np.nanmax(model.grid.node_y[model.grid.at_node['ice_thickness'] > 0.5]) / model.grid.dy


def regrid(field):
    return np.flip(np.reshape(field, model.grid.shape), axis = 0)

maxmap = {'disp': {'slow': 2.0, 'fast': 2.0}, 'fringe': {'slow': 5.0, 'fast': 5.0}}
titles = {'disp'}
labelmap = {'slow': '$\mathtt{SLOW}$ scenario', 'fast': '$\mathtt{FAST}$ scenario'}
labelloc = {'slow': 0.2, 'fast': 0.675}
plt.rcParams['font.size'] = 16

for layer in ['disp', 'fringe']:
    fig, axes = plt.subplots(2, 2, figsize = (16, 16))
    a = 0

    for scenario in ['fast', 'slow']:
        input_dir = './experiments/static-effective-pressure/outputs/' + scenario + '/spatial/'
        Ns = [60, 90]

        plt.annotate(labelmap[scenario], (0.008, labelloc[scenario]), xycoords = 'figure fraction', rotation = 90, size = 26)

        for i in range(len(Ns)):
            data = np.loadtxt(input_dir + layer + '_Pw_' + str(Ns[i]) + '.txt')
            ax = axes[a, i]

            ax.imshow(regrid(mask), cmap = 'Greys_r')
            field = np.where(regrid(model.grid.at_node['ice_thickness'][:] > 0.5), regrid(data), np.nan)

            im = ax.imshow(field, vmax = maxmap[layer][scenario], vmin = 0)
            cbar = plt.colorbar(im, ax = ax, fraction = 0.0543, pad = 0.04)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # ax.arrow(38, 190, 15, 0, width = 0.5, color = 'white')
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymax, ymin])
            ax.set_xlabel('Grid x')
            ax.set_ylabel('Grid y')

            if a == 0:
                ax.set_title('N = ' + str(100 - Ns[i]) + '% P$_i$', size = 26)

        a += 1

    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, wspace=0.225, hspace=0.175)
    plt.savefig('./figures/' + layer + '_layer_results.png', dpi = 300)

# for scenario in ['', 'fast']:
#     input_dir = './experiments/static-effective-pressure/outputs/' + scenario + '/spatial/'

#     fig, axes = plt.subplots(4, 2, figsize = (14, 28))
#     a = 0

#     if scenario == 'slow':
#         fmax = 5.0
#         fmin = 0.0
#         dmax = 2.0
#         dmin = 0.0
#     else:
#         fmax = 3.5
#         fmin = 0.0
#         dmax = 1.0
#         dmin = 0.0

#     for N in [60, 80, 90, 95]:
#         fringe = np.loadtxt(input_dir + 'fringe_Pw_' + str(N) + '.txt')
#         disp = np.loadtxt(input_dir + 'disp_Pw_' + str(N) + '.txt')

#         axd = axes[a, 0]
#         axd.imshow(regrid(mask), cmap = 'Greys_r')

#         field = regrid(disp)
#         icemask = regrid(model.grid.at_node['ice_thickness'])

#         toplot = np.where(
#             icemask > 0.5,
#             field,
#             np.nan
#         )

#         im = axd.imshow(toplot, vmax = dmax, vmin = dmin)
#         cbar = plt.colorbar(im, ax = axd, fraction = 0.0543, pad = 0.04)
#         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#         axd.set_xlim([xmin, xmax])
#         axd.set_ylim([ymax, ymin])
#         axd.set_title('Dispersed layer (m)', size = 22)
#         axd.set_xlabel('Grid x')
#         axd.set_ylabel('Grid y')
        
#         axf = axes[a, 1]
#         axf.imshow(regrid(mask), cmap = 'Greys_r')

#         field = regrid(fringe)
#         toplot = np.where(
#             field > 1e-3,
#             field,
#             np.nan
#         )

#         im = axf.imshow(toplot, vmax = fmax, vmin = fmin)
#         cbar = plt.colorbar(im, ax = axf, fraction = 0.0543, pad = 0.04)
#         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#         axf.set_xlim([xmin, xmax])
#         axf.set_ylim([ymax, ymin])        
#         axf.set_title('Frozen fringe (m)', size = 22)
#         axf.set_xlabel('Grid x')
#         axf.set_ylabel('Grid y')

#         a += 1

#     plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.35, hspace=0.2)
#     plt.savefig('./figures/' + scenario + '_scenario_results.png', dpi = 300)
