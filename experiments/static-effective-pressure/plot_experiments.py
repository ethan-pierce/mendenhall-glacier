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

for scenario in ['slow', 'fast']:
    input_dir = './experiments/static-effective-pressure/outputs/' + scenario + '/spatial/'

    fig, axes = plt.subplots(4, 2, figsize = (14, 28))
    a = 0

    if scenario == 'slow':
        fmax = 5.0
        fmin = 0.0
        dmax = 2.0
        dmin = 0.0
    else:
        fmax = 3.5
        fmin = 0.0
        dmax = 1.0
        dmin = 0.0

    for N in [60, 80, 90, 95]:
        fringe = np.loadtxt(input_dir + 'fringe_Pw_' + str(N) + '.txt')
        disp = np.loadtxt(input_dir + 'disp_Pw_' + str(N) + '.txt')

        axd = axes[a, 0]
        axd.imshow(regrid(mask), cmap = 'Greys_r')

        field = regrid(disp)
        icemask = regrid(model.grid.at_node['ice_thickness'])

        toplot = np.where(
            icemask > 0.5,
            field,
            np.nan
        )

        im = axd.imshow(toplot, vmax = dmax, vmin = dmin)
        cbar = plt.colorbar(im, ax = axd, fraction = 0.0543, pad = 0.04)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        axd.set_xlim([xmin, xmax])
        axd.set_ylim([ymax, ymin])
        axd.set_title('Dispersed layer (m)', size = 22)
        axd.set_xlabel('Grid x')
        axd.set_ylabel('Grid y')
        
        axf = axes[a, 1]
        axf.imshow(regrid(mask), cmap = 'Greys_r')

        field = regrid(fringe)
        toplot = np.where(
            field > 1e-3,
            field,
            np.nan
        )

        im = axf.imshow(toplot, vmax = fmax, vmin = fmin)
        cbar = plt.colorbar(im, ax = axf, fraction = 0.0543, pad = 0.04)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        axf.set_xlim([xmin, xmax])
        axf.set_ylim([ymax, ymin])        
        axf.set_title('Frozen fringe (m)', size = 22)
        axf.set_xlabel('Grid x')
        axf.set_ylabel('Grid y')

        a += 1

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.35, hspace=0.2)
    plt.savefig('./figures/' + scenario + '_scenario_results.png', dpi = 300)
