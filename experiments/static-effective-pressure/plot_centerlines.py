"""Plot the centerlines of static experiments."""
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri as cmc
import copy
from scipy.ndimage import gaussian_filter1d
from basis.src.basis import BasalIceStratigrapher
from basis.src.centerlines import identify_centerlines

plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'image.cmap': 'cmc.bilbaoS'})

Ns = [60, 80, 90, 95]
ps = [70, 70, 70, 70]

for scenario in ['slow', 'fast']:
    results = {N: {'fvals': None, 'dvals': None} for N in Ns}

    model = BasalIceStratigrapher()
    model.initialize('./experiments/static-effective-pressure/' + scenario + '_input_file.toml')
    glacier = np.flip(np.reshape(model.grid.at_node['ice_thickness'][:], model.grid.shape), axis = 0)

    for i in range(len(Ns)):
        N = Ns[i]
        p = ps[i]

        fringe = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/fringe_Pw_' + str(N) + '.txt')
        disp = np.loadtxt('./experiments/static-effective-pressure/outputs/' + scenario + '/spatial/disp_Pw_' + str(N) + '.txt')

        fringe = np.flip(fringe.reshape(224, 197), axis = 0)
        disp = np.flip(disp.reshape(224, 197), axis = 0)

        fringe = np.ma.masked_where(glacier > 0.5, fringe)
        disp = np.ma.masked_where(glacier > 0.5, disp)

        fgroup, fdist = identify_centerlines(fringe, percentile=p)
        dgroup, ddist = identify_centerlines(disp, percentile=p)

        fringe_masked = np.where(fdist > 0, fringe, 0)
        disp_masked = np.where(ddist > 0, disp, 0)

        bins = np.arange(1, 145, 0.5)
        fvals = np.zeros_like(bins)
        dvals = np.zeros_like(bins)

        for i in range(len(bins)):
            if i != 0:
                mn = bins[i - 1]
            else:
                mn = 0
            mx = bins[i]
            
            all_fvals = fringe_masked[(fdist != 0) & (fdist <= mx) & (fdist >= mn)]

            if len(all_fvals > 0):
                fvals[i] = np.mean(all_fvals)
            else:
                fvals[i] = 0

            all_dvals = disp_masked[(ddist != 0) & (ddist <= mx) & (ddist >= mn)]

            if len(all_dvals > 0):
                dvals[i] = np.mean(all_dvals)
            else:
                dvals[i] = 0

        results[N]['fvals'] = copy.deepcopy(fvals)
        results[N]['dvals'] = copy.deepcopy(dvals)

        base = 0
        fringe_top = gaussian_filter1d(fvals, sigma = 1)[0]
        disp_top = gaussian_filter1d(dvals + fvals, sigma = 1)[0]

        fig, ax = plt.subplots(figsize = (18, 6))

        frg = ax.plot(bins * 100 / 1e3, gaussian_filter1d(fvals, sigma = 1), label = 'Frozen fringe layer')
        dsp = ax.plot(bins * 100 / 1e3, gaussian_filter1d(dvals + fvals, sigma = 1), label = 'Dispersed layer')

        plt.vlines(bins[0] * 100 / 1e3, fringe_top, disp_top, color = 'C1', linestyle = ':')
        plt.vlines(bins[0] * 100 / 1e3, base, fringe_top, color = 'C0', linestyle = ':')

        plt.annotate(str(round(disp_top - fringe_top, 2)) + ' m', [-0.5, fringe_top + (disp_top - fringe_top) / 2], size = 14, color = 'C1')
        plt.annotate(str(round(fringe_top, 2)) + ' m', [-0.5, fringe_top / 2], size = 14, color = 'C0')

        ax.set_ylim([0, 8])

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc = 'upper left')

        plt.xlabel('Distance from terminus (km)')
        plt.ylabel('Height in the ice column (m)')
        plt.title('Effective pressure = ' + str(100-N) + '% overburden')
        
        plt.savefig('./figures/centerlines/' + scenario + '_centerline_Pw_' + str(N) + '.png', dpi = 300)
