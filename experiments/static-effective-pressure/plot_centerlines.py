"""Plot the centerlines of static experiments."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from basis.src.basis import BasalIceStratigrapher
from basis.src.centerlines import identify_centerlines

plt.rcParams.update({'font.size': 18})

for N in [60, 65, 70, 75, 80, 85, 90, 95]:
    fringe = np.loadtxt('./experiments/static-effective-pressure/outputs/fringe_Pw_' + str(N) + '_pct.txt')
    disp = np.loadtxt('./experiments/static-effective-pressure/outputs/dispersed_Pw_' + str(N) + '_pct.txt')

    fringe = np.flip(fringe.reshape(328, 274), axis = 0)
    disp = np.flip(disp.reshape(328, 274), axis = 0)

    fringe = np.ma.masked_not_equal(fringe, 1e-6)
    disp = np.ma.masked_where(fringe != 1e-6, disp)

    if N >= 90:
        fgroup, fdist = identify_centerlines(fringe, percentile=72)
        dgroup, ddist = identify_centerlines(disp, percentile=72)
    else:    
        fgroup, fdist = identify_centerlines(fringe, percentile=70)
        dgroup, ddist = identify_centerlines(disp, percentile=70)

    fringe_masked = np.where(fdist > 0, fringe, 0)
    disp_masked = np.where(ddist > 0, disp, 0)

    bins = np.arange(5, 350, 0.1)
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

    fig, ax = plt.subplots(figsize = (18, 6))

    mask = (fvals != 0) & (dvals != 0)

    ax.plot(bins[mask] * 50 / 1e3, gaussian_filter1d(fvals[mask], sigma = 1), label = 'Frozen fringe layer')
    ax.plot(bins[mask] * 50 / 1e3, gaussian_filter1d(dvals[mask] + fvals[mask], sigma = 1), label = 'Dispersed layer')

    plt.legend(loc = 'upper left')
    plt.xlabel('Distance from terminus (km)')
    plt.ylabel('Height in the ice column (m)')
    plt.title('Effective pressure = ' + str(100-N) + '% overburden')
   
    plt.savefig('./experiments/static-effective-pressure/outputs/centerlines/centerline_' + str(N) +'.png')
