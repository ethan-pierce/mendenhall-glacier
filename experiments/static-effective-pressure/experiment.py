import numpy as np
import matplotlib.pyplot as plt

from basis.src.basis import BasalIceStratigrapher

working_dir = './experiments/static-effective-pressure/'

cfg = working_dir + 'input_file.toml'
BIS = BasalIceStratigrapher()
BIS.initialize(cfg)

BIS.calc_effective_pressure()
BIS.calc_shear_stress()
BIS.calc_erosion_rate()

# Identify terminus nodes
dx = BIS.grid.dx
dy = BIS.grid.dy
bounds = [50 * dx, 100 * dx, 300 * dy, 350 * dy]
BIS.identify_terminus(bounds)

basal_water_pressure = 0.9 * (
    BIS.grid.at_node['ice_thickness'] * BIS.params['ice_density'] * BIS.params['gravity']
)
BIS.set_value('basal_water_pressure', basal_water_pressure)

initial_till = np.full(BIS.grid.number_of_nodes, 40)
BIS.set_value('till_thickness', initial_till)

# Second, spin-up the sediment entrainment module
initial_fringe = np.full(BIS.grid.number_of_nodes, 1e-3)
BIS.set_value('fringe_thickness', initial_fringe)

initial_dispersed = np.full(BIS.grid.number_of_nodes, 1e-9)
BIS.set_value('dispersed_layer_thickness', initial_dispersed)

for t in range(100):
    BIS.entrain_sediment(t * 1e-2)

for t in range(100):
    BIS.entrain_sediment(t)

for t in range(100):
    BIS.entrain_sediment(100)
    BIS.time_elapsed += 100

for t in range(2500):
    dt = BIS.sec_per_a / 100
    BIS.entrain_sediment(dt)
    BIS.time_elapsed += dt

    if t % 100 == 0:
        print('Completed step #' + str(t))

print('Completed spin-up: ' + str(np.round(BIS.time_elapsed / BIS.sec_per_a, 2)) + ' years elapsed.')

Qf, Qd = BIS.calc_sediment_flux()
print('Qf = ' + str(Qf * BIS.sec_per_a))
print('Qd = ' + str(Qd * BIS.sec_per_a))

BIS.plot_var(
    'fringe_thickness', working_dir + '/outputs/Hf_spinup.png', 
    units_label='m',
    imshow_args={'vmin': 1e-3, 'vmax': np.percentile(BIS.grid.at_node['fringe_thickness'][:], 99)}
)

BIS.plot_var(
    'dispersed_layer_thickness', working_dir + '/outputs/Hd_spinup.png', 
    units_label='m'
)

nt = 10000
n_save = 10
BIS.create_output_file(working_dir + '/outputs/sediment.nc', n_steps = n_save)

for t in range(nt):
    dt = 0.01 * BIS.sec_per_a
    # BIS.entrain_sediment(dt)
    BIS.advect_fringe(dt)

    BIS.time_elapsed += dt

    if t % int(nt / n_save) == 0:
        print('Completed step #' + str(t))

        BIS.write_output(working_dir + '/outputs/sediment.nc')

print('Completed simulation: ' + str(np.round(BIS.time_elapsed / BIS.sec_per_a, 2)) + ' years elapsed.')

Qf, Qd = BIS.calc_sediment_flux()
print('Qf = ' + str(Qf * BIS.sec_per_a))
print('Qd = ' + str(Qd * BIS.sec_per_a))

# Plot output figures

BIS.plot_var('till_thickness', working_dir + 'outputs/Ht.png', units_label='m')
BIS.plot_var('basal_melt_rate', working_dir + 'outputs/Mb.png', units_label='m/a', scalar=BIS.sec_per_a)
BIS.plot_var('fringe_heave_rate', working_dir + 'outputs/heave.png', units_label='m/a', scalar=BIS.sec_per_a)

BIS.plot_var(
    'fringe_thickness', working_dir + '/outputs/Hf.png', 
    units_label='m',
    imshow_args={'vmin': 1e-3, 'vmax': np.percentile(BIS.grid.at_node['fringe_thickness'][:], 99)}
)

BIS.plot_var(
    'dispersed_layer_thickness', working_dir + '/outputs/Hd.png', 
    units_label='m'
)

BIS.plot_var(
    'dispersed_layer_growth_rate', working_dir + 'outputs/dHd_dt.png', 
    units_label='m/a',
    scalar=BIS.sec_per_a
)

BIS.plot_var(
    'dispersed_concentration', working_dir + '/outputs/Cd.png',
    units_label='g sed. / g ice'
)

BIS.plot_var(
    'fringe_growth_rate', working_dir + '/outputs/dHf_dt.png', 
    units_label='m/a',
    scalar=BIS.sec_per_a,
    imshow_args={
        'vmin': 0, 
        'vmax': np.percentile(BIS.grid.at_node['fringe_growth_rate'][:], 99)
    }
)

