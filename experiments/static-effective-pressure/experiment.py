import numpy as np
import matplotlib.pyplot as plt

from basis.src.basis import BasalIceStratigrapher

working_dir = './experiments/static-effective-pressure/'

cfg = working_dir + 'input_file.toml'
BIS = BasalIceStratigrapher()
BIS.initialize(cfg)

pressure_scalars = np.arange(0.99, 0.59, -0.01)

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

BIS.write_output(path_to_file)

BIS.plot_var(
    'fringe_thickness', working_dir + '/outputs/Hf_spinup.png', 
    units_label='m',
    imshow_args={'vmin': 1e-3, 'vmax': np.percentile(BIS.grid.at_node['fringe_thickness'][:], 99)}
)

BIS.plot_var(
    'dispersed_layer_thickness', working_dir + '/outputs/Hd_spinup.png', 
    units_label='m'
)

for t in range(20000):
    dt = 0.0001 * BIS.sec_per_a
    BIS.run_one_step(dt, advect=True)

    if t % 1000 == 0:
        print('Completed step #' + str(t))

print('Completed simulation: ' + str(np.round(BIS.time_elapsed / BIS.sec_per_a, 2)) + ' years elapsed.')

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
    imshow_args={'vmin': 0}
)

var = BIS.grid.map_mean_of_links_to_node(BIS.grid.calc_grad_at_link('dispersed_layer_thickness'))
field = np.where(
    BIS.grid.at_node['ice_thickness'][:] > 0.5,
    var,
    np.nan
)
field = np.reshape(field, [BIS.grid.shape[1], BIS.grid.shape[0]])
im = plt.imshow(field)
plt.colorbar(im)
plt.title('gradient in dispersed thickness (m/m)')
plt.savefig(working_dir + 'outputs/gradHd.png')
plt.close('all')

# var = BIS.grid.map_mean_of_links_to_node(BIS.grid.calc_grad_at_link('sliding_velocity_y'))
# field = np.where(
#     BIS.grid.at_node['ice_thickness'][:] > 0.5,
#     var * BIS.sec_per_a,
#     np.nan
# )
# field = np.reshape(field, [BIS.grid.shape[1], BIS.grid.shape[0]])
# im = plt.imshow(field)
# plt.colorbar(im)
# plt.title('sliding velocity y')
# plt.savefig(working_dir + 'outputs/gradUy.png')
# plt.close('all')


var = BIS.grid.map_mean_of_links_to_node(BIS.grid.calc_grad_at_link('fringe_thickness'))
field = np.where(
    BIS.grid.at_node['ice_thickness'][:] > 0.5,
    var,
    np.nan
)
field = np.reshape(field, [BIS.grid.shape[1], BIS.grid.shape[0]])
im = plt.imshow(field)
plt.colorbar(im)
plt.title('gradient of fringe thickness (m/m)')
plt.savefig(working_dir + 'outputs/gradHf.png')
plt.close('all')

