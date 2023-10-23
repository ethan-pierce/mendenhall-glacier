import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from basis.src.basis import BasalIceStratigrapher
from basis.src.tvd_advection import AdvectTVD

scenarios = ['fast', 'slow']
Ns = [0.95, 0.9, 0.8, 0.6]

for scenario in scenarios:
    input_file = './experiments/static-effective-pressure/' + scenario + '_input_file.toml'
    output_dir = './experiments/static-effective-pressure/outputs/' + scenario

    for N in Ns:
        fringe_fname = 'fringe_Pw_' + str(int(N*100)) + '.txt'
        disp_fname = 'disp_Pw_' + str(int(N*100)) + '.txt'

        # Initialize the model
        model = BasalIceStratigrapher()
        model.initialize(input_file)

        basal_water_pressure = N * (
            model.grid.at_node['ice_thickness'] * model.params['ice_density'] * model.params['gravity']
        )
        model.set_value('basal_water_pressure', basal_water_pressure)

        model.calc_effective_pressure()
        model.calc_shear_stress()
        model.calc_erosion_rate()
        model.calc_melt_rate()

        # Identify terminus nodes
        dx = model.grid.dx
        dy = model.grid.dy
        bounds = [50 * dx, 100 * dx, 0 * dy, 35 * dy]
        model.identify_terminus(bounds, depth = 3)

        outflow = model.grid.add_field(
            'terminus_velocity', 
            (
                model.grid.at_node['is_terminus'][:]
                * model.grid.at_node['sliding_velocity_magnitude'][:]
            ),
            at = 'node'
        )

        mask = model.grid.at_node['ice_thickness'] > 0.1

        # Start with initial sediment package
        initial_till = np.full(model.grid.number_of_nodes, 40)
        model.set_value('till_thickness', initial_till)

        initial_fringe = np.full(model.grid.number_of_nodes, 1e-3)
        model.set_value('fringe_thickness', initial_fringe)

        initial_dispersed = np.full(model.grid.number_of_nodes, 1e-9)
        model.set_value('dispersed_layer_thickness', initial_dispersed)

        # Spin-up the sediment entrainment module
        for t in range(100):
            model.entrain_sediment(t * 1e-2)
            model.time_elapsed += t
            
        for t in range(100):
            model.entrain_sediment(t)
            model.time_elapsed += t

        for t in range(100):
            model.entrain_sediment(100)
            model.time_elapsed += 100

        for t in range(5000):
            dt = model.sec_per_a / 1000
            model.entrain_sediment(dt)
            model.time_elapsed += dt

        for t in range(10000):
            dt = model.sec_per_a / 500
            model.entrain_sediment(dt)
            model.time_elapsed += dt

            if t % 1000 == 0:
                print('Spin-up step #' + str(t))
            
        print('Completed spin-up: ' + str(np.round(model.time_elapsed / model.sec_per_a, 2)) + ' years elapsed.')

        np.savetxt(output_dir + '/spinup/' + fringe_fname, model.grid.at_node['fringe_thickness'])
        np.savetxt(output_dir + '/spinup/' + disp_fname, model.grid.at_node['dispersed_layer_thickness'])

        #################
        # Advection model
        #################

        ux = model.grid.map_mean_of_link_nodes_to_link('sliding_velocity_x')
        uy = model.grid.map_mean_of_link_nodes_to_link('sliding_velocity_y')

        velocity = np.zeros(model.grid.number_of_links)
        velocity = np.where(
            np.isin(np.arange(model.grid.number_of_links), model.grid.horizontal_links),
            ux,
            uy
        )
        model.grid.add_field('velocity_links', velocity, at = 'link', clobber = True)

        advect_fringe = AdvectTVD(model.grid, 'fringe_thickness', 'velocity_links')
        advect_disp = AdvectTVD(model.grid, 'dispersed_layer_thickness', 'velocity_links')
        advect_conc = AdvectTVD(model.grid, 'fringe_concentration', 'velocity_links')
        
        dt = 1e6

        for advector in [advect_fringe, advect_disp, advect_conc]:
            test_courant = np.max(np.abs(advector.calc_courant(advector._grid, advector._vel, dt)))
            print('CFL condition = ' + str(test_courant))
            if test_courant > 1:
                print('Advection likely unstable under CFL condition.')
            
        end_year = 300
        n_steps = int(np.ceil(model.sec_per_a * end_year / dt))
        Qfs = []
        Qds = []

        fringe_layers = []
        disp_layers = []
        concentrations = []

        old_fringe = model.grid.at_node['fringe_thickness'][:]
        
        for i in range(n_steps):
            
            advect_fringe.update(dt)
            advect_disp.update(dt)
            advect_conc.update(dt)

            model.entrain_sediment(dt, clip = 99)
            model.time_elapsed += dt
            
            # Introduce outflow condition
            Hd = model.grid.at_node['dispersed_layer_thickness'][:]
            Hf = model.grid.at_node['fringe_thickness'][:]

            Hd[mask][outflow[mask] > 0] -= (
                (Hd[mask][outflow[mask] > 0] / model.grid.dx) * outflow[mask][outflow[mask] > 0]
            ) * dt
            Hd[mask][Hd[mask] < 1e-9] = 1e-9
            
            # Add a small amount of diffusion for stability
            cut = 99
            
            kernel = np.ones((3, 3)) / 8
            kernel[1, 1] = 0
            convolution = np.ravel(convolve2d(Hf.reshape(model.grid.shape), kernel, mode = 'same'))
            Hf[Hf > np.percentile(Hf[mask], cut)] = convolution[Hf > np.percentile(Hf[mask], cut)]
            
            convolution = np.ravel(convolve2d(Hd.reshape(model.grid.shape), kernel, mode = 'same'))
            Hd[Hd > np.percentile(Hd[mask], cut)] = convolution[Hd > np.percentile(Hd[mask], cut)]

            # Set values outside of the glacier to (almost) zero
            Hd[~mask] = 1e-9
            Hf[~mask] = 1e-6

            # Track fringe concentration near the terminus
            terminus_cf = np.where(
                    model.grid.at_node['adjacent_to_terminus'],
                    model.grid.at_node['fringe_concentration'],
                    np.nan
                )

            if i % 30 == 0:
                fringe_layers.append(model.grid.at_node['fringe_thickness'][:])
                disp_layers.append(model.grid.at_node['dispersed_layer_thickness'][:])
                concentrations.append(np.nanmean(terminus_cf))

                imshow_grid(model.grid, 'fringe_thickness')
                plt.show()

            if i % 100 == 0:
                print('Completed step ' + str(i))
                
                __, Qf, Qd = model.calc_sediment_flux()
                Qfs.append(Qf)
                Qds.append(Qd)
                
        print('Completed simulation with Pw = ' + str(N) + ' * Pi.')

        np.savetxt(output_dir + '/concentration/' + fringe_fname, concentrations)    

        np.savetxt(output_dir + '/history/' + fringe_fname, fringe_layers)    
        np.savetxt(output_dir + '/history/' + disp_fname, disp_layers)    
            
        np.savetxt(output_dir + '/spatial/' + fringe_fname, model.grid.at_node['fringe_thickness'][:])
        np.savetxt(output_dir + '/spatial/' + disp_fname, model.grid.at_node['dispersed_layer_thickness'][:])

        np.savetxt(output_dir + '/flux/' + fringe_fname, Qfs)
        np.savetxt(output_dir + '/flux/' + disp_fname, Qds)