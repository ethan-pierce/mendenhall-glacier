import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from basis.src.basis import BasalIceStratigrapher

working_dir = './experiments/sensitivity/'
parameters = pd.read_csv(
    working_dir + 'sensitivity-parameters.txt',
    header = 0,
    sep = ',',
    usecols = [0, 1, 2, 3, 4]
)

n_runs = 20
n_years = 200

results = pd.DataFrame(
    columns = [
        'variable',
        'value',
        'entry_pressure',
        'basal_shear_stress',
        'basal_melt_rate',
        'fringe_heave_rate',
        'fringe_growth_rate',
        'fringe_thickness',
        'fringe_sedflux',
        'dispersed_layer_thickness',
        'dispersed_layer_growth_rate',
        'dispersed_sedflux'
    ]
)

for idx, info in parameters.iterrows():
    if info.variable == 'default':
        experiments = [0]
    elif info.variable in ['effective_pressure', 'sliding_velocity_x']:
        experiments = np.geomspace(info.values[1], info.values[2], n_runs)
    else:
        experiments = np.linspace(info.values[1], info.values[2], n_runs)        
    
    for exp in experiments:
        BIS = BasalIceStratigrapher()
        BIS.initialize(working_dir + 'default.toml')

        try:
            BIS.set_value(info.variable, np.full(BIS.grid.number_of_nodes, exp))
        except:
            BIS.params[info.variable] = exp

        if info.variable == 'sliding_velocity_x':
            BIS.grid.at_node['sliding_velocity_x'] *= (1 / BIS.sec_per_a)

            BIS.grid.at_node['sliding_velocity_magnitude'][:] = np.abs(
                np.sqrt(BIS.grid.at_node['sliding_velocity_x'][:]**2 + BIS.grid.at_node['sliding_velocity_y'][:]**2)
            )

        if info.variable == 'coupled_grain_size':
            BIS.params['pore_throat_radius'] = exp
            BIS.params['till_grain_radius'] = exp

        BIS.set_value('till_thickness', np.full(BIS.grid.number_of_nodes, 20))
        BIS.set_value('fringe_thickness', np.full(BIS.grid.number_of_nodes, 1e-3))

        if info.variable != 'effective_pressure':
            BIS.calc_effective_pressure()

        if info.variable == 'basal_shear_stress':
            BIS.set_value('basal_shear_stress', np.full(BIS.grid.number_of_nodes, exp))
        else:
            BIS.calc_shear_stress()

        BIS.calc_melt_rate()
        
        for t in range(100):
            BIS.entrain_sediment(t * 1e-2)

        for t in range(100):
            BIS.entrain_sediment(t)

        for t in range(100):
            BIS.entrain_sediment(100)

        for t in range(100):
            dt = BIS.sec_per_a / 10000
            BIS.run_one_step(dt, erode = False, advect = False)

        steps_per_year = 100
        for t in range(int(n_years * steps_per_year)):
            dt = BIS.sec_per_a / steps_per_year
            BIS.run_one_step(dt, erode = False, advect = False)

        BIS.grid.at_node['dispersed_layer_thickness'][:] += (
            BIS.grid.at_node['dispersed_layer_growth_rate'][:] * (250 - n_years) * BIS.sec_per_a
        )

        fringe_sedflux = (
            BIS.grid.at_node['fringe_thickness'][4] * 
            BIS.grid.at_node['sliding_velocity_x'][4] * 
            BIS.sec_per_a * 
            (1 - BIS.params['frozen_fringe_porosity']) *
            BIS.params['sediment_density']
        )

        dispersed_sedflux = (
            BIS.grid.at_node['dispersed_layer_thickness'][4] * 
            BIS.grid.at_node['sliding_velocity_x'][4] * 
            BIS.sec_per_a *
            BIS.grid.at_node['dispersed_concentration'][4] *
            BIS.params['sediment_density']
        )

        result = [
            info.variable,
            exp,
            BIS.params['entry_pressure'],
            BIS.grid.at_node['basal_shear_stress'][4],
            BIS.grid.at_node['basal_melt_rate'][4],
            BIS.grid.at_node['fringe_heave_rate'][4],
            BIS.grid.at_node['fringe_growth_rate'][4],
            BIS.grid.at_node['fringe_thickness'][4],
            fringe_sedflux,
            BIS.grid.at_node['dispersed_layer_thickness'][4],
            BIS.grid.at_node['dispersed_layer_growth_rate'][4],
            dispersed_sedflux
        ]

        results.loc[len(results.index + 1)] = result

        print('Completed experiment: ' + str(info.variable) + ' = ' + str(exp))
        print('------------------------------------------')
        print(results.tail())
        print('------------------------------------------')

results.to_csv(working_dir + '/outputs/results.csv')