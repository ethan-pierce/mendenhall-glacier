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
n_years = 100

results = pd.DataFrame(
    columns = [
        'variable',
        'value',
        'fringe_thickness',
        'fringe_sedflux',
        'dispersed_layer_thickness',
        'dispersed_layer_growth_rate',
        'dispersed_sedflux'
    ]
)

for idx, info in parameters.iterrows():
    experiments = np.linspace(info.values[1], info.values[2], n_runs)
    results.append
    
    for exp in experiments:
        BIS = BasalIceStratigrapher()
        BIS.initialize(working_dir + 'default.toml')

        BIS.set_value(info.variable, np.full(BIS.grid.number_of_nodes, exp))
        BIS.set_value('till_thickness', np.full(BIS.grid.number_of_nodes, 20))
        BIS.set_value('fringe_thickness', np.full(BIS.grid.number_of_nodes, 1e-3))

        BIS.calc_shear_stress()
        BIS.calc_melt_rate()

        for t in range(100):
            BIS.entrain_sediment(t * 1e-2)

        for t in range(100):
            BIS.entrain_sediment(t)

        for t in range(100):
            BIS.entrain_sediment(100)

        for t in range(10000):
            dt = BIS.sec_per_a / 100
            BIS.run_one_step(dt, erode = False, advect = False)
            BIS.time_elapsed += dt

        fringe_sedflux = (
            BIS.grid.at_node['fringe_thickness'][4] * 
            BIS.grid.at_node['sliding_velocity_x'][4] * 
            (1 - BIS.params['frozen_fringe_porosity']) *
            BIS.params['sediment_density']
        )

        dispersed_sedflux = (
            BIS.grid.at_node['dispersed_layer_thickness'][4] * 
            BIS.grid.at_node['sliding_velocity_x'][4] * 
            BIS.grid.at_node['dispersed_concentration'][4] *
            BIS.params['sediment_density']
        )

        result = [
            info.variable,
            exp,
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

    break

results.to_csv(working_dir + 'results.csv')