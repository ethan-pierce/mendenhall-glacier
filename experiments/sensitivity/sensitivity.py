import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

working_dir = './experiments/sensitivity/'
parameters = pd.read_csv(
    working_dir + 'sensitivity-parameters.txt',
    header = 0,
    sep = ',',
    usecols = [0, 1, 2, 3, 4]
)

n_runs = 20
n_years = 100

print(parameters)

for idx, info in parameters.iterrows():
    experiments = np.linspace(info.values[1], info.values[2], n_runs)
    
    for exp in experiments:
        BIS = BasalIceStratigrapher()
        BIS.initialize(working_dir + 'default.toml')
        