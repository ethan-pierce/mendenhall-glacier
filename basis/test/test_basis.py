import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_approx_equal

from basis.src.basis import BasalIceStratigrapher

def test_initialize():
    """Test that the model can be initialized with a config file."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    assert BIS.grid.shape == (5, 5)

    for required in ['ice_thickness', 'sliding_velocity_x', 'sliding_velocity_y', 'basal_water_pressure']:
        assert required in BIS.grid.at_node.keys()

def test_set_value():
    """Test that we can override model values."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    assert_approx_equal(BIS.grid.at_node['basal_water_pressure'][5] / 1e6, 0.8096, significant=4)

def test_calc_effective_pressure():
    """Test that the model calculates effective pressure correctly."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    assert_approx_equal(BIS.grid.at_node['effective_pressure'][5] / 1e3, 89.96, significant=4)

def test_calc_shear_stress():
    """Test that the model calculates shear stress correctly."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    BIS.calc_shear_stress()

    assert_approx_equal(BIS.grid.at_node['basal_shear_stress'][5] / 1e3, 51.10, significant=4)

def test_calc_erosion_rate():
    """Test that the model calculates erosion rates correctly."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    BIS.calc_erosion_rate()

    assert_approx_equal(BIS.grid.at_node['erosion_rate'][5], 6.369e-10, significant=4)

def test_calc_melt_rate():
    """Test that the model calculates melt rates correctly."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    BIS.calc_shear_stress()

    BIS.calc_melt_rate()

    assert_approx_equal(BIS.grid.at_node['basal_melt_rate'][5], 4.616e-10, significant=2)

def test_calc_thermal_gradients():
    """Test that the model calculates thermal gradients correctly."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9
    BIS.set_value('basal_water_pressure', water_pressure)
    BIS.calc_effective_pressure()
    BIS.calc_shear_stress()
    BIS.calc_melt_rate()

    BIS.calc_thermal_gradients()

    assert_approx_equal(BIS.grid.at_node['fringe_thermal_gradient'][5], -0.03099, significant=4)
    assert_approx_equal(BIS.grid.at_node['transition_temperature'][5], 272.9, significant=4)

def test_calc_fringe_growth_rate():
    """Test that the model calculates the fringe growth rate correctly."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9
    BIS.set_value('basal_water_pressure', water_pressure)
    BIS.calc_effective_pressure()
    BIS.calc_shear_stress()
    BIS.calc_melt_rate()

    initial_fringe = np.full(BIS.grid.number_of_nodes, 1e-6)
    BIS.set_value('fringe_thickness', initial_fringe)

    BIS.calc_thermal_gradients()
    BIS.calc_fringe_growth_rate()

    assert_approx_equal(BIS.grid.at_node['fringe_undercooling'][5], 11.22, significant=4)
    assert_approx_equal(BIS.grid.at_node['fringe_saturation'][5], 0.9568, significant=4)
    assert_approx_equal(BIS.grid.at_node['fringe_heave_rate'][5], 1.530e-7, significant=4)
    assert_approx_equal(BIS.grid.at_node['fringe_growth_rate'][5], -0.0241, significant=4)

def test_calc_fringe_growth_rate():
    """Test that the model calculates the fringe growth rate correctly."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9
    BIS.set_value('basal_water_pressure', water_pressure)
    BIS.calc_effective_pressure()
    BIS.calc_shear_stress()
    BIS.calc_melt_rate()

    initial_fringe = np.full(BIS.grid.number_of_nodes, 1e-6)
    BIS.set_value('fringe_thickness', initial_fringe)

    BIS.calc_thermal_gradients()
    BIS.calc_fringe_growth_rate()

    BIS.calc_regelation_rate()

    assert_approx_equal(BIS.grid.at_node['dispersed_layer_gradient'][5], 0.006061, significant=4)
    assert_approx_equal(BIS.grid.at_node['dispersed_layer_growth_rate'][5] * 3.14e7, 0.003728, significant=4)

def test_erode_bedrock():
    "Test that the eroder updates till thickness."
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9
    BIS.set_value('basal_water_pressure', water_pressure)
    
    BIS.erode_bedrock(BIS.sec_per_a)

    assert_approx_equal(BIS.grid.at_node['till_thickness'][5], 0.0201, significant=4)

def test_entrain_sediment():
    "Test that the entrainer updates layer thicknesses."
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9
    BIS.set_value('basal_water_pressure', water_pressure)
    
    till_thickness = np.full(BIS.grid.number_of_nodes, 5.0)
    BIS.set_value('till_thickness', till_thickness)

    fringe_thickness = np.full(BIS.grid.number_of_nodes, 1e-3)
    BIS.set_value('fringe_thickness', fringe_thickness)

    BIS.entrain_sediment(1.0)

    assert_approx_equal(BIS.grid.at_node['till_thickness'][5], 5.0, significant=4)
    assert_approx_equal(BIS.grid.at_node['fringe_thickness'][5], 0.001027, significant=4)
    assert_approx_equal(BIS.grid.at_node['dispersed_layer_thickness'][5], 1.188e-10, significant=4)

def test_advect_sediment():
    """Test that the model correctly advects sediment layers."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9
    BIS.set_value('basal_water_pressure', water_pressure)
    
    till_thickness = np.full(BIS.grid.number_of_nodes, 5.0)
    BIS.set_value('till_thickness', till_thickness)

    fringe_thickness = np.full(BIS.grid.number_of_nodes, 1e-3)
    BIS.set_value('fringe_thickness', fringe_thickness)

    BIS.entrain_sediment(100.0)

    BIS.advect_sediment(100.0)

    assert_approx_equal(BIS.grid.at_node['till_thickness'][5], 4.999, significant=4)
    assert_approx_equal(BIS.grid.at_node['fringe_thickness'][5], 0.003655, significant=4)
    assert_approx_equal(BIS.grid.at_node['dispersed_layer_thickness'][5], 1.188e-8, significant=4)

def test_run_one_step():
    """Test that the model can rull all update routines together."""
    cfg = './basis/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9
    BIS.set_value('basal_water_pressure', water_pressure)
    
    till_thickness = np.full(BIS.grid.number_of_nodes, 5.0)
    BIS.set_value('till_thickness', till_thickness)

    fringe_thickness = np.full(BIS.grid.number_of_nodes, 1e-3)
    BIS.set_value('fringe_thickness', fringe_thickness)

    BIS.run_one_step(1.0)

    assert_approx_equal(BIS.grid.at_node['till_thickness'][5], 5.0, significant=4)
    assert_approx_equal(BIS.grid.at_node['fringe_thickness'][5], 0.001027, significant=4)
    assert_approx_equal(BIS.grid.at_node['dispersed_layer_thickness'][5], 1.188e-10, significant=4)
