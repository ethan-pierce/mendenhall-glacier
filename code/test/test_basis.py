import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_approx_equal

from src.basis import BasalIceStratigrapher

def test_initialize():
    """Test that the model can be initialized with a config file."""
    cfg = './code/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    assert BIS.grid.shape == (5, 5)

    for required in ['ice_thickness', 'sliding_velocity_x', 'sliding_velocity_y', 'basal_water_pressure']:
        assert required in BIS.grid.at_node.keys()

def test_set_value():
    """Test that we can override model values."""
    cfg = './code/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    assert_approx_equal(BIS.grid.at_node['basal_water_pressure'][5] / 1e6, 0.8096, significant=4)

def test_calc_effective_pressure():
    """Test that the model calculates effective pressure correctly."""
    cfg = './code/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    assert_approx_equal(BIS.grid.at_node['effective_pressure'][5] / 1e3, 89.96, significant=4)

def test_calc_shear_stress():
    """Test that the model calculates shear stress correctly."""
    cfg = './code/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    BIS.calc_shear_stress()

    assert_approx_equal(BIS.grid.at_node['basal_shear_stress'][5] / 1e3, 56.21, significant=4)

def test_calc_erosion_rate():
    """Test that the model calculates erosion rates correctly."""
    cfg = './code/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    BIS.calc_erosion_rate()

    assert_approx_equal(BIS.grid.at_node['erosion_rate'][5], 0.0201, significant=4)

def test_calc_melt_rate():
    """Test that the model calculates melt rates correctly."""
    cfg = './code/test/input_file.toml'
    BIS = BasalIceStratigrapher()
    BIS.initialize(config = cfg)

    water_pressure = BIS.grid.at_node['ice_thickness'][:] * BIS.params['ice_density'] * BIS.params['gravity'] * 0.9

    BIS.set_value('basal_water_pressure', water_pressure)

    BIS.calc_effective_pressure()

    BIS.calc_shear_stress()

    BIS.calc_melt_rate()

    assert_approx_equal(BIS.grid.at_node['basal_melt_rate'][5], 0.0092, significant=2)

