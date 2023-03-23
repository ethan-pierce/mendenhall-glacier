import pytest
import numpy as np
from landlab import RasterModelGrid
from numpy.testing import assert_array_equal, assert_approx_equal

from basis.src.tvd_advection import AdvectTVD

@pytest.fixture
def tvd():
    grid = RasterModelGrid((5, 5), 10.)
    grid.add_ones('test', at = 'node')
    grid.at_node['test'][12] = 10

    grid.add_zeros('velocity', at = 'link')
    grid.at_link['velocity'][grid.active_links] = 5.0

    tvd = AdvectTVD(grid, 'test', 'velocity')
    return tvd

def test_set_parallel_links(tvd):
    assert_array_equal(tvd._parallel_links[15], [6, 24])
    assert_array_equal(tvd._parallel_links[20], [19, 21])

def test_find_upwind_links(tvd):
    assert_array_equal(tvd._upwind_links[15], [6])
    assert_array_equal(tvd._upwind_links[20], [19])

def test_calc_courant(tvd):
    courant = tvd.calc_courant(tvd._grid, tvd._vel, 100.0)
    assert_array_equal(courant[15], [50.0])

def test_map_value_linear_upwind(tvd):
    value_at_links = tvd.map_value_to_links_linear_upwind(tvd._grid, tvd._field, tvd._vel)
    assert_array_equal(value_at_links[15], [1])
    assert_array_equal(value_at_links[24], [10])
    assert_array_equal(value_at_links[19], [1])
    assert_array_equal(value_at_links[20], [10])

def test_map_value_lax_wendroff(tvd):
    value_at_links = tvd.map_value_to_links_lax_wendroff(tvd._grid, tvd._field, tvd._vel, 1.0)
    assert_array_equal(value_at_links[15], [3.25])
    assert_array_equal(value_at_links[24], [7.75])
    assert_array_equal(value_at_links[19], [3.25])
    assert_array_equal(value_at_links[20], [7.75])

def test_calc_grad_ratio(tvd):
    ratio = tvd.calc_upwind_to_local_grad_ratio(tvd._grid, tvd._field, tvd._vel, 1.0)
    assert_array_equal(ratio[15], [0.0])
    assert_array_equal(ratio[24], [-1])
    assert_array_equal(ratio[19], [0.0])
    assert_array_equal(ratio[20], [-1])

def test_calc_flux_limiter(tvd):
    ratio = tvd.calc_upwind_to_local_grad_ratio(tvd._grid, tvd._field, tvd._vel, 1.0)
    flux_limiter = tvd.calc_flux_limiter(ratio)
    assert_array_equal(flux_limiter[15], [0.0])
    assert_array_equal(flux_limiter[24], [0.0])
    assert_array_equal(flux_limiter[19], [0.0])
    assert_array_equal(flux_limiter[20], [0.0])

def test_calc_rate_of_change(tvd):
    tvd.calc_rate_of_change(tvd._grid, tvd._field, tvd._vel, 1.0)

    assert_array_equal(tvd._grid.at_link['flux'][15], [5.0])
    assert_array_equal(tvd._grid.at_link['flux'][24], [50.0])
    assert_array_equal(tvd._grid.at_link['flux'][19], [5.0])
    assert_array_equal(tvd._grid.at_link['flux'][20], [50.0])

    assert_array_equal(tvd._grid.at_node['flux_div'][12], [-9.0])
    assert_array_equal(tvd._grid.at_node['flux_div'][13], [4.5])
    assert_array_equal(tvd._grid.at_node['flux_div'][17], [4.5])

def test_update(tvd):
    tvd.update(1.0)
    assert tvd.time_elapsed == 1.0
    assert_array_equal(tvd._grid.at_node['test'][12], 1.0)
    assert_array_equal(tvd._grid.at_node['test'][13], 5.5)
    assert_array_equal(tvd._grid.at_node['test'][17], 5.5)