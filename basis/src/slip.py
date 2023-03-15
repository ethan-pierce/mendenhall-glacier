import numpy as np
import matplotlib.pyplot as plt

def calc_slip_coefficient(
    effective_viscosity=3.2e12, 
    clast_radius=15e-3,
    exposed_radius_fraction=0.25,
    clapeyron_slope=7.4e-8,
    ice_thermal_conductivity=2.55,
    volumetric_latent_heat=3e8,
    bearing_capacity_factor=33,
    till_strength_coeff=0.1
): 
    k0 = (2 * np.pi) / (4 * clast_radius)

    regelation_coeff = clapeyron_slope * ice_thermal_conductivity / volumetric_latent_heat

    first_term = (
        1 / (effective_viscosity * (clast_radius * exposed_radius_fraction)**2 * k0**3)
    )

    second_term = (
        (4 * regelation_coeff) / ((clast_radius * exposed_radius_fraction)**2 * k0)
    )

    numerator = (first_term + second_term) * bearing_capacity_factor

    denominator = 2 + bearing_capacity_factor * till_strength_coeff

    return numerator / denominator

def calc_threshold_velocity(
    effective_pressure,
    **kwargs
):
    C = calc_slip_coefficient(**kwargs)
    return C * effective_pressure

def calc_slip_law(
    effective_pressure,
    sliding_velocity,
    friction_angle=32,
    slip_exponent=5,
    **kwargs
):
    threshold = calc_threshold_velocity(effective_pressure, **kwargs)
    tau = (
        effective_pressure * np.tan(np.deg2rad(friction_angle)) *
        (sliding_velocity / (sliding_velocity + threshold))**(1/slip_exponent)
    )
    return tau