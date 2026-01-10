"""
Electro-thermal solution for Transition Edge Sensors (TES) read out with DfMux.

Author: Tijmen de Haan
Email: <tijmen.dehaan@gmail.com>
Date: 15 May 2024

This module provides functions for simulating the thermal and electrical behavior of 
Transition Edge Sensors (TES) based on the MNTES model. As of May 2024, this is
being drafted as a paper for the SPIE Astronomical Telescopes and Instrumentation
conference proceedings.

Functions:
- r_frac: fractional resistance as a function of temperature
- r: resistance as a function of temperature
- alpha: logarithmic temperature sensitivity of the resistance
- loop_gain: Computes the ETF loop gain using the MNTES model.
- responsivity: Computes the responsivity of the TES using the MNTES model.
- power_balance_eq: Calculates the deviation from power balance. This should be
                    zeroed to find the equilibrium temperature of the TES.
- calc_tes: Computes various TES parameters given input parameters.
- calc_nonlinearity: Computes TES nonlinearity.

Usage:
- The `calc_tes` function is the main entry point for solving the power balance equation.
- The `calc_nonlinearity` function can be used to get the leading-order nonlinearity by finite difference.

Dependencies:
- numpy
- scipy.optimize
- numba
"""

import numpy as np
from scipy.optimize import brentq
from numba import jit

@jit(nopython=True)
def r_frac(temperature, t_c, transition_width):
    return (np.arctan((temperature - t_c) / (transition_width / 2.0)) / (np.pi / 2) + 1) / 2

@jit(nopython=True)
def r(temperature, r_normal, t_c, transition_width):
    return r_normal * r_frac(temperature, t_c, transition_width)

@jit(nopython=True)
def alpha(temperature, r_normal, t_c, transition_width):
    return temperature / r(temperature, r_normal, t_c, transition_width) * (
        r_normal
        * (transition_width / 2.0)
        / (np.pi * ((temperature - t_c) ** 2 + (transition_width / 2.0) ** 2))
    )

@jit(nopython=True)
def loop_gain(v_thev, k, n_index, t, r_t, z_thev, alpha_t, beta_t=0):
    """
    Implements the loop gain equation from the MNTES paper.
    $\mathcal{L} = \frac{\alpha V_\mathrm{Th\acute{e}v}^2}{K n T^n R} \frac{R^2 \left( R^2 - \left|  z_\mathrm{Th\acute{e}v} \right |^2 \right)}{\left| R + z_\mathrm{Th\acute{e}v} \right|^2 \left( \left| R + z_\mathrm{Th\acute{e}v} \right|^2 + \beta R (R +  \Re{(z_\mathrm{Th\acute{e}v})}) \right) }$
    """
    loop_gain_ideal = alpha_t*v_thev**2/(k*n_index*t**n_index*r_t)
    abs_z_thev = np.sqrt(z_thev.real**2 + z_thev.imag**2)
    abs_z_total = np.sqrt((r_t + z_thev.real)**2 + z_thev.imag**2)
    loop_gain_excess_numerator = r_t**2 * (r_t**2 - abs_z_thev**2)
    loop_gain_excess_denominator = abs_z_total**2 * (abs_z_total**2 + beta_t * r_t * (r_t + z_thev.real))
    return loop_gain_ideal * loop_gain_excess_numerator / loop_gain_excess_denominator

@jit(nopython=True)
def responsivity(v_thev, loop_gain, r_t, z_thev):
    """
    Implements the responsivity equation from the MNTES paper.
    S \equiv \frac{\delta I}{\delta P_\mathrm{opt}} = - \frac{\sqrt{2}}{V_\mathrm{Th\acute{e}v}} \frac{\mathcal{L}}{\mathcal{L} + 1} \left( 1 + 2 z_\mathrm{Th\acute{e}v}^\star \frac{R + \Re{(z_\mathrm{Th\acute{e}v})}}{R^2 - \left| z_\mathrm{Th\acute{e}v} \right|^2 } \right) } \ .
    """
    abs_z_thev_squared = z_thev.real**2 + z_thev.imag**2
    responsivity_factor = -1 / v_thev * loop_gain / (loop_gain + 1)
    responsivity_excess = 1 + 2 * z_thev.conjugate() * (r_t + z_thev.real) / (r_t**2 - abs_z_thev_squared)
    return responsivity_factor * responsivity_excess

@jit(nopython=True)
def power_balance_eq(temperature, p_loading, v_thev, k, n_index, t_bath, r_normal, t_c, transition_width, z_thev):
    r_t = r(temperature, r_normal, t_c, transition_width)
    return (
        p_loading
        + r_t * v_thev**2 / np.abs(r_t + z_thev) ** 2
        - k * (temperature**n_index - t_bath**n_index)
    )

def calc_tes(
    t_c=180e-3,
    transition_width=0.001143,
    r_normal=1.0,
    p_loading=0.5e-12,
    t_bath=100e-3,
    z_thev=0.05 + 0.05j,
    n_index=3.6,
    v_thev=0.8e-6,
    p_sat_for_g=1.25e-12,
    debug=False,
):
    k = p_sat_for_g / (t_c**n_index - t_bath**n_index)
    
    def power_balance_eq_wrapper(temperature):
        return power_balance_eq(temperature, p_loading, v_thev, k, n_index, t_bath, r_normal, t_c, transition_width, z_thev)
    
    t_0 = brentq(power_balance_eq_wrapper, t_c, 2*t_c)
    
    r_0 = r(t_0, r_normal, t_c, transition_width)
    I_0 = v_thev / (r_0 + z_thev)
    alpha_0 = alpha(t_0, r_normal, t_c, transition_width)
    loop_gain_0 = loop_gain(v_thev, k, n_index, t_0, r_0, z_thev, alpha_0)
    responsivity_0 = responsivity(v_thev, loop_gain_0, r_0, z_thev)
    p_electrical = r_0 * v_thev**2 / np.abs(r_0 + z_thev) ** 2
    
    if debug:
        # Add debugging code here if needed
        pass
    
    return {
        "t": t_0,
        "r": r_0,
        "i": I_0,
        "l": loop_gain_0,
        "s": responsivity_0,
        "p_electrical": p_electrical,
        "p_sat_for_g": p_sat_for_g,
        "alpha": alpha_0,
        "k": k,
    }

def calc_nonlinearity(param, value, fiducial_params, r_frac):
    params = fiducial_params.copy()
    if param.startswith('z_thev'):
        z_real, z_imag = params['z_thev'].real, params['z_thev'].imag
        params['z_thev'] = complex(value if param == 'z_thev_real' else z_real,
                                   value if param == 'z_thev_imag' else z_imag)
    else:
        params[param] = value

    v = 1e-6
    r = calc_tes(v_thev=v, **params)['r']
    r_target = r_frac * params['r_normal']
    
    while r > r_target:
        v -= 0.1e-9
        r = calc_tes(v_thev=v, **params)['r']

    delta_p = 0.03 * 5e-13
    S0 = calc_tes(v_thev=v, **params)['s']
    params['p_loading'] += delta_p
    S1 = calc_tes(v_thev=v, **params)['s']
    
    return S0, (S1 - S0) / delta_p

if __name__ == "__main__":
    print(calc_tes(debug=True))
