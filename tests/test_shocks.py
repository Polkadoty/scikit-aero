# coding: utf-8

"""
Tests of the shocks package.

References
----------
1. NACA-TR-1135 http://hdl.handle.net/2060/19930091059
2. NASA-TN-D-2221 http://hdl.handle.net/2060/19640007246
3. Normal shock tables on Wikipedia
   http://en.wikipedia.org/wiki/Normal_shock_tables

"""

from __future__ import division, absolute_import

import numpy as np
import numpy.testing

import pytest

from skaero.gasdynamics import shocks


def test_normal_shock_constructor():
    gamma = 1.4
    M_1 = 2.0
    shocks.Shock(M_1=M_1, gamma=gamma)


def test_normal_shock_default_specific_heat_ratio():
    ns = shocks.Shock(M_1=2.0)
    np.testing.assert_equal(ns.gamma, 7 / 5)


@pytest.mark.parametrize("mach_1,expected_mach_2", [
    (1.5, 0.7011),
    (1.8, 0.6165),
    (2.1, 0.5613),
    (3.0, 0.4752)
])
def test_normal_shock_mach(mach_1, expected_mach_2):
    """Test post-shock Mach number calculations"""
    shock = shocks.Shock(M_1=mach_1, gamma=1.4)
    assert np.isclose(shock.M_2, expected_mach_2, rtol=1e-4)


def test_normal_shock_fails_subsonic_M_1():
    with pytest.raises(ValueError):
        shocks.Shock(M_1=0.8)


@pytest.mark.parametrize("mach_1,ratios", [
    (1.5, {'p': 2.4583, 'rho': 1.8621, 'T': 1.3202}),
    (1.8, {'p': 3.6133, 'rho': 2.3592, 'T': 1.5316}),
    (2.1, {'p': 4.9783, 'rho': 2.8119, 'T': 1.7705}),
    (3.0, {'p': 10.3333, 'rho': 3.8571, 'T': 2.6790})
])
def test_normal_shock_ratios(mach_1, ratios):
    """Test shock property ratios"""
    shock = shocks.Shock(M_1=mach_1, gamma=1.4)
    assert np.isclose(shock.p2_p1, ratios['p'], rtol=1e-4)
    assert np.isclose(shock.rho2_rho1, ratios['rho'], rtol=1e-4)
    assert np.isclose(shock.T2_T1, ratios['T'], rtol=1e-4)


def test_normal_shock_infinite_limit():
    gamma = 1.4
    ns = shocks.Shock(M_1=np.inf, gamma=gamma)
    np.testing.assert_almost_equal(
        ns.M_2, np.sqrt((gamma - 1) / 2 / gamma), decimal=3)
    np.testing.assert_almost_equal(
        ns.rho2_rho1, (gamma + 1) / (gamma - 1), decimal=3)


def test_normal_shock_zero_deflection():
    ns = shocks.Shock(M_1=2.0)
    assert ns.theta == 0.0


def test_error_max_deflection():
    with pytest.raises(ValueError):
        shocks.Shock(M_1=5, theta=np.radians(50))


def test_error_mach_angle():
    with pytest.raises(ValueError):
        shocks.Shock(M_1=5, beta=np.radians(10))


def test_max_deflection():
    M_1_list = [1.4, 1.9, 2.2, 3.0, np.inf]
    theta_max_degrees_list = [
        9.427,
        21.17,
        26.1,
        34.07,
        45.58
    ]
    beta_theta_max_degrees_list = [
        67.72,
        64.78,
        64.62,
        65.24,
        67.79
    ]
    angles_pairs_list = [shocks.max_deflection(M_1, 1.4) for M_1 in M_1_list]

    for i in range(len(angles_pairs_list)):
        np.testing.assert_almost_equal(
            angles_pairs_list[i][0], np.radians(theta_max_degrees_list[i]),
            decimal=3)
        np.testing.assert_almost_equal(
            angles_pairs_list[i][1],
            np.radians(beta_theta_max_degrees_list[i]), decimal=3)


def test_parallel_shock_infinity_mach():
    M_1 = np.inf
    beta = 0.0
    os = shocks.Shock(M_1=M_1, beta=beta)
    assert os.M_1n == 0.0
    assert np.isfinite(os.theta)


def test_oblique_shock_from_deflection_angle():
    # Anderson, example 4.1
    # Notice that only graphical accuracy is achieved in the original example
    M_1 = 3.0
    theta = np.radians(20.0)
    os = shocks.Shock(M_1=M_1, theta=theta, weak=True)

    np.testing.assert_almost_equal(os.M_1n, 1.839, decimal=2)
    np.testing.assert_almost_equal(os.M_2n, 0.6078, decimal=2)
    np.testing.assert_almost_equal(os.M_2, 1.988, decimal=1)
    np.testing.assert_almost_equal(os.p2_p1, 3.783, decimal=1)
    np.testing.assert_almost_equal(os.T2_T1, 1.562, decimal=2)
