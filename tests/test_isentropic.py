# coding: utf-8

"""
Tests of the isentropic package.

References
----------
1. NACA-TR-1135 http://hdl.handle.net/2060/19930091059
2. Anderson, J.D.: "Modern compressible flow", 3rd edition.

"""

from __future__ import division, absolute_import

import numpy as np
import numpy.testing

import pytest

from skaero.gasdynamics import isentropic


def test_isentropic_flow_default_gamma():
    """Test default gamma value is 1.4"""
    flow = isentropic.IsentropicFlow()
    assert flow.gamma == 1.4


@pytest.mark.parametrize("mach,expected", [
    (1.1, np.radians(65.38)),
    (1.38, np.radians(46.44)),
    (2.05, np.radians(29.20)),
    (3.0, np.radians(19.47)),
    (np.inf, 0.0)
])
def test_mach_angle(mach, expected):
    """Test Mach angle calculations"""
    assert np.isclose(isentropic.mach_angle(mach), expected, rtol=1e-3)


def test_mach_angle_subsonic_error():
    """Test error raised for subsonic Mach numbers"""
    with pytest.raises(ValueError, match="Mach number must be supersonic"):
        isentropic.mach_angle(0.8)


@pytest.mark.parametrize("mach,area_ratio,expected", [
    (1.0, 1.0, 1.0),
    (1.24, 1.043, 1.043),
    (2.14, 1.902, 1.902)
])
def test_area_ratio(isentropic_flow, mach, area_ratio, expected):
    """Test area ratio calculations"""
    assert np.isclose(isentropic_flow.A_Astar(mach), expected, rtol=1e-3)
