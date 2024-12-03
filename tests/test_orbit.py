import pytest
import numpy as np
from skaero.orbitalmechanics.orbit import (
    periapsis, apoapsis, semi_major_axis, semi_minor_axis,
    eccentricity, distance_from_center, velocity,
    specific_angular_momentum, orbital_energy
)

# Constants for testing
MU_EARTH = 398600.4418  # Earth's gravitational parameter (km³/s²)
TOL = 1e-6  # Tolerance for floating point comparisons

@pytest.fixture
def circular_orbit():
    """Fixture for a circular orbit (e=0)"""
    return {
        'a': 7000.0,  # km
        'e': 0.0,
        'mu': MU_EARTH
    }

@pytest.fixture
def elliptical_orbit():
    """Fixture for an elliptical orbit (0<e<1)"""
    return {
        'a': 26600.0,  # km
        'e': 0.74,
        'mu': MU_EARTH
    }

def test_periapsis():
    """Test periapsis calculation"""
    # Test circular orbit
    assert periapsis(7000.0, 0.0) == pytest.approx(7000.0)
    
    # Test elliptical orbit
    assert periapsis(26600.0, 0.74) == pytest.approx(6916.0)

def test_apoapsis():
    """Test apoapsis calculation"""
    # Test circular orbit
    assert apoapsis(7000.0, 0.0) == pytest.approx(7000.0)
    
    # Test elliptical orbit
    assert apoapsis(26600.0, 0.74) == pytest.approx(46284.0)

def test_semi_major_axis():
    """Test semi-major axis calculation"""
    rp = 6916.0
    ra = 46284.0
    expected_a = (ra + rp) / 2
    assert semi_major_axis(ra, rp) == pytest.approx(expected_a)

def test_semi_minor_axis():
    """Test semi-minor axis calculation"""
    a = 26600.0
    e = 0.74
    expected_b = a * np.sqrt(1 - e**2)
    assert semi_minor_axis(a, e) == pytest.approx(expected_b)

def test_eccentricity():
    """Test eccentricity calculation"""
    a = 26600.0
    b = a * np.sqrt(1 - 0.74**2)
    assert eccentricity(a, b) == pytest.approx(0.74, rel=TOL)

def test_distance_from_center():
    """Test distance calculation at various true anomalies"""
    a = 26600.0
    e = 0.74
    
    # At periapsis (θ = 0°)
    assert distance_from_center(a, e, 0) == pytest.approx(periapsis(a, e))
    
    # At apoapsis (θ = 180°)
    assert distance_from_center(a, e, 180) == pytest.approx(apoapsis(a, e))

def test_velocity(circular_orbit, elliptical_orbit):
    """Test velocity calculation"""
    # Test circular orbit velocity (should be constant)
    r = circular_orbit['a']
    v_circ = velocity(circular_orbit['mu'], circular_orbit['a'], r)
    expected_v_circ = np.sqrt(circular_orbit['mu'] / r)
    assert v_circ == pytest.approx(expected_v_circ)
    
    # Test elliptical orbit velocities at periapsis and apoapsis
    rp = periapsis(elliptical_orbit['a'], elliptical_orbit['e'])
    ra = apoapsis(elliptical_orbit['a'], elliptical_orbit['e'])
    
    v_peri = velocity(elliptical_orbit['mu'], elliptical_orbit['a'], rp)
    v_apo = velocity(elliptical_orbit['mu'], elliptical_orbit['a'], ra)
    
    assert v_peri > v_apo  # Velocity should be highest at periapsis

def test_specific_angular_momentum(circular_orbit, elliptical_orbit):
    """Test specific angular momentum calculation"""
    # For circular orbit, h should be constant
    h_circ = specific_angular_momentum(
        circular_orbit['mu'],
        circular_orbit['a'],
        circular_orbit['e']
    )
    expected_h_circ = np.sqrt(circular_orbit['mu'] * circular_orbit['a'])
    assert h_circ == pytest.approx(expected_h_circ)

def test_orbital_energy(circular_orbit, elliptical_orbit):
    """Test specific orbital energy calculation"""
    # Test circular orbit energy
    e_circ = orbital_energy(circular_orbit['mu'], circular_orbit['a'])
    expected_e_circ = -circular_orbit['mu'] / (2 * circular_orbit['a'])
    assert e_circ == pytest.approx(expected_e_circ)
    
    # Test elliptical orbit energy
    e_ellip = orbital_energy(elliptical_orbit['mu'], elliptical_orbit['a'])
    expected_e_ellip = -elliptical_orbit['mu'] / (2 * elliptical_orbit['a'])
    assert e_ellip == pytest.approx(expected_e_ellip)

def test_edge_cases():
    """Test edge cases and error conditions"""
    # Test with zero semi-major axis
    with pytest.raises(ZeroDivisionError):
        velocity(MU_EARTH, 0, 1000)
    
    # Test with negative eccentricity
    with pytest.raises(ValueError):
        assert np.isnan(semi_minor_axis(1000, -0.5)) 