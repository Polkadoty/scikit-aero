import pytest
import numpy as np
import os
from datetime import datetime

def wing_loading(weight, wing_area):
    """Calculate wing loading (W/S)"""
    return weight / wing_area

def aspect_ratio(wingspan, wing_area):
    """Calculate wing aspect ratio"""
    return wingspan**2 / wing_area

def thrust_to_weight(thrust, weight):
    """Calculate thrust-to-weight ratio"""
    return thrust / weight

def empty_weight_fraction(mtow, empty_weight):
    """Calculate empty weight fraction"""
    return empty_weight / mtow

def fuel_fraction(fuel_weight, mtow):
    """Calculate fuel fraction"""
    return fuel_weight / mtow

def wing_reynolds_number(velocity, chord, altitude=0):
    """
    Calculate Reynolds number for wing
    
    Parameters
    ----------
    velocity : float
        Airspeed in m/s
    chord : float
        Wing chord in m
    altitude : float
        Altitude in m (default sea level)
        
    Returns
    -------
    float
        Reynolds number
    """
    # Standard atmosphere properties at sea level
    rho = 1.225  # kg/m³
    mu = 1.789e-5  # kg/m·s
    
    # Very basic altitude effects (simplified)
    if altitude > 0:
        rho *= np.exp(-altitude/7400)  # Approximate density variation
        
    return (rho * velocity * chord) / mu

def oswald_efficiency(aspect_ratio, sweep_angle=0):
    """
    Calculate Oswald efficiency factor
    Using simplified approximation
    """
    sweep_rad = np.radians(sweep_angle)
    return 1.78 * (1 - 0.045 * aspect_ratio**0.68) - 0.64

def induced_drag_coefficient(cl, aspect_ratio, oswald_efficiency):
    """Calculate induced drag coefficient"""
    return cl**2 / (np.pi * aspect_ratio * oswald_efficiency)

def wetted_area_ratio(fuselage_length, fuselage_diameter, wing_area, tail_area=None):
    """
    Calculate wetted area ratio
    Simplified estimation using component buildup method
    """
    fuselage_wetted = np.pi * fuselage_diameter * fuselage_length
    wing_wetted = 2.1 * wing_area  # Approximate both sides plus thickness effects
    tail_wetted = 2.1 * (tail_area if tail_area else 0.2 * wing_area)  # Typical tail sizing
    return (fuselage_wetted + wing_wetted + tail_wetted) / wing_area

def zero_lift_drag_coefficient(mach, wetted_area_ratio, form_factor=1.2):
    """
    Calculate zero-lift drag coefficient
    Using flat-plate skin friction with form factor
    """
    # Simplified skin friction coefficient (turbulent flat plate)
    cf = 0.455 / (np.log10(reynolds_number(mach))**2.58)
    return cf * form_factor * wetted_area_ratio

def reynolds_number(mach, altitude=0, characteristic_length=1):
    """
    Calculate Reynolds number based on Mach number
    """
    # Standard atmosphere properties at sea level
    rho_sl = 1.225  # kg/m³
    mu_sl = 1.789e-5  # kg/m·s
    a_sl = 340.3  # m/s
    
    # Altitude effects
    temperature_ratio = 1 - 0.0065 * altitude / 288.15
    rho = rho_sl * temperature_ratio**4.256
    mu = mu_sl * temperature_ratio**0.5
    
    velocity = mach * a_sl * np.sqrt(temperature_ratio)
    return (rho * velocity * characteristic_length) / mu

def takeoff_parameter(wing_loading, thrust_to_weight, cl_max):
    """
    Calculate takeoff parameter
    TOP = (W/S) / (T/W * CL_max)
    """
    return wing_loading / (thrust_to_weight * cl_max)

def landing_distance(wing_loading, cl_max, altitude=0):
    """
    Estimate landing distance (in meters)
    Simplified calculation assuming typical transport aircraft
    """
    rho = 1.225 * np.exp(-altitude/7400)
    approach_speed = np.sqrt(2 * wing_loading / (rho * cl_max))
    return 0.3 * approach_speed**2  # Simplified correlation

def turn_radius(velocity, bank_angle, load_factor=1):
    """Calculate turn radius"""
    g = 9.81  # m/s²
    return velocity**2 / (g * np.tan(np.radians(bank_angle)) * load_factor)

def specific_excess_power(thrust, drag, weight, velocity):
    """Calculate specific excess power"""
    return velocity * (thrust - drag) / weight

def range_estimate(lift_drag_ratio, specific_fuel_consumption, initial_weight, final_weight):
    """
    Estimate range using Breguet range equation
    Returns range in meters
    """
    g = 9.81  # m/s²
    velocity = 250  # m/s (assumed cruise velocity)
    return (velocity / (g * specific_fuel_consumption)) * lift_drag_ratio * np.log(initial_weight / final_weight)

def endurance_estimate(lift_drag_ratio, specific_fuel_consumption, initial_weight, final_weight):
    """
    Estimate endurance using Breguet endurance equation
    Returns endurance in seconds
    """
    return (1 / specific_fuel_consumption) * lift_drag_ratio * np.log(initial_weight / final_weight)

def tail_volume_coefficient(tail_area, tail_arm, wing_area, wing_mac):
    """Calculate tail volume coefficient"""
    return (tail_area * tail_arm) / (wing_area * wing_mac)

def static_margin(xcg, xac):
    """Calculate static margin"""
    return xac - xcg

# Test cases
def test_wing_loading():
    weight = 100000  # N
    wing_area = 100  # m²
    assert wing_loading(weight, wing_area) == 1000  # N/m²

def test_aspect_ratio():
    wingspan = 30  # m
    wing_area = 100  # m²
    assert aspect_ratio(wingspan, wing_area) == 9.0

def test_thrust_to_weight():
    thrust = 50000  # N
    weight = 100000  # N
    assert thrust_to_weight(thrust, weight) == 0.5

def test_empty_weight_fraction():
    mtow = 100000  # N
    empty_weight = 60000  # N
    assert empty_weight_fraction(mtow, empty_weight) == 0.6

def test_fuel_fraction():
    fuel_weight = 20000  # N
    mtow = 100000  # N
    assert fuel_fraction(fuel_weight, mtow) == 0.2

def test_wing_reynolds_number():
    velocity = 100  # m/s
    chord = 2  # m
    re = wing_reynolds_number(velocity, chord)
    assert re == pytest.approx(1.37e7, rel=0.01)

def test_oswald_efficiency():
    ar = 8
    sweep = 25  # degrees
    e = oswald_efficiency(ar, sweep)
    assert 0 < e < 1  # Efficiency should be between 0 and 1

def test_induced_drag():
    cl = 0.5
    ar = 8
    e = 0.8
    cdi = induced_drag_coefficient(cl, ar, e)
    assert cdi == pytest.approx(0.0124, rel=0.01)

def test_wetted_area_ratio():
    length = 30  # m
    diameter = 3  # m
    wing_area = 100  # m²
    ratio = wetted_area_ratio(length, diameter, wing_area)
    assert 4 < ratio < 8  # Typical transport aircraft range

def test_zero_lift_drag():
    mach = 0.8
    wetted_ratio = 6
    cd0 = zero_lift_drag_coefficient(mach, wetted_ratio)
    assert 0.015 < cd0 < 0.025  # Typical range for transport aircraft

def test_takeoff_parameter():
    ws = 5000  # N/m²
    tw = 0.3
    cl_max = 2.0
    top = takeoff_parameter(ws, tw, cl_max)
    assert 8000 < top < 9000  # Typical range for transport aircraft

def test_landing_distance():
    ws = 3000  # N/m²
    cl_max = 2.5
    distance = landing_distance(ws, cl_max)
    assert 800 < distance < 1200  # Typical range in meters

def test_turn_radius():
    velocity = 100  # m/s
    bank_angle = 30  # degrees
    radius = turn_radius(velocity, bank_angle)
    assert radius == pytest.approx(3094.7, rel=0.01)

def test_specific_excess_power():
    thrust = 100000  # N
    drag = 80000  # N
    weight = 200000  # N
    velocity = 250  # m/s
    sep = specific_excess_power(thrust, drag, weight, velocity)
    assert sep == pytest.approx(25, rel=0.01)

def test_range_estimate():
    ld_ratio = 15
    sfc = 2e-5  # 1/s
    initial_weight = 100000  # N
    final_weight = 80000  # N
    range_m = range_estimate(ld_ratio, sfc, initial_weight, final_weight)
    assert 3000000 < range_m < 4000000  # Typical medium-range aircraft

def test_endurance_estimate():
    ld_ratio = 15
    sfc = 2e-5  # 1/s
    initial_weight = 100000  # N
    final_weight = 80000  # N
    endurance = endurance_estimate(ld_ratio, sfc, initial_weight, final_weight)
    assert 15000 < endurance < 16000  # Typical endurance in seconds

def test_tail_volume_coefficient():
    tail_area = 20  # m²
    tail_arm = 15  # m
    wing_area = 100  # m²
    wing_mac = 3  # m
    vt = tail_volume_coefficient(tail_area, tail_arm, wing_area, wing_mac)
    assert 0.8 < vt < 1.2  # Typical range for horizontal tail

def test_static_margin():
    xcg = 0.25  # fraction of MAC
    xac = 0.30  # fraction of MAC
    sm = static_margin(xcg, xac)
    assert sm == pytest.approx(0.05)  # Typical transport aircraft static margin

def test_output_results():
    """Run all tests and output results to a .vsp3.txt file."""
    # Create tests directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)
    
    # Prepare the output file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/Sparrow3-Rev1.vsp3.txt"
    
    # Run the tests and capture results
    test_results = []
    test_functions = [
        test_wing_loading,
        test_aspect_ratio,
        test_thrust_to_weight,
        test_empty_weight_fraction,
        test_fuel_fraction,
        test_wing_reynolds_number,
        test_oswald_efficiency,
        test_induced_drag,
        test_wetted_area_ratio,
        test_zero_lift_drag,
        test_takeoff_parameter,
        test_landing_distance,
        test_turn_radius,
        test_specific_excess_power,
        test_range_estimate,
        test_endurance_estimate,
        test_tail_volume_coefficient,
        test_static_margin
    ]
    
    # Run each test and capture results
    with open(output_file, 'w') as f:
        f.write("Aircraft Sizing Test Results\n")
        f.write("==========================\n\n")
        f.write(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for test_func in test_functions:
            try:
                test_func()
                result = "PASS"
            except Exception as e:
                result = f"FAIL: {str(e)}"
            
            f.write(f"{test_func.__name__}: {result}\n")

if __name__ == "__main__":
    # Run the tests
    test_output_results()


