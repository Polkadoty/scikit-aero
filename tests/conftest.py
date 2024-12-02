import pytest
import numpy as np
from skaero.gasdynamics import isentropic, shocks

@pytest.fixture
def isentropic_flow():
    """Standard isentropic flow with gamma=1.4"""
    return isentropic.IsentropicFlow(gamma=1.4)

@pytest.fixture
def mach_test_cases():
    """Common test cases for Mach number calculations"""
    return {
        'subsonic': [0.0, 0.27, 0.89, 1.0],
        'supersonic': [1.3, 2.05, 3.0, np.inf],
        'ratios': {
            'pressure': [1.0, 0.9506, 0.5977, 0.5283],
            'density': [1.0, 0.96446008, 0.69236464, 0.63393815],
            'temperature': [1.0, 0.98571429, 0.86363636, 0.83333333]
        }
    }

@pytest.fixture
def shock_test_cases():
    """Common test cases for shock calculations"""
    return {
        'mach_1': [1.5, 1.8, 2.1, 3.0],
        'mach_2': [0.7011, 0.6165, 0.5613, 0.4752],
        'ratios': {
            'pressure': [2.4583, 3.6133, 4.9783, 10.3333],
            'density': [1.8621, 2.3592, 2.8119, 3.8571],
            'temperature': [1.3202, 1.5316, 1.7705, 2.6790]
        }
    } 