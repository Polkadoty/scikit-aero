# Scikit-aero

![Logo](docs/source/_static/logo.PNG)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/pypi/l/scikit-aero.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/scikit-aero/badge/?version=latest)](https://scikit-aero.readthedocs.io/)
[![Coverage](https://codecov.io/gh/AeroPython/scikit-aero/branch/master/graph/badge.svg)](https://codecov.io/gh/AeroPython/scikit-aero)

A Python package for aeronautical engineering calculations with support for parametric aircraft design.  Scikit-aero provides engineers and researchers with tools for atmospheric modeling, gas dynamics analysis, and preliminary aircraft design following industry-standard methodologies.

## Features

- Pythonic interface with modern type hints
- SI units support through Pint
- Full NumPy array support
- Standard atmosphere properties up to 86 kilometers
- Gas dynamics calculations
- Aircraft parametric design equations (Raymer)
- Orbital Mechanics equations (Curtis)
- Fully tested and documented

## Installation

```bash
# Requires Python 3.8 or later
pip install scikit-aero
```
## Quick Start

```python
    import skaero.atmosphere as atm
    import skaero.gasdynamics as gd

    # Get atmospheric properties at 10km

    # Define the altitude in meters
    h = 10000  # Altitude at 10 km
    # Retrieve the atmospheric properties at the specified altitude
    atmosphere = atm.coesa.table(h)
    # Print the temperature at the given altitude in Kelvin
    print(f"Temperature at {h}m: {atmosphere.temperature:.2f} K")

    # Calculate isentropic flow properties
    flow = gd.isentropic.IsentropicFlow(gamma=1.4)
    mach = 2.0
    print(f"Pressure ratio at M={mach}: {flow.p_p0(mach):.3f}")
    # Parametric design example
    from skaero.design import raymer
    wing = raymer.Wing(aspect_ratio=8.0, sweep=25.0)
    print(f"Wing weight fraction: {wing.weight_fraction():.3f}")
```


## Documentation

### Scikit-Aero documentation

Full documentation is available at [scikit-aero.readthedocs.io](https://scikit-aero.readthedocs.io/).

The documentation includes:
- API Reference
- Theory Guide
- Examples & Tutorials
- Contributing Guidelines

### Orbital Mechanics and Aircraft Design Documentation

- Added equations from Curtis' textbook and demonstrate practical implementation of spacecraft attitude dynamics and control within the scikit-aero framework.

- The Raymer equations module in scikit-aero implements the parametric aircraft design methodology developed by Daniel P. Raymer in his seminal work "Aircraft Design: A Conceptual Approach". These equations enable preliminary sizing and weight estimation during the conceptual design phase of aircraft development.

- The module provides a systematic approach to aircraft design by implementing statistical regression equations derived from historical aircraft data. These equations relate key aircraft parameters like wing loading, aspect ratio, and sweep angle to predict important design characteristics such as component weights, drag coefficients, and overall performance metrics.

### Curtis Equations Example:

```python
    # Import necessary modules
    from skaero.orbital import kepler
    import numpy as np
    
    # Define orbital parameters
    a = 7000  # semi-major axis in km
    e = 0.01  # eccentricity
    M = np.radians(30)  # mean anomaly in radians
    
    # Solve Kepler's Equation to find the eccentric anomaly
    E = kepler.solve_kepler(e, M)
    
    # Calculate true anomaly from eccentric anomaly
    true_anomaly = kepler.true_anomaly(e, E)
    
    print(f"Eccentric Anomaly: {np.degrees(E):.2f} degrees")
    print(f"True Anomaly: {np.degrees(true_anomaly):.2f} degrees")
```

### Attitude Control Example: 
This code implements a quaternion-based attitude control simulation for spacecraft detumbling. It uses a PD (Proportional-Derivative) controller to stabilize a rotating spacecraft to a desired orientation.

```python 
    import numpy as np
    import matplotlib.pyplot as plt

    # Constants
    A = 1000; B = 500; C = 800; J = [A, B, C]; J_m = np.diag(J)

    # scikit-aero functions
    def E_matrix(q):
        """
        Construct the E matrix for quaternion operations.
    
        Parameters:
        q (array-like): Quaternion represented as a 4-element array [q1, q2, q3, qs],
                        where q1, q2, q3 are the vector components and qs is the scalar component.
    
        Returns:
        numpy.ndarray: A 4x3 matrix used in quaternion calculations.
        """
        qs = q[3]
        qv = q[:3]
        E = np.zeros((4, 3))
        E[:3, :] = qs * np.eye(3) + skew(qv)
        E[3:, :] = -qv
        return E
    def skew(v):
        return np.array([[0, -v[2], v[1]], 
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    def EulerPD(w0, J, N):
        A, B, C = J
        dw0_x = (B - C) / A * w0[1] * w0[2] + N[0] / A
        dw0_y = (C - A) / B * w0[0] * w0[2] + N[1] / B
        dw0_z = (A - B) / C * w0[0] * w0[1] + N[2] / C
        return np.array([dw0_x, dw0_y, dw0_z])
    def quatMultiply(q1, q2):
        q1s, q1v = q1[3], q1[:3]
        q2s, q2v = q2[3], q2[:3]
        q3s = q1s * q2s - np.dot(q2v, q1v)
        q3v = q1s * q2v + q2s * q1v - np.cross(q2v, q1v)
        return np.concatenate([q3v, [q3s]])
      
    # Initial conditions
    omega_0 = np.array([0.2, -0.1, 0.15])
    q = np.array([0.5, -0.5, 0.5, 0.5])
    q_r = np.array([0, 0, 0, 1])
    # Control parameters
    Kp = 4; Kd = 10
    # Initialize variables
    w = omega_0.copy()
    t = 0
    dt = 0.1  # Reduced time step for better stability
    t_end = 1000  # Reduced simulation time
    time = [t]
    q1, q2, q3, q4, qnorm = [q[0]], [q[1]], [q[2]], [q[3]], [np.linalg.norm(q)]
    T = [0.5 * w @ J_m @ w]
    normN = []

    # Simulation loop
    while t < t_end:
    # Current quaternion error
    Beta = quatMultiply(q, q_r); Beta_v = Beta[:3]; Beta_s = Beta[3]
    # Control torque
    w_e = w; N = -Kp * Beta_v - Kd * w_e; normN.append(np.linalg.norm(N))
    # Derivatives
    w_dot = EulerPD(w, J, N); q_dot = 0.5 * E_matrix(q) @ w
    # Update states using Euler integration
    q_new = q + q_dot * dt
    q = q_new / np.linalg.norm(q_new)  # Normalize quaternion
    w = w + w_dot * dt
      
    # Store results
    time.append(t + dt)
    q1.append(q[0]); q2.append(q[1]); q3.append(q[2]); q4.append(q[3])
    qnorm.append(np.linalg.norm(q)); T.append(0.5 * w @ J_m @ w)
    t += dt
    # Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(121); plt.plot(time, T)
    plt.title('Kinetic Energy'); plt.xlabel('Time (s)')
    plt.ylabel('Kinetic Energy'); plt.grid(True)
    
    plt.subplot(122) 
    plt.plot(time, q1, label='q₁')
    plt.plot(time, q2, label='q₂')
    plt.plot(time, q3, label='q₃')
    plt.plot(time, q4, label='q₄')
    plt.plot(time, qnorm, '--', label='|q|')
    plt.legend()
    plt.title('Quaternion Components')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

```

## Development

To set up a development environment:
```bash
    git clone https://github.com/AeroPython/scikit-aero.git
    cd scikit-aero
    pip install -e ".[dev]"
```

To run tests:
```bash
    pytest
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas we're looking to improve:

- Additional parametric design equations from Raymer
- More aircraft design examples
- Performance improvements
- Documentation enhancements

## License

scikit-aero is released under a BSD license. See the [LICENSE](LICENSE) file for details.

## Citation

If you use scikit-aero in your research, please cite:

```bibtex
@software{scikit_aero,
author = {Cano, Juan Luis and Doty, Andrew James and Hom, Dennis},
title = {scikit-aero: Aeronautical engineering calculations in Python},
year = {2024},
publisher = {GitHub},
url = {https://github.com/AeroPython/scikit-aero}
}
```
