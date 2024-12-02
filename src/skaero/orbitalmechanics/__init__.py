import numpy as np


"""
Orbital Mechanics Calculations

The following notation is used throughout the different modules:

* x = sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune, pluto
    * mu_x = standard gravitational parameter for different planets
    * omega_x = 2*pi / rotation time of x
    * n_earth = angular velocity of earth around the sun
    * radius_x = Radius of x
    * G = gravitational constant (6.67*10^-20) km^3/(sec^2*kg)
    * sidereal_sec = Sidereal day in seconds
    * solarday_sec = Solar day in seconds
    * J2 = Earth perturbation value
* Conversions
    * AU to km
    * TU to sec
* ORBITS
    * a = periapsis
    * b = apoapsis
    * c = semi-major axis
    * d = semi-minor axis
    * e = eccentricity
    * i = inclination
    * omega = argument of periapsis
    * Omega = longitude of the ascending node
    * theta = true anomaly
    * T = orbital period
    * M = mean anomaly
    * E = eccentric anomaly
    * r = distance from the center of the ellipse to the satellite
    * v = velocity of the satellite
    * h = specific angular momentum
    * p = semi-latus rectum
    * n = mean motion
"""

# Gravitational constant
G = 6.67e-20  # km^3/(sec^2*kg)

# Sidereal and Solar day in seconds
sidereal_sec = 86164.1
solarday_sec = 86400

# Earth perturbation value
J2 = 1.08263e-3

# Standard gravitational parameters (mu) in km^3/sec^2
mu_sun = 1.32712440018e11
mu_mercury = 2.2032e4
mu_venus = 3.24859e5
mu_earth = 3.986004418e5
mu_moon = 4.9048695e3
mu_mars = 4.282837e4
mu_jupiter = 1.26686534e8
mu_saturn = 3.7931187e7
mu_uranus = 5.793939e6
mu_neptune = 6.836529e6
mu_pluto = 8.71e2

# Angular velocities (omega) in rad/sec
omega_sun = 2 * np.pi / (25.38 * 24 * 3600)
omega_mercury = 2 * np.pi / (58.646 * 24 * 3600)
omega_venus = 2 * np.pi / (243.025 * 24 * 3600)
omega_earth = 2 * np.pi / sidereal_sec
omega_moon = 2 * np.pi / (27.321661 * 24 * 3600)
omega_mars = 2 * np.pi / (1.025957 * 24 * 3600)
omega_jupiter = 2 * np.pi / (0.41354 * 24 * 3600)
omega_saturn = 2 * np.pi / (0.44401 * 24 * 3600)
omega_uranus = 2 * np.pi / (0.71833 * 24 * 3600)
omega_neptune = 2 * np.pi / (0.67125 * 24 * 3600)
omega_pluto = 2 * np.pi / (6.387230 * 24 * 3600)

# Radius in km
radius_sun = 696340
radius_mercury = 2439.7
radius_venus = 6051.8
radius_earth = 6371.0
radius_moon = 1737.1
radius_mars = 3389.5
radius_jupiter = 69911
radius_saturn = 58232
radius_uranus = 25362
radius_neptune = 24622
radius_pluto = 1188.3

# Earth's angular velocity around the sun in rad/sec
n_earth = 2 * np.pi / (365.256363004 * 24 * 3600)

################ CONVERSIONS ################

# Astronomical Unit (AU) in km
AU = 1.496e8

# Time Unit (TU) in seconds
TU = 58.1324409 * 24 * 3600

# Function to convert km to AU
def km_to_au(km):
    return km / AU

# Function to convert AU to km
def au_to_km(au):
    return au * AU

# Function to convert seconds to TU
def sec_to_tu(seconds):
    return seconds / TU

# Function to convert TU to seconds
def tu_to_sec(tu):
    return tu * TU