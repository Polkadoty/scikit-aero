
import numpy as np
"""
This module provides functions for various orbital mechanics calculations.
Functions
---------
- periapsis(a, e):
- apoapsis(a, e):
- semi_major_axis(a, b):
- semi_minor_axis(a, e):
- eccentricity(a, b):
- distance_from_center(a, e, theta):
- velocity(mu, a, r):
- specific_angular_momentum(mu, a, e):
- semi_latus_rectum(a, e):
- mean_motion(mu, a):
- position_vector(a, e, theta):
- velocity_vector(mu, a, e, theta):
- orbital_energy(mu, a):
- orbital_period_from_semi_major_axis(a, mu):
- true_anomaly_from_mean_anomaly(M, e, tol=1e-6):
- mean_anomaly_to_eccentric_anomaly(M, e, tol=1e-6):
"""
def periapsis(a, e):
    """
    Calculate the periapsis distance of an orbit.

    The periapsis is the point in the orbit of a celestial body where it is closest to the body it is orbiting.

    Parameters:
    a (float): Semi-major axis of the orbit (in meters).
    e (float): Eccentricity of the orbit (dimensionless).

    Returns:
    float: The periapsis distance (in meters).

    Equation:
    periapsis = a * (1 - e)

    This equation is useful in orbital mechanics to determine the closest approach of an orbiting body to the primary body it orbits.

    Edge Cases:
    - If the eccentricity (e) is 0, the orbit is circular, and the periapsis distance is equal to the semi-major axis (a).
    - If the eccentricity (e) is 1, the orbit is parabolic, and the periapsis distance is 0.
    - The function does not handle eccentricities greater than or equal to 1, which would result in non-closed orbits (parabolic or hyperbolic trajectories).
    """
    return a * (1 - e)

def apoapsis(a, e):
    """
    Calculate the apoapsis of an orbit.

    The apoapsis is the point in the orbit of an object where it is farthest from the body it is orbiting.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.

    Returns
    -------
    float
        The apoapsis distance.
    """
    return a * (1 + e)

def semi_major_axis(a, b):
    """
    Calculate the semi-major axis of an ellipse.

    The semi-major axis is the average of the maximum and minimum distances
    from the center of the ellipse to its perimeter.

    Parameters
    ----------
    a : float
        The maximum distance from the center of the ellipse to its perimeter (apoapsis).
    b : float
        The minimum distance from the center of the ellipse to its perimeter (periapsis).

    Returns
    -------
    float
        The semi-major axis of the ellipse.
    """
    return (a + b) / 2

def semi_minor_axis(a, e):
    """
    Calculate the semi-minor axis of an ellipse.

    Parameters
    ----------
    a : float
        Semi-major axis of the ellipse.
    e : float
        Eccentricity of the ellipse.

    Returns
    -------
    float
        Semi-minor axis of the ellipse.
    """
    return a * np.sqrt(1 - e**2)

def eccentricity(a, b):
    """
    Calculate the eccentricity of an ellipse.

    Parameters
    ----------
    a : float
        Semi-major axis of the ellipse.
    b : float
        Semi-minor axis of the ellipse.

    Returns
    -------
    float
        Eccentricity of the ellipse.

    Notes
    -----
    The eccentricity is a measure of how much the ellipse deviates from being circular. It ranges from 0 (a circle) to 1 (a parabola).
    """
    return np.sqrt(1 - (b**2 / a**2))

def distance_from_center(a, e, theta): 
    """
    Calculate the distance from the center of the orbit to a point on the orbit.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.
    theta : float
        True anomaly in degrees.

    Returns
    -------
    float
        Distance from the center of the orbit to the point specified by the true anomaly.
    """
    return a * (1 - e**2) / (1 + e * np.cos(np.radians(theta)))

def velocity(mu, a, r):
    """
    Calculate the orbital velocity at a given distance from the central body.

    Parameters
    ----------
    mu : float
        Standard gravitational parameter of the central body (GM), in units of km^3/s^2.
    a : float
        Semi-major axis of the orbit, in units of km.
    r : float
        Distance from the central body to the point of interest, in units of km.

    Returns
    -------
    float
        Orbital velocity at the given distance, in units of km/s.
    """
    return np.sqrt(mu * (2/r - 1/a))

def specific_angular_momentum(mu, a, e): 
    """
    Calculate the specific angular momentum of an orbit.

    The specific angular momentum is a measure of the angular momentum per unit mass of an orbiting body.

    Parameters
    ----------
    mu : float
        Standard gravitational parameter (GM) of the central body.
    a : float
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.

    Returns
    -------
    float
        Specific angular momentum of the orbit.
    """
    return np.sqrt(mu * a * (1 - e**2))

def semi_latus_rectum(a, e): 
    """
    Calculate the semi-latus rectum of an orbit.

    The semi-latus rectum is a measure of the size of an orbit and is related to the semi-major axis and the eccentricity.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.

    Returns
    -------
    float
        The semi-latus rectum of the orbit.
    """
    return a * (1 - e**2)

def mean_motion(mu, a): 
    """
    Calculate the mean motion of an orbit.
    Parameters
    ----------
    mu : float
        Standard gravitational parameter of the central body (GM).
    a : float
        Semi-major axis of the orbit.
    Returns
    -------
    float
        Mean motion of the orbit.
    """
    return np.sqrt(mu / a**3)

def position_vector(a, e, theta): 
    """
    Calculate the position vector of an orbiting body in the orbital plane.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.
    theta : float
        True anomaly in degrees.

    Returns
    -------
    numpy.ndarray
        Position vector [x, y] in the orbital plane.

    Notes
    -----
    The position vector is calculated using the polar coordinates (r, theta) 
    where r is the distance from the center of the orbit to the orbiting body 
    and theta is the true anomaly.
    """
    r = distance_from_center(a, e, theta)
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    return np.array([x, y])

def velocity_vector(mu, a, e, theta): 
    """
    Calculate the velocity vector of an orbiting body at a given true anomaly.

    Parameters
    ----------
    mu : float
        Standard gravitational parameter of the primary body (e.g., Earth).
    a : float
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.
    theta : float
        True anomaly in degrees.

    Returns
    -------
    numpy.ndarray
        Velocity vector [vx, vy] in the orbital plane.

    Notes
    -----
    The velocity vector is calculated using the vis-viva equation and the specific angular momentum.
    """
    r = distance_from_center(a, e, theta)
    h = specific_angular_momentum(mu, a, e)
    vr = (mu / h) * e * np.sin(np.radians(theta))
    vtheta = (mu / h) * (1 + e * np.cos(np.radians(theta)))
    vx = vr * np.cos(np.radians(theta)) - vtheta * np.sin(np.radians(theta))
    vy = vr * np.sin(np.radians(theta)) + vtheta * np.cos(np.radians(theta))
    return np.array([vx, vy])

def orbital_energy(mu, a): 
    """
    Calculate the specific orbital energy of an orbit.

    Parameters
    ----------
    mu : float
        Standard gravitational parameter of the primary body (e.g., Earth).
    a : float
        Semi-major axis of the orbit.

    Returns
    -------
    float
        Specific orbital energy of the orbit.

    Notes
    -----
    The specific orbital energy is given by the formula:
        E = -mu / (2 * a)
    where E is the specific orbital energy, mu is the standard gravitational parameter, and a is the semi-major axis of the orbit.
    """
    return -mu / (2 * a)

def orbital_period_from_semi_major_axis(a, mu): 
    """
    Calculate the orbital period from the semi-major axis.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit (in meters).
    mu : float
        Standard gravitational parameter (in m^3/s^2).

    Returns
    -------
    float
        Orbital period (in seconds).

    Notes
    -----
    The orbital period is calculated using the formula:
    T = 2 * pi * sqrt(a^3 / mu)
    where T is the orbital period, a is the semi-major axis, and mu is the standard gravitational parameter.
    """
    return 2 * np.pi * np.sqrt(a**3 / mu)

def true_anomaly_from_mean_anomaly(M, e, tol=1e-6): 
    """
    Calculate the true anomaly from the mean anomaly for an orbit.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    e : float
        Eccentricity of the orbit.
    tol : float, optional
        Tolerance for the iterative solution of the eccentric anomaly, by default 1e-6.

    Returns
    -------
    float
        True anomaly in degrees.

    Notes
    -----
    The true anomaly is the angle between the direction of periapsis and the current position of the body on its orbit, 
    measured at the focus of the ellipse (the position of the central body).
    """
    E = mean_anomaly_to_eccentric_anomaly(M, e, tol)
    return np.degrees(2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)))

def mean_anomaly_to_eccentric_anomaly(M, e, tol=1e-6):
    """
    Converts mean anomaly to eccentric anomaly for an orbit.

    Parameters
    ----------
    M : float
        Mean anomaly in degrees.
    e : float
        Eccentricity of the orbit.
    tol : float, optional
        Tolerance for the iterative solution (default is 1e-6).

    Returns
    -------
    float
        Eccentric anomaly in radians.

    Notes
    -----
    This function uses an iterative method to solve Kepler's equation for the
    eccentric anomaly.
    """
    M = np.radians(M)
    E = M if e < 0.8 else np.pi
    F = E - e * np.sin(E) - M
    while abs(F) > tol:
        E = E - F / (1 - e * np.cos(E))
        F = E - e * np.sin(E) - M
    return E
def radius(a=None, e=None, theta=None, E=None, p=None):
    """
    Calculate the radius of an orbit.

    Parameters
    ----------
    a : float, optional
        Semi-major axis of the orbit.
    e : float
        Eccentricity of the orbit.
    theta : float, optional
        True anomaly in degrees.
    E : float, optional
        Eccentric anomaly in radians.
    p : float, optional
        Semi-latus rectum of the orbit.

    Returns
    -------
    float
        Radius of the orbit.

    Notes
    -----
    If `theta` is provided, the function uses the formula:
        r = p / (1 + e * np.cos(np.radians(theta)))
    If `E` is provided, the function uses the formula:
        r = a * (1 - e * np.cos(E))
    """
    if theta is not None and p is not None:
        return p / (1 + e * np.cos(np.radians(theta)))
    elif E is not None and a is not None:
        return a * (1 - e * np.cos(E))
    else:
        raise ValueError("Invalid arguments. Provide either (p, e, theta) or (a, e, E).")
def specific_orbital_energy(V=None, mu=None, r=None, a=None):
    """
    Calculate the specific orbital energy of an orbit.

    Parameters
    ----------
    V : float, optional
        Orbital velocity at a given distance from the central body.
    mu : float
        Standard gravitational parameter of the central body.
    r : float, optional
        Distance from the central body to the point of interest.
    a : float, optional
        Semi-major axis of the orbit.

    Returns
    -------
    float
        Specific orbital energy of the orbit.

    Notes
    -----
    If V and r are provided, the function uses the formula:
        epsilon = 0.5 * V**2 - mu / r
    If a is provided, the function uses the formula:
        epsilon = -mu / (2 * a)
    """
    if V is not None and r is not None:
        return 0.5 * V**2 - mu / r
    elif a is not None:
        return -mu / (2 * a)
    else:
        raise ValueError("Invalid arguments. Provide either (V, mu, r) or (mu, a).")
def mean_anomaly(E=None, e=None, n=None, t=None, tp=None):
    """
    Calculate the mean anomaly.

    Parameters
    ----------
    E : float, optional
        Eccentric anomaly in radians.
    e : float, optional
        Eccentricity of the orbit.
    n : float, optional
        Mean motion of the orbit.
    t : float, optional
        Time since periapsis passage.
    tp : float, optional
        Time of periapsis passage.

    Returns
    -------
    float
        Mean anomaly in radians.

    Notes
    -----
    If E and e are provided, the function uses the formula:
        M = E - e * np.sin(E)
    If n, t, and tp are provided, the function uses the formula:
        M = n * (t - tp)
    """
    if E is not None and e is not None:
        return E - e * np.sin(E)
    elif n is not None and t is not None and tp is not None:
        return n * (t - tp)
    else:
        raise ValueError("Invalid arguments. Provide either (E, e) or (n, t, tp).")
def Eccentric_anomaly(e,theta):
    """
    Calculate the eccentric anomaly from the true anomaly.

    Parameters
    ----------
    e : float
        Eccentricity of the orbit.
    theta : float
        True anomaly in degrees.

    Returns
    -------
    float
        Eccentric anomaly in radians.

    Notes
    -----
    The eccentric anomaly is the angle between the direction of periapsis and the position of the body on its orbit, 
    measured at the center of the ellipse.
    """
    theta = np.radians(theta)
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta / 2))
    return E
def flight_path_angle(r_dot,r,theta_dot,mu,p):
    """
    Calculate the flight path angle of an orbiting body.

    Parameters
    ----------
    r_dot : float
        Radial velocity of the orbiting body.
    r : float
        Distance from the center of the orbit to the orbiting body.
    theta_dot : float
        Tangential velocity of the orbiting body.
    mu : float
        Standard gravitational parameter of the central body.
    p : float
        Semi-latus rectum of the orbit.

    Returns
    -------
    float
        Flight path angle in radians.

    Notes
    -----
    The flight path angle is the angle between the velocity vector of the orbiting body and the local horizontal.
    """
    return np.arctan((r * theta_dot**2) / (mu) - 1)
def r_dot(mu,p,e,theta):
    """
    Calculate the radial velocity of an orbiting body.

    Parameters
    ----------
    mu : float
        Standard gravitational parameter of the central body.
    p : float
        Semi-latus rectum of the orbit.
    e : float
        Eccentricity of the orbit.
    theta : float
        True anomaly in degrees.

    Returns
    -------
    float
        Radial velocity of the orbiting body.

    Notes
    -----
    The radial velocity is the component of the velocity vector of the orbiting body that is directed towards or away from the central body.
    """
    return np.sqrt(mu / p) * e * np.sin(np.radians(theta))
def theta_dot(r,mu,p,e,theta):
    """
    Calculate the tangential velocity of an orbiting body.

    Parameters
    ----------
    r : float
        Distance from the center of the orbit to the orbiting body.
    mu : float
        Standard gravitational parameter of the central body.
    p : float
        Semi-latus rectum of the orbit.
    e : float
        Eccentricity of the orbit.
    theta : float
        True anomaly in degrees.

    Returns
    -------
    float
        Tangential velocity of the orbiting body.

    Notes
    -----
    The tangential velocity is the component of the velocity vector of the orbiting body that is perpendicular to the radial direction.
    """
    return 1/r * np.sqrt(mu / p) * (1 + e * np.cos(np.radians(theta)))
def eccentricity(ra,rp):
    """
    Calculate the eccentricity of an orbit from the apoapsis and periapsis distances.

    Parameters
    ----------
    ra : float
        Apoapsis distance.
    rp : float
        Periapsis distance.

    Returns
    -------
    float
        Eccentricity of the orbit.

    Notes
    -----
    The eccentricity is a measure of how much the orbit deviates from being circular. It ranges from 0 (a circle) to 1 (a parabola).
    """
    return (ra - rp) / (ra + rp)
def thetaGMST(GMST_0,omega_E,t,t0):
    """
    Calculate the Greenwich Mean Sidereal Time (GMST) at a given time.

    Parameters
    ----------
    GMST_0 : float
        Initial Greenwich Mean Sidereal Time (GMST) at a reference time.
    omega_E : float
        Angular velocity of the Earth's rotation.
    t : float
        Time elapsed since the reference time.
    t0 : float
        Reference time.

    Returns
    -------
    float
        Greenwich Mean Sidereal Time (GMST) at the given time.

    Notes
    -----
    The Greenwich Mean Sidereal Time (GMST) is the angle between the Greenwich meridian and the vernal equinox, measured in the plane of the celestial equator.
    """
    return GMST_0 + omega_E * (t - t0)
def rPQW(r,theta):
    """
    Convert the position vector from the inertial frame to the Perifocal frame.

    Parameters
    ----------
    r : numpy.ndarray
        Position vector in the inertial frame.
    theta : float
        True anomaly in degrees.

    Returns
    -------
    numpy.ndarray
        Position vector in the Perifocal frame.

    Notes
    -----
    The Perifocal frame is a coordinate system that is fixed to the orbiting body, with the x-axis pointing towards the periapsis, 
    the y-axis in the orbital plane, and the z-axis perpendicular to the orbital plane.
    """
    theta = np.radians(theta)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return np.dot(R, r)
def rECI(rPQW,omega_E,t):
    """
    Convert the position vector from the Perifocal frame to the Earth-Centered Inertial (ECI) frame.

    Parameters
    ----------
    rPQW : numpy.ndarray
        Position vector in the Perifocal frame.
    omega_E : float
        Angular velocity of the Earth's rotation.
    t : float
        Time elapsed since the reference time.

    Returns
    -------
    numpy.ndarray
        Position vector in the Earth-Centered Inertial (ECI) frame.

    Notes
    -----
    The Earth-Centered Inertial (ECI) frame is a coordinate system that is fixed to the Earth, with the x-axis pointing towards the vernal equinox, 
    the y-axis in the plane of the equator, and the z-axis perpendicular to the equator.
    """
    theta = omega_E * t
    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return np.dot(R, rPQW)
def omegadot_J2(J2,R_E,a):
    """
    Calculate the rate of change of the argument of perigee due to the J2 effect.

    Parameters
    ----------
    J2 : float
        J2 coefficient of the central body.
    R_E : float
        Equatorial radius of the central body.
    a : float
        Semi-major axis of the orbit.

    Returns
    -------
    float
        Rate of change of the argument of perigee due to the J2 effect.

    Notes
    -----
    The J2 effect is a perturbation in the orbit of a satellite caused by the oblateness of the central body.
    """
    return -1.5 * J2 * (R_E / a)**2
def omegadot_3body(mu1,mu2,a):
    """
    Calculate the rate of change of the argument of perigee due to the third-body effect.

    Parameters
    ----------
    mu1 : float
        Standard gravitational parameter of the primary body.
    mu2 : float
        Standard gravitational parameter of the third body.
    a : float
        Semi-major axis of the orbit.

    Returns
    -------
    float
        Rate of change of the argument of perigee due to the third-body effect.

    Notes
    -----
    The third-body effect is a perturbation in the orbit of a satellite caused by the gravitational influence of a third body.
    """
    return -1.5 * mu2 / (mu1 * a)**2
