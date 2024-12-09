import numpy as np
''' This module provides functions for various attitude dynamics Functions and Simulation 
Functions
----------
- General Equations
    - skew
    - DCM (all 3 axes)
    - Euler313 and 321 Sequences
- Euler Method
    - Rotation Matrix
- DCM Method
    -
    -
- Quaterion Method
    - E_matrix
    - invQuat
    - quatMultiply
    - quat2dcm and dcm2quat
- Torque-Free Motion
    - TF_EOM
    - 
    - 
-
-
-
'''
###################
###   GENERAL   ###
###################  

def skew(v):
    """
    Create a skew-symmetric matrix from a 3D vector.

    A skew-symmetric (or antisymmetric) matrix is a square matrix A such that A = -A^T.
    This function is particularly useful in attitude dynamics and rotational mechanics
    to represent cross products as matrix multiplications.

    Parameters:
    v (array-like): A 3D vector [v1, v2, v3].

    Returns:
    numpy.ndarray: A 3x3 skew-symmetric matrix of the form:
        [[ 0,  -v3,  v2],
         [ v3,   0, -v1],
         [-v2,  v1,   0]]

    Edge Cases:
    - If input vector is not 3D, function will raise an error
    - Zero vector input will return a 3x3 zero matrix

    The skew-symmetric matrix S(v) of vector v satisfies the property:
    S(v)x = v × x (cross product) for any vector x
    """
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
def DCM_1(theta):
    """
    Create a Direction Cosine Matrix for rotation about the 1-axis (x-axis).
    
    The DCM represents a right-handed rotation through angle theta about the 1-axis.
    
    Parameters:
    theta (float): Rotation angle in radians
    
    Returns:
    numpy.ndarray: A 3x3 rotation matrix of the form:
        [[1,    0,         0     ],
         [0, cos(theta), -sin(theta)],
         [0, sin(theta),  cos(theta)]]
         
    Properties:
    - Orthogonal matrix (R * R^T = I)
    - Determinant = 1
    - Inverse = Transpose
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])
def DCM_2(theta):
    """
    Create a Direction Cosine Matrix for rotation about the 2-axis (y-axis).
    
    The DCM represents a right-handed rotation through angle theta about the 2-axis.
    
    Parameters:
    theta (float): Rotation angle in radians
    
    Returns:
    numpy.ndarray: A 3x3 rotation matrix of the form:
        [[ cos(theta), 0, sin(theta)],
         [     0,      1,     0     ],
         [-sin(theta), 0, cos(theta)]]
         
    Properties:
    - Orthogonal matrix (R * R^T = I)
    - Determinant = 1
    - Inverse = Transpose
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])
def DCM_3(theta):
    """
    Create a Direction Cosine Matrix for rotation about the 3-axis (z-axis).
    
    The DCM represents a right-handed rotation through angle theta about the 3-axis.
    
    Parameters:
    theta (float): Rotation angle in radians
    
    Returns:
    numpy.ndarray: A 3x3 rotation matrix of the form:
        [[cos(theta), -sin(theta), 0],
         [sin(theta),  cos(theta), 0],
         [    0,           0,      1]]
         
    Properties:
    - Orthogonal matrix (R * R^T = I)
    - Determinant = 1
    - Inverse = Transpose
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])
def Euler313(phi, theta, psi):
    """
    Create a Direction Cosine Matrix for 3-1-3 Euler angle sequence.
    
    Represents successive rotations:
    1. Rotation about 3-axis (z) by angle phi
    2. Rotation about 1-axis (x) by angle theta 
    3. Rotation about 3-axis (z) by angle psi
    
    Parameters:
    phi (float): First rotation angle about 3-axis in radians
    theta (float): Second rotation angle about 1-axis in radians
    psi (float): Third rotation angle about 3-axis in radians
    
    Returns:
    numpy.ndarray: The combined 3x3 rotation matrix R = R3(psi)R1(theta)R3(phi)
    
    Properties:
    - Orthogonal matrix (R * R^T = I)
    - Determinant = 1
    - Inverse = Transpose
    """
    return DCM_3(psi) @ DCM_1(theta) @ DCM_3(phi)
def Euler321(phi, theta, psi):
    """
    Create a Direction Cosine Matrix for 3-2-1 Euler angle sequence.
    
    Represents successive rotations:
    1. Rotation about 3-axis (z) by angle phi
    2. Rotation about 2-axis (y) by angle theta
    3. Rotation about 1-axis (x) by angle psi
    
    Parameters:
    phi (float): First rotation angle about 3-axis in radians
    theta (float): Second rotation angle about 2-axis in radians 
    psi (float): Third rotation angle about 1-axis in radians
    
    Returns:
    numpy.ndarray: The combined 3x3 rotation matrix R = R1(psi)R2(theta)R3(phi)
    
    Properties:
    - Orthogonal matrix (R * R^T = I) 
    - Determinant = 1
    - Inverse = Transpose
    """
    return DCM_1(psi) @ DCM_2(theta) @ DCM_3(phi)

#################
###   EULER   ###
#################

def rotation_matrix(e, theta):
    """
    Create a rotation matrix for rotation about an arbitrary Euler axis.
    
    Rotates a vector through angle theta about the Euler axis e using Euler's rotation formula:
    R = cos(θ)I + (1-cos(θ))ee^T + sin(θ)e×
    
    Parameters:
    e (array-like): Unit vector representing the Euler axis of rotation [e1, e2, e3]
    theta (float): Rotation angle in radians
    
    Returns:
    numpy.ndarray: A 3x3 rotation matrix
    
    Properties:
    - Orthogonal matrix (R * R^T = I)
    - Determinant = 1
    - Inverse = Transpose
    - For unit vector e: R*e = e (eigenvalue = 1)
    
    Edge Cases:
    - If e is not a unit vector, function will still work but rotation will be scaled
    - Zero rotation angle returns identity matrix
    """
    # Ensure e is a numpy array
    e = np.array(e)
    
    # Calculate rotation matrix components
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Outer product ee^T
    ee_t = np.outer(e, e)
    
    # Skew symmetric matrix of e
    e_skew = skew(e)
    
    # Construct rotation matrix using Euler's rotation formula
    R = c * np.eye(3) + (1 - c) * ee_t + s * e_skew
    
    return R

#######################
###   QUATERNIONS   ###
#######################
def quat2dcm(q):
    """
    Convert a quaternion to a Direction Cosine Matrix (DCM).
    
    Parameters:
    q (array-like): A quaternion in the form [qv1, qv2, qv3, qs] where qv is 
                    the vector part and qs is the scalar part.
    
    Returns:
    numpy.ndarray: The equivalent 3x3 DCM
    
    Properties:
    - Output is orthogonal (R * R^T = I)
    - Determinant = 1
    - More computationally efficient than Euler angles
    """
    q0, q1, q2, q3 = q[3], q[0], q[1], q[2]
    dcm = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return dcm
def dcm2quat(dcm):
    """
    Convert a Direction Cosine Matrix (DCM) to a quaternion.
    
    Uses Shepperd's method which is numerically stable.
    
    Parameters:
    dcm (array-like): A 3x3 rotation matrix
    
    Returns:
    numpy.ndarray: The equivalent quaternion [qv1, qv2, qv3, qs]
    
    Properties:
    - Handles numerical precision issues
    - Returns the quaternion with positive scalar part
    """
    tr = np.trace(dcm)
    if tr > 0:
        S = 2 * np.sqrt(tr + 1)
        qs = 0.25 * S
        qx = (dcm[2,1] - dcm[1,2]) / S
        qy = (dcm[0,2] - dcm[2,0]) / S
        qz = (dcm[1,0] - dcm[0,1]) / S
    else:
        if dcm[0,0] > dcm[1,1] and dcm[0,0] > dcm[2,2]:
            S = 2 * np.sqrt(1 + dcm[0,0] - dcm[1,1] - dcm[2,2])
            qs = (dcm[2,1] - dcm[1,2]) / S
            qx = 0.25 * S
            qy = (dcm[0,1] + dcm[1,0]) / S
            qz = (dcm[0,2] + dcm[2,0]) / S
        elif dcm[1,1] > dcm[2,2]:
            S = 2 * np.sqrt(1 + dcm[1,1] - dcm[0,0] - dcm[2,2])
            qs = (dcm[0,2] - dcm[2,0]) / S
            qx = (dcm[0,1] + dcm[1,0]) / S
            qy = 0.25 * S
            qz = (dcm[1,2] + dcm[2,1]) / S
        else:
            S = 2 * np.sqrt(1 + dcm[2,2] - dcm[0,0] - dcm[1,1])
            qs = (dcm[1,0] - dcm[0,1]) / S
            qx = (dcm[0,2] + dcm[2,0]) / S
            qy = (dcm[1,2] + dcm[2,1]) / S
            qz = 0.25 * S
    return np.array([qx, qy, qz, qs])
def invQuat(q):
    """
    Calculate the inverse of a quaternion.

    The inverse quaternion represents the opposite rotation of the input quaternion.
    For unit quaternions, the inverse is equal to the conjugate.

    Parameters:
    q (array-like): A quaternion in the form [qv1, qv2, qv3, qs] where qv is the vector part
                    and qs is the scalar part.

    Returns:
    numpy.ndarray: The inverse quaternion [-qv1, -qv2, -qv3, qs].

    Edge Cases:
    - If input quaternion is not normalized, function will still return conjugate
    - Zero quaternion input will return zero quaternion

    Properties:
    - For a unit quaternion q, q * q^(-1) = [0, 0, 0, 1]
    - The inverse preserves the magnitude of the quaternion
    """
    qv = q[:3]
    qs = q[3]
    return np.concatenate([-qv, [qs]])
def E_matrix(q):
    """
    Calculate the E matrix used in quaternion kinematics equations.

    The E matrix relates the angular velocity vector to the quaternion derivative:
    q_dot = 1/2 * E(q) * omega

    Parameters:
    q (array-like): A quaternion in the form [qv1, qv2, qv3, qs] where qv is the vector part
                    and qs is the scalar part.

    Returns:
    numpy.ndarray: A 4x3 matrix E(q) defined as:
        [[qs*I + S(qv)],
         [   -qv^T    ]]
        where I is the 3x3 identity matrix and S(qv) is the skew-symmetric matrix of qv.

    Edge Cases:
    - If input quaternion is not 4D, function will raise an error
    - Zero quaternion input will return a 4x3 zero matrix

    Properties:
    - E(q) is used to propagate quaternion attitude kinematics
    - For a unit quaternion q, E(q)^T * E(q) = I_3x3
    """
    qs = q[3]
    qv = q[:3]
    E = np.zeros((4, 3))
    E[:3, :] = qs * np.eye(3) + skew(qv)
    E[3:, :] = -qv
    return E
def quatMultiply(q1, q2):
    """
    Multiply two quaternions using quaternion multiplication rules.

    The quaternion multiplication is non-commutative and follows the Hamilton product rule:
    q1 * q2 = [q1s*q2s - q1v·q2v, q1s*q2v + q2s*q1v + q1v×q2v]

    Parameters:
    q1 (array-like): First quaternion in the form [qv1, qv2, qv3, qs]
    q2 (array-like): Second quaternion in the form [qv1, qv2, qv3, qs]

    Returns:
    numpy.ndarray: The quaternion product q1*q2 in the form [qv1, qv2, qv3, qs]

    Edge Cases:
    - If input quaternions are not normalized, output will not be normalized
    - Zero quaternion inputs will return zero quaternion

    Properties:
    - Non-commutative: q1*q2 ≠ q2*q1 in general
    - Associative: (q1*q2)*q3 = q1*(q2*q3)
    - Distributive over addition
    """
    q1s, q1v = q1[3], q1[:3]
    q2s, q2v = q2[3], q2[:3]
    q3s = q1s * q2s - np.dot(q2v, q1v)
    q3v = q1s * q2v + q2s * q1v - np.cross(q2v, q1v)
    return np.concatenate([q3v, [q3s]])

##############################
###   TORQUE FREE MOTION   ###
##############################

def TF_EOM(w0, J, N):
    """
    Calculate the angular acceleration of a rigid body under torque-free motion using Euler's equations.

    This function implements Euler's equations of motion for a rigid body, which describe how the angular 
    velocity changes over time due to both the body's inertial properties and applied external torques.

    Parameters:
    w0 (array-like): Current angular velocity vector [wx, wy, wz]
    J (array-like): Principal moments of inertia [A, B, C]
    N (array-like): External torques applied to the body [Nx, Ny, Nz]

    Returns:
    numpy.ndarray: Angular acceleration vector [dwx/dt, dwy/dt, dwz/dt]

    Edge Cases:
    - Zero moments of inertia will cause division by zero
    - If input vectors are not 3D, function will raise an error

    Properties:
    - For torque-free motion (N = [0,0,0]), total angular momentum is conserved
    - For axisymmetric bodies (A = B), motion simplifies to regular precession
    """
    A, B, C = J
    dw0_x = (B - C) / A * w0[1] * w0[2] + N[0] / A
    dw0_y = (C - A) / B * w0[0] * w0[2] + N[1] / B
    dw0_z = (A - B) / C * w0[0] * w0[1] + N[2] / C
    return np.array([dw0_x, dw0_y, dw0_z])

#############################
