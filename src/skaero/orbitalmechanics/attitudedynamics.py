import numpy as np
from scipy.integrate import ode

# Constants
A = 1000
B = 500 
C = 800
J = [A, B, C]
J_m = np.diag(J)

# Initial conditions
omega_0 = np.array([0.2, -0.1, 0.15])
q = np.array([0.5, -0.5, 0.5, 0.5])  # Current quaternion
q_r = np.array([0, 0, 0, 1])  # Desired quaternion

# Control parameters
Kp = 4
Kd = 10
k_max = 1 / np.linalg.norm(omega_0)
kp = k_max - 0.05
kd = 1 / 5

# Initialize variables
w = omega_0
t = 0
dt = 1
t_end = 1000
time = [t]
q1, q2, q3, q4, qnorm = [q[0]], [q[1]], [q[2]], [q[3]], [np.linalg.norm(q)]
T = [0.5 * w @ J_m @ w]
normN = []

# Simulation loop
while t < t_end:
    # Current quaternion error
    Beta = quatMultiply(q, q_r)
    Beta_v = Beta[:3]
    Beta_s = Beta[3]

    # Quaternion kinematics
    w_e = w
    
    # Control torque
    N = -Kp * Beta_v - Kd * w_e
    normN.append(np.linalg.norm(N))

    # Derivatives
    w_dot = EulerPD(w, J, N)
    q_dot = 0.5 * E_matrix(q) @ w
    Beta_dot = 0.5 * E_matrix(Beta) @ w_e

    # Error measures
    v1 = 0.5 * w_e @ J_m @ w_e
    v1_dot = w_e @ J_m @ w_dot
    v2 = 2 * Kp * (1 - Beta_s)
    v2_dot = 2 * Kp * (-Beta_dot[3])
    v_dot = w_e @ (N + Kp * Beta_v)

    # Update angular velocity and position
    q = (q_dot * dt + q) / np.linalg.norm(q_dot * dt + q)
    w = w_dot * dt + w

    # Append to lists
    time.append(t + dt)
    q1.append(q[0])
    q2.append(q[1])
    q3.append(q[2])
    q4.append(q[3])
    qnorm.append(np.linalg.norm(q))
    T.append(0.5 * w @ J_m @ w)

    # Iterate
    t += dt

# Plot results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(time, T)
plt.title('Kinetic Energy')
plt.xlabel('Time (s)')
plt.ylabel('Kinetic Energy')

plt.figure()
plt.plot(time, q1, time, q2, time, q3, time, q4, time, qnorm)
plt.legend(['q_1', 'q_2', 'q_3', 'q_4', 'Norm'])
plt.title('Quaternion Position')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion')

plt.show()

def E_matrix(q):
    qs = q[3]
    qv = q[:3]
    return np.block([[qs * np.eye(3) + skew(qv), -qv.T], [qv, qs]])

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
    return np.hstack([q3v, q3s])
