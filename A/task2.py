######################################################
# Task 2
########################################################

########################################################
# Part A and B
########################################################

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
e = 0.6  # Eccentricity
Tf = 200  # Final time

# Initial conditions
q1_0 = 1 - e
q2_0 = 0
p1_0 = 0
p2_0 = np.sqrt((1 + e) / (1 - e))

# Function to compute accelerations (Hamiltonian derivatives)
def acceleration(q1, q2):
    r = (q1**2 + q2**2)**(3/2)
    return -q1 / r, -q2 / r

# Explicit Euler Method
N1 = 100000  # Number of time steps
Dt1 = Tf / N1  # Time step

q1_euler = np.zeros(N1)
q2_euler = np.zeros(N1)
p1_euler = np.zeros(N1)
p2_euler = np.zeros(N1)

q1_euler[0], q2_euler[0] = q1_0, q2_0
p1_euler[0], p2_euler[0] = p1_0, p2_0

for n in range(N1 - 1):
    q1_euler[n+1] = q1_euler[n] + Dt1 * p1_euler[n]
    q2_euler[n+1] = q2_euler[n] + Dt1 * p2_euler[n]
    a1, a2 = acceleration(q1_euler[n], q2_euler[n])
    p1_euler[n+1] = p1_euler[n] + Dt1 * a1
    p2_euler[n+1] = p2_euler[n] + Dt1 * a2

# Symplectic Euler Method
N2 = 400000  # Number of time steps
Dt2 = Tf / N2  # Time step

q1_symp = np.zeros(N2)
q2_symp = np.zeros(N2)
p1_symp = np.zeros(N2)
p2_symp = np.zeros(N2)

q1_symp[0], q2_symp[0] = q1_0, q2_0
p1_symp[0], p2_symp[0] = p1_0, p2_0

for n in range(N2 - 1):
    a1, a2 = acceleration(q1_symp[n], q2_symp[n])
    p1_symp[n+1] = p1_symp[n] + Dt2 * a1
    p2_symp[n+1] = p2_symp[n] + Dt2 * a2
    q1_symp[n+1] = q1_symp[n] + Dt2 * p1_symp[n+1]
    q2_symp[n+1] = q2_symp[n] + Dt2 * p2_symp[n+1]

# Plot results
plt.figure(figsize=(8, 8))
plt.plot(q1_euler, q2_euler, label='Explicit Euler', alpha=0.6)
plt.plot(q1_symp, q2_symp, label='Symplectic Euler', alpha=0.8)
plt.xlabel("q1")
plt.ylabel("q2")
plt.title("Orbit of the Planet (Explicit vs Symplectic Euler)")
plt.legend()
plt.grid()
plt.savefig('plots/2AB_explicit_symplectic_euler.png', bbox_inches='tight')
plt.show()
