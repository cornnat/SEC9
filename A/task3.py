###########################################################
# Task 3
###################################################################

import numpy as np
import matplotlib.pyplot as plt

# Define the noisy phi^4 Hamiltonian
def H(theta):
    return theta**4 - 8*theta**2 - 2*np.cos(4*np.pi*theta)

# Gradient of H(theta)
def grad_H(theta):
    return 4*theta**3 - 16*theta + 8*np.pi*np.sin(4*np.pi*theta)



#########################################################
# PART A: Gradient Descent
#########################################################


def gradient_descent(theta0, alpha=0.01, tol=1e-5, max_iter=500):
    theta = theta0
    history = [theta]
    for _ in range(max_iter):
        theta_new = theta - alpha * grad_H(theta)
        if abs(theta_new - theta) < tol:
            break
        theta = theta_new
        history.append(theta)
    return np.array(history)

# Run gradient descent for different initial values
theta_inits = [-1, 0.5, 3]
paths = [gradient_descent(theta0) for theta0 in theta_inits]


#########################################################
# PART B: Metropolis-Hastings Algorithm
#########################################################


def metropolis_hastings(theta0, beta=1.0, sigma=0.5, steps=10000):
    theta = theta0
    history = [theta]
    for _ in range(steps):
        theta_star = theta + np.random.normal(0, sigma)
        dH = H(theta_star) - H(theta)
        if np.exp(-beta * dH) > np.random.rand():
            theta = theta_star
        history.append(theta)
    return np.array(history)

# Run Metropolis-Hastings for different initial values
mh_paths = [metropolis_hastings(theta0) for theta0 in theta_inits]



#########################################################
# PART C: Simulated Annealing
#########################################################

def simulated_annealing(theta0, beta_init=0.1, delta_beta=0.01, sigma=0.5, steps=10000):
    theta = theta0
    beta = beta_init
    history = [theta]
    for _ in range(steps):
        theta_star = theta + np.random.normal(0, sigma)
        dH = H(theta_star) - H(theta)
        if np.exp(-beta * dH) > np.random.rand():
            theta = theta_star
        history.append(theta)
        beta += delta_beta  # Cooling schedule
    return np.array(history)

# Run Simulated Annealing
sa_paths = [simulated_annealing(theta0) for theta0 in theta_inits]

# Plot results
plt.figure(figsize=(10, 6))
theta_vals = np.linspace(-3, 3, 500)
plt.plot(theta_vals, H(theta_vals), label='H(θ)')

colors = ['r', 'g', 'b']
for path, color, label in zip(paths, colors, theta_inits):
    plt.scatter(path, H(path), color=color, label=f'Grad Descent from {label}', s=5)

for path, color, label in zip(mh_paths, colors, theta_inits):
    plt.scatter(path, H(path), color=color, label=f'Metropolis-Hastings from {label}', s=5, alpha=0.5)

for path, color, label in zip(sa_paths, colors, theta_inits):
    plt.scatter(path, H(path), color=color, label=f'Simulated Annealing from {label}', s=5, alpha=0.5, marker='x')

plt.legend()
plt.xlabel('θ')
plt.ylabel('H(θ)')
plt.title('Optimization Methods Comparison')
plt.savefig('plots/3C_optimization_annealing.png', bbox_inches='tight')
plt.show()
