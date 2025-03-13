########################################
# Task 1
#######################################

import numpy as np
from scipy.integrate import quad

# Constants
k = 1.38064852e-23  # Boltzmann constant in J/K
h = 6.62607015e-34  # Planck's constant in J·s
pi = np.pi
c = 3e8  # Speed of light in m/s
hbar = h / (2 * pi)  # Reduced Planck's constant
prefactor = (k**4 * c**2 * hbar) / (34 * pi**2)  # Prefactor for the expression

####################################################################
# Part A
#######################################################################

# Change of variable: x -> z, where x = z / (1 - z)
def integrand(z):
    x = z / (1 - z)  # Reverse the substitution
    return (x**3) / (np.exp(x) - 1)

# Perform the integration from z=0 to z=1 (the finite range after the substitution)
def evaluate_integral():
    result, error = quad(integrand, 0, 1)
    return result

####################################################################
# Part B
#######################################################################

def calculate_stefan_boltzmann_constant():
    integral_value = evaluate_integral()
    W = prefactor * integral_value  # The total rate of energy radiation
    return W

####################################################################
# Part C
#######################################################################

def evaluate_infinite_integral():
    def integrand_infinite(x):
        return (x**3) / (np.exp(x) - 1)
    
    # Integration from 0 to infinity
    result, error = quad(integrand_infinite, 0, np.inf)
    return result

# Evaluate the integral and calculate the Stefan-Boltzmann constant
stefan_boltzmann_constant = calculate_stefan_boltzmann_constant()
infinite_integral_value = evaluate_infinite_integral()

# Prepare the results
true_value = 5.670367e-8  # True value of Stefan-Boltzmann constant in W/m^2K^4
difference = abs(stefan_boltzmann_constant - true_value)

# Write the results to a file
with open("task1_results.txt", "w") as file:
    file.write(f"Value of the Stefan-Boltzmann constant from the integral: {stefan_boltzmann_constant:.6e} W/m^2K^4\n")
    file.write(f"Result of the integral from 0 to infinity using scipy's 'quad': {infinite_integral_value:.6e}\n")
    file.write(f"Known Stefan-Boltzmann constant: {true_value:.6e} W/m^2K^4\n")
    file.write(f"Difference between calculated and known value: {difference:.6e}\n")

# Output to inform the user that the results were saved
print("Results have been written to 'task1_results.txt'.")
