import tauchen_rouwenhorst as tr
import Ramsey_VFI_Howard_Improvement as rvfi
#import Ramsey_model_Endoeneous_Grid_Method as regm
import numpy as np
import matplotlib.pyplot as plt

# Parameters

rho = 0.979
mu = 0
sigma_es = 0.0072
N = 7

BETA = 0.984
ALPHA = 0.323
DELTA = 0.025

# Discretize continuous AR(1) process using Rouwenhorst's method

x_grid_rouwenhorst, P_rouwenhorst = tr.rouwenhorst(rho, mu, sigma_es, N)
print("Rouwenhorst State Grid for aggregate productivity:", np.exp(x_grid_rouwenhorst))
print("Rouwenhorst Transition Matrix:\n", P_rouwenhorst)

# Solve the deterministic Ramsey model via VFI with Howard Improvement
theta_values = np.exp(x_grid_rouwenhorst)
all_g_num = []
for theta in theta_values:
    g_num = rvfi.solve_ramsey_vfi_howard(beta = BETA,
                                         alpha = ALPHA,
                                         delta = DELTA,
                                         theta = theta)
    all_g_num.append(g_num)

# visualize the policy functions for different productivity states
plt.figure(figsize=(10, 6))
k_ss = ((ALPHA * theta_values[0]) / (1.0 / BETA - 1.0 + DELTA)) ** (1.0 / (1.0 - ALPHA))
k_min = 0.2 * k_ss
k_max = 3.0 * k_ss
grid = np.linspace(k_min, k_max, 2000)

for i, theta in enumerate(theta_values):
    plt.plot(grid, all_g_num[i], label=f'Theta={theta:.4f}')
    plt.title('Optimal Policy Functions k\'(k) for Different Productivity States')
    plt.xlabel('Capital k')
    plt.ylabel('Optimal Next Period Capital k\'')
    plt.legend()
    plt.grid()
plt.show()

#------
theta_value = np.exp((0.0072 ** 2) / ((1 - rho ** 2) * 2) ) # mean of log-normal distribution
k_ss = ((ALPHA * theta_value) / (1.0 / BETA - 1.0 + DELTA)) ** (1.0 / (1.0 - ALPHA))
print(k_ss)











