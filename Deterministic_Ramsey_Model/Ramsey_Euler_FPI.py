'''Solve numerically the capital policy function of Ramsey model using Euler equation method plus fixed point iteration.'''

import numpy as np
import matplotlib.pyplot as plt


# === parameters setting ===

beta = 0.99
alpha = 0.3
tol = 1e-9
n = 200
damp_factor = 0.2   # for conservative updating

# === full discretiazation ===
a = alpha * beta
k_ss = (a)**(1.0/(1.0 - alpha))
k_min = 0.2 * k_ss
k_max = 3.0 * k_ss
grid = np.linspace(k_min, k_max, n)


# --- primitives: δ=1, z=1, log utility ---
def f(k):      return k**alpha
def fprime(k): return alpha * k**(alpha - 1.0)
def uprime(c): return 1.0 / c


# === analytical solution ===
def k_policy_analytical(k):
    return a * (k ** alpha)


# === numerical solution ===


# --- Fixed Point Iteration state ---

k_new = 0.3 * f(grid)
k_old = np.zeros_like(k_new)
err = err = np.max(np.abs(k_new - k_old)) / np.max(np.abs(k_old))


flag = bool(err > tol)

import time

start_time = time.time()
iter_times = 0

while flag:
    
    k_old = k_new.copy()
    iter_times += 1

    for i, k_val in enumerate(grid):
        k_new[i] = f(k_val) - ( f(k_old[i]) - np.interp(k_old[i], grid, k_old)  ) / (beta * fprime(k_old[i]))  # initial guess

    k_new = damp_factor * k_new + (1 - damp_factor) * k_old  # conservative dampening
    err = np.max(np.abs(k_new - k_old)) / np.max(np.abs(k_old))
    print(f'Current error: {err:.2e}')
    flag = bool(err > tol)


end_time = time.time()

mean_error = np.mean(np.abs(k_new - k_policy_analytical(grid)) / np.abs(k_policy_analytical(grid)))
max_error = np.max(np.abs(k_new - k_policy_analytical(grid)) / np.abs(k_policy_analytical(grid)))

print(f"maximum absolute error in policy function: {max_error:.5e}")
print(f"mean error in policy function: {mean_error:.5e}")

print(f"Converged in {end_time - start_time:.2f} seconds.")
print(f"The final error is {err:.2e}.")
print(f"Number of iterations: {iter_times}")

# === plot results ===



plt.figure(figsize=(8, 5))
plt.plot(grid, k_new, label="Numerical g(k)", lw=2)
plt.plot(grid, k_policy_analytical(grid), '--', label="Analytical g(k)", lw=2)
plt.xlabel("Capital k")
plt.ylabel("Next period k")
plt.title("Policy function comparison")
plt.grid(True)
plt.legend()  
plt.show()


# === plot error distribution in policy ===
rel_err = np.abs(k_new - k_policy_analytical(grid)) / np.abs(k_policy_analytical(grid))
plt.figure(figsize=(8, 5))
plt.plot(grid, rel_err, lw=2)
plt.xlabel("Capital k")
plt.ylabel("relative error")
plt.title("Policy function relative error (Euler(FPI) vs analytical)")
plt.grid(True)
plt.show()