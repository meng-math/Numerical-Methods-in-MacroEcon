'''Solve numerically the capital policy function of Ramsey model using Euler equation method plus time iteration.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


# === parameters setting ===

beta = 0.99
alpha = 0.3
tol = 1e-9
n = 2000

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

def euler_residual(k, kp, g_prev):
    """F(k,k') = 1/(f(k)-k') - beta*f'(k')/(f(k')-g_prev(k'))"""
    '''k: current capital
       kp: next period capital
       g_prev: previous policy function g(k)'''
    c = f(k) - kp
    if c <= 0:      
        return np.inf
    gp = np.interp(kp, grid, g_prev)      # g_prev(k')
    cp = f(kp) - gp
    if cp <= 0:
        return -np.inf
    return uprime(c) - beta * uprime(cp) * fprime(kp)


# === analytical solution ===
def k_policy_analytical(k):
    return a * (k ** alpha)


# === numerical solution ===


# --- Time Iteration state ---

k_new = 0.3 * f(grid)
k_old = np.zeros_like(k_new)
err = np.max(np.abs(k_new - k_old)) / np.max(np.abs(k_old))


flag = bool(err > tol)

import time

start_time = time.time()


while flag:
    
    k_old = k_new.copy()
    k_old_interp = lambda x: np.interp(x, grid, k_old)

    for i, k_val in enumerate(grid):
        F = lambda x: euler_residual(k_val, x, k_old)   # use F consistently

        eps = 1e-12
        left, right = k_min, min(k_max, f(k_val) - eps)
        sol = root_scalar(F, bracket=[left, right], method='bisect')
        k_new[i] = sol.root

    err = np.max(np.abs(k_new - k_old)) / np.max(np.abs(k_old))
    print(f'Current error: {err:.2e}')
    flag = bool(err > tol)


end_time = time.time()
print(f"Converged in {end_time - start_time:.2f} seconds.")
print(f"The final error is {err:.2e}.")

mean_error = np.mean(np.abs(k_new - k_policy_analytical(grid)) / np.abs(k_policy_analytical(grid)))
max_error = np.max(np.abs(k_new - k_policy_analytical(grid)) / np.abs(k_policy_analytical(grid)))

print(f"maximum absolute error in policy function: {max_error:.5e}")
print(f"mean error in policy function: {mean_error:.5e}")

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
plt.title("Policy function relative error (VFI(TI) vs analytical)")
plt.grid(True)
plt.show()
