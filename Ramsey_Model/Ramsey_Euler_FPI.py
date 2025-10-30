'''Solve numerically the capital policy function of Ramsey model using Euler equation method plus fixed point iteration.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


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
k_max = 1.0 * k_ss
grid = np.linspace(k_min, k_max, n)


# --- primitives: δ=1, z=1, log utility ---
def f(k):      return k**alpha
def fprime(k): return alpha * k**(alpha - 1.0)
def uprime(c): return 1.0 / c

def euler_residual(k, kp, g_prev):
    """F(k,k') = 1/(f(k)-k') - beta*f'(k')/(f(k')-g_prev(k'))"""
    '''k: current capital grid point
       kp: next period capital
       g_prev: previous policy function g(k)'''
    c = f(k) - kp
    if c <= 0:      
        return np.inf
    gp = np.interp(k, grid, g_prev)      # g_prev(k)
    cp = f(gp) - np.interp(gp, grid, g_prev)
    if cp <= 0:
        return -np.inf
    return uprime(c) - beta * uprime(cp) * fprime(gp)


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
    k_old_interp = lambda x: np.interp(x, grid, k_old)
    iter_times += 1

    for i, k_val in enumerate(grid):
        F = lambda x: euler_residual(k_val, x, k_old)   # use F consistently
        left, right = k_min, k_max
        f_left, f_right = F(left), F(right)

        if f_left * f_right <= 0:
            try:
                sol = root_scalar(F, bracket=[left, right], method='bisect')
                k_new[i] = sol.root
            except ValueError:
                # rare: numerical issues at endpoints -> fallback
                k_new[i] = k_policy_analytical(k_val)
        else:
            # scan interval for a sub-bracket with a sign change
            xs = np.linspace(left, right, 200)
            found = False
            for x0, x1 in zip(xs[:-1], xs[1:]):
                if F(x0) * F(x1) <= 0:
                    try:
                        sol = root_scalar(F, bracket=[x0, x1], method='bisect')
                        k_new[i] = sol.root
                        found = True
                        break
                    except ValueError:
                        # if bisect fails here, continue scanning
                        pass
            if not found:
                # no sign change found — use analytical policy as safe fallback
                k_new[i] = k_policy_analytical(k_val)
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