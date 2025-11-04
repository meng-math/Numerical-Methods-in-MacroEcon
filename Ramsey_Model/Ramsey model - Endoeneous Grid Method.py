'''Ramsey model - Endogenous Grid Method (cash-at-hand) with Euler Equation'''

import numpy as np
import matplotlib.pyplot as plt
import time

# === parameters setting ===
beta = 0.99
alpha = 0.3
tol = 1e-9
n = 200
grid_style = 'linear'   # 'linear' or 'log'

# === steady state and grids ===
a = alpha * beta
k_ss = (a)**(1.0/(1.0 - alpha))
k_min = 0.2 * k_ss
k_max = 3.0 * k_ss

if grid_style == 'linear':
    grid_k = np.linspace(k_min, k_max, n)            # grid_k is exogeneous grid
elif grid_style == 'log':
    grid_k = np.exp(np.linspace(np.log(k_min), np.log(k_max), n))



# ---- primitives: δ=1, z=1, log utility ----
def f(k):      return k**alpha
def fprime(k): return alpha * k**(alpha - 1.0)
def uprime(c): return 1.0 / c
def uprime_inv(mu): return 1.0 / mu  # inverse of u'(c) = 1/c

# === analytical solution (for δ=1, z=1, log u): k' = αβ * k^α ===
def k_policy_analytical(k):
    return a * (k ** alpha)

# === EGM (cash-at-hand) ===
# We represent the policy function as g(x), where x = f(k), then interpolate it to x=f(k_grid)
# Initial guess: use analytical solution or a moderate linear savings rate
g_old_on_k = 0.3 * f(grid_k)                 # your original initialization


# To construct g(x), we need a k' grid; here use the same range and style as k
if grid_style == 'linear':
    grid_kp = np.linspace(k_min, k_max, n)   # control grid for k'
else:
    grid_kp = np.exp(np.linspace(np.log(k_min), np.log(k_max), n))

# Convergence criterion: compare g_new(k) with g_old(k) sup-norm on the same k set
def sup_norm_rel(new, old):
    denom = np.max(np.abs(old))
    return np.max(np.abs(new - old)) / denom

err = 1.0
it = 0
start_time = time.time()

while err > tol:
    it += 1

    # Step 1: use old policy g_old(k) to construct g_old(k') for evaluation on k' grid
    # Note: here g_old_on_k is defined on k-grid, we need its values on k' grid: g_old(k') by interpolation
    g_old_on_kp = np.interp(grid_kp, grid_k, g_old_on_k)

    # Step 2: for each k' compute c_next, c_today, x_today = k' + c_today
    xs = np.empty_like(grid_kp)   # x_today
    kps = np.empty_like(grid_kp)  # k'
    grid_endog = np.empty_like(grid_kp)  # placeholder for endogenous grid

    for i, kp in enumerate(grid_kp):

        gp = g_old_on_kp[i]                 # k'' = g_old(k')
        c_next = f(kp) - gp                 # c_{t+1} = f(k') - g(k')
        rhs = beta * uprime(c_next) * fprime(kp)   # β * u'(c_{t+1}) * f'(k')
        c_today = uprime_inv(rhs)                  # c_t = (u')^{-1}(rhs)
        xs[i] = kp + c_today                       # x_today = k' + c_t
        kps[i] = kp
    # Now we have pairs of (x_today, k') = (xs, kps)
        grid_endog[i] = xs[i] ** (1.0 / alpha)   # corresponding k for x_today


    # # To evaluate g(k) on the k grid, we first take x = f(k_grid) as the argument
    # x_on_k = f(grid_k)

    g_new_on_k = np.interp( grid_k, grid_endog, kps)

    # # Interpolation: g_new_on_k = g(x_on_k)
    # # Note boundary extrapolation strategy: clamp to boundary to avoid NaN
    # g_new_on_k = np.interp(x_on_k, xs, kps,
    #                        left=kps[0], right=kps[-1])

    err = sup_norm_rel(g_new_on_k, g_old_on_k)
    g_old_on_k = g_new_on_k.copy()
    print(f"[EGM] iter={it}  sup-norm rel err = {err:.2e}")

end_time = time.time()
print(f"Converged in {it} iterations, {end_time - start_time:.4f} seconds.")
print(f"Final sup-norm rel err = {err:.2e}")

# === error assessment (comparison with analytical policy; analytical solution exists only in this special case) ===
rel_err = np.abs(g_old_on_k - k_policy_analytical(grid_k)) / np.maximum(1e-12, np.abs(k_policy_analytical(grid_k)))
mean_error = np.mean(rel_err)
max_error = np.max(rel_err)
print(f"maximum relative error in policy function: {max_error:.5e}")
print(f"mean relative error in policy function: {mean_error:.5e}")

# === Optional: Euler-equation error (static version, one period) ===
def euler_error_curve(g_on_k, grid_k):
    # EE(k) = | (c - c_imp) / c_imp |
    ee = np.zeros_like(grid_k)
    for i, k in enumerate(grid_k):
        kp = np.interp(k, grid_k, g_on_k)          # k' = g(k)
        c  = f(k) - kp
        kp2 = np.interp(kp, grid_k, g_on_k)        # k'' = g(k')
        c_next = f(kp) - kp2
        rhs = beta * uprime(c_next) * fprime(kp)
        c_imp = uprime_inv(rhs)
        ee[i] = np.abs((c - c_imp) / max(c_imp, 1e-14))
    return ee

ee_curve = euler_error_curve(g_old_on_k, grid_k)
print(f"EE mean = {ee_curve.mean():.5e}, EE max = {ee_curve.max():.5e}")

# === plot results: policy ===
plt.figure(figsize=(8, 5))
plt.plot(grid_k, g_old_on_k, label="Numerical g(k) via EGM", lw=2)
plt.plot(grid_k, k_policy_analytical(grid_k), '--', label="Analytical g(k)", lw=2)
plt.xlabel("Capital k")
plt.ylabel("Next period k'")
plt.title("Policy function comparison (EGM)")
plt.grid(True)
plt.legend()
plt.show()

# === plot error distribution in policy ===
plt.figure(figsize=(8, 5))
plt.plot(grid_k, rel_err, lw=2)
plt.xlabel("Capital k")
plt.ylabel("relative error")
plt.title("Policy function relative error (EGM vs analytical)")
plt.grid(True)
plt.show()

# === plot Euler-equation error across capital ===
plt.figure(figsize=(8, 5))
plt.plot(grid_k, ee_curve, lw=2)
plt.xlabel("Capital k")
plt.ylabel("Euler equation error")
plt.title("EE error across capital (EGM)")
plt.grid(True)
plt.show()