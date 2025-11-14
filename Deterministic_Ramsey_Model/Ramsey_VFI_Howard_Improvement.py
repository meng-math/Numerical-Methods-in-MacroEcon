# ------------------------------------------------------------------------------
# Howard Policy Improvement in Value Function Iteration (VFI)
# ------------------------------------------------------------------------------

import numpy as np
import time
import matplotlib.pyplot as plt


# === parameters setting ===

beta = 0.99
alpha = 0.3
tol = 1e-9
n = 2000 # number of grid points, or choose n = 500 for problem 1


# === full discretiazation ===
a = alpha * beta
k_ss = (a)**(1.0/(1.0 - alpha))
k_min = 0.2 * k_ss
k_max = 3.0 * k_ss
grid = np.linspace(k_min, k_max, n)


V = np.zeros(n)
g_num = np.zeros(n)
err_g = 1.0
err_V = 1.0


BIG_NEG = -1e16

# pre-compute payoff matrix


def payoff(k_col, kp_row):
    ''' 
    payoff function 
    input: k (current capital) with shape (n, 1), kp (next period capital) with shape (1, n)
    output: payoff matrix with shape (n, n)

    '''
    c = k_col**alpha - kp_row

    return np.where(c > 0, np.log(c), BIG_NEG)

# k as column, kp as row -> shape (n, n)
k_col = grid[:, None]                # shape (n,1)
kp_row = grid[None, :]               # shape (1,n)
util_mat = payoff(k_col, kp_row)
val_mat = util_mat + beta * V[None, :]  # value matrix: u(c) + beta * V(kp) broadcast V over rows
best_idx = np.argmax(val_mat, axis=1)          # shape (n,)


import time

start_time = time.time()

while err_g > tol:

    V_new = val_mat[np.arange(n), best_idx]        # max values, shape (n,)
    g_num = grid[best_idx]                         # chosen kp values
    err_V = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

    # Howard Policy Improvement step using the current policy function (best_idx)
    while err_V > tol:

        V = V_new.copy()

        V_new = util_mat[np.arange(n), best_idx] + beta * V[best_idx]  # update V_new using current policy

        err_V = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

    # after Howard step, update policy function
    val_mat = util_mat + beta * V_new[None, :]  # broadcast V_new over rows
    best_idx = np.argmax(val_mat, axis=1)          # shape (n,)
    err_g = np.max(np.abs(grid[best_idx] - g_num)) / (1.0 + np.max(np.abs(g_num)))
    g_num = grid[best_idx]                         # chosen kp values


end_time = time.time()


print(f"VFI with Howard Improvement converged in {end_time - start_time:.4f} seconds")



# === analytical value function ===


B = alpha / (1 - alpha * beta)

A = ( np.log(1 - alpha * beta) + beta * B * np.log(alpha * beta) ) / (1 - beta)



V_sol = A + B * np.log(grid)
g_ana = a * (grid ** alpha)

print("maximum absolute error in policy function:", np.max(np.abs(g_num - g_ana) / np.abs(g_ana)))
print("mean error in policy function:", np.mean(np.abs(g_num - g_ana) / np.abs(g_ana)))

# === visualization of numerical value function and analytical value function

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(grid, V_new, label="Numerical V(k)", lw=2)
plt.plot(grid, V_sol, '--', label="Analytical V(k)", lw=2)
plt.xlabel("Capital")
plt.ylabel("Value Function")
plt.title("Value Function: Numerical vs Analytical")
plt.grid(True)
plt.legend()  
plt.show()



plt.figure(figsize=(8, 5))
plt.plot(grid, g_num, label="Numerical g(k)", lw=2)
plt.plot(grid, g_ana, '--', label="Analytical g(k)", lw=2)
plt.xlabel("Capital k")
plt.ylabel("Next period k")
plt.title("Policy function comparison")
plt.grid(True)
plt.legend()  
plt.show()



# === plot error distribution in policy ===
rel_err = np.abs(g_num - g_ana) / np.abs(g_ana)
plt.figure(figsize=(8, 5))
plt.plot(grid, rel_err, lw=2)
plt.xlabel("Capital k")
plt.ylabel("relative error")
plt.title("Policy function relative error (VFI(FD) vs analytical)")
plt.grid(True)
plt.show()

