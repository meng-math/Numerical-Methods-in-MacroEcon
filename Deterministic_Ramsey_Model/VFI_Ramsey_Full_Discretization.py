'''VFI with Full Discretization to solve the Ramsey Model '''

import numpy as np


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
V_new = np.zeros(n)
g_num = np.zeros(n)
error = 1.0


BIG_NEG = -1e16

# === VFI iteration ===


def payoff(k_col, kp_row):
    ''' 
    payoff function 
    input: k (current capital) with shape (n, 1), kp (next period capital) with shape (1, n)
    output: payoff matrix with shape (n, n)

    '''
    c = k_col**alpha - kp_row

    return np.where(c > 0, np.log(c), BIG_NEG)

# pre-compute payoff matrix

# k as column, kp as row -> shape (n, n)
k_col = grid[:, None]                # shape (n,1)
kp_row = grid[None, :]               # shape (1,n)
util_mat = payoff(k_col, kp_row)

import time

start_time = time.time()

while error > tol:
    
    V = V_new.copy()

    # value matrix: u(c) + beta * V(kp)
    val_mat = util_mat + beta * V[None, :]  # broadcast V over rows

    # for each k (each row) pick best kp (max over columns)
    best_idx = np.argmax(val_mat, axis=1)          # shape (n,)
    V_new = val_mat[np.arange(n), best_idx]        # max values, shape (n,)
    g_num = grid[best_idx]                         # chosen kp valuesi

    error = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))
    print(f"VFI FD iteration error: {error:.10f}")


end_time = time.time()


print(f"VFI with Full Discretization converged in {end_time - start_time:.4f} seconds")



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

