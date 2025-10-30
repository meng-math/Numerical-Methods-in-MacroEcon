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

def utility(c):
    return np.log(c) if c > 0 else BIG_NEG

import time

start_time = time.time()

while error > tol:
    
    V = V_new.copy()

    # k as column, kp as row -> shape (n, n)
    k_col = grid[:, None]                # shape (n,1)
    kp_row = grid[None, :]               # shape (1,n)

    # consumption matrix: c[i,j] = k_i^alpha - kp_j
    cons = k_col**alpha - kp_row        # shape (n,n)

    # utility matrix; infeasible consumption -> BIG_NEG
    util_mat = np.where(cons > 0, np.log(cons), BIG_NEG)  # shape (n,n)

    # value matrix: u(c) + beta * V(kp)
    val_mat = util_mat + beta * V[None, :]  # broadcast V over rows

    # for each k (each row) pick best kp (max over columns)
    best_idx = np.argmax(val_mat, axis=1)          # shape (n,)
    V_new = val_mat[np.arange(n), best_idx]        # max values, shape (n,)
    g_num = grid[best_idx]                         # chosen kp values

    error = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

# while error > tol:
    
#     V = V_new.copy()

#     for i, k in enumerate(grid):

#         best = -np.inf

#         for j, kp in enumerate(grid):

#             c = k ** alpha - kp
#             val = utility(c) + beta * V[j]
            

#             if val > best:
#                 best = val
#                 best_idx = kp
                

#         V_new[i] = best
#         g_num[i] = best_idx

#     error = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

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

