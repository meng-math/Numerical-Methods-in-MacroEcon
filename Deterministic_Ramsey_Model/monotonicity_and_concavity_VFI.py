'''VFI with Full Discretization to solve the Ramsey Model with Monotonicity and Concavity Properties Exploited'''

import numpy as np


# === parameters setting ===

beta = 0.99
alpha = 0.3
tol = 1e-9
n = 500


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
    j_start = 0

    for i, k in enumerate(grid):

        best = -np.inf

        for j in range(j_start, n):

            c = k ** alpha - grid[j]
            val = utility(c) + beta * V[j]
            

            if val > best:
                best = val
                best_idx = grid[j]
                j_start = j  # exploit monotonicity

            else:

                break # value decreases beyond this point due to concavity

        V_new[i] = best
        g_num[i] = best_idx

    error = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

end_time = time.time()

print(f"VFI converged in {end_time - start_time:.4f} seconds.")



# === analytical value function ===


B = alpha / (1 - alpha * beta)

A = ( np.log(1 - alpha * beta) + beta * B * np.log(alpha * beta) ) / (1 - beta)



V_sol = A + B * np.log(grid)
g_ana = a * (grid ** alpha)

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



