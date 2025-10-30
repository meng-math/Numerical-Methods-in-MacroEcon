'''VFI with log spaced Discretization to solve the Ramsey Model '''

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
grid = np.exp(np.linspace(np.log(k_min), np.log(k_max), n))


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

    for i, k in enumerate(grid):

        best = -np.inf

        for j, kp in enumerate(grid):

            c = k ** alpha - kp
            val = utility(c) + beta * V[j]
            

            if val > best:
                best = val
                best_idx = kp
                

        V_new[i] = best
        g_num[i] = best_idx

    error = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

end_time = time.time()

print(f"VFI iteration time: {end_time - start_time:.4f} seconds")

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



