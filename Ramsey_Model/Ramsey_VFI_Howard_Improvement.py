# ------------------------------------------------------------------------------
# Howard Policy Improvement (non-vectorized)
#   Outer loop: policy improvement (greedy update)
#   Inner loop: policy evaluation (Gauss-Seidel)
# ------------------------------------------------------------------------------

import numpy as np
import time
import matplotlib.pyplot as plt

# === Parameters ===
beta = 0.99
alpha = 0.3
tol = 1e-9        # tolerance for Bellman residual (outer loop)
eval_tol = 1e-12  # tolerance for policy evaluation (inner loop)
n = 500
BIG_NEG = -1e16

# === Capital grid ===
a = alpha * beta
k_ss = a ** (1.0 / (1.0 - alpha))         # steady state
k_min, k_max = 0.2 * k_ss, 3.0 * k_ss
grid = np.linspace(k_min, k_max, n)

def utility(c: float) -> float:
    """Log utility function with large negative penalty for infeasible consumption."""
    return np.log(c) if c > 0 else BIG_NEG

def howard_loops(grid, alpha, beta, tol, eval_tol,
                 max_outer=10_000, max_eval=1_000_000):
    n = len(grid)
    V = np.zeros(n)
    pol_idx = np.zeros(n, dtype=int)  # initial policy (will be overwritten)
    outer = 0

    t0 = time.perf_counter()
    while outer < max_outer:
        # 1) Policy improvement: find greedy policy w.r.t current V
        pol_new = np.zeros(n, dtype=int)
        for i in range(n):
            k = grid[i]
            k_alpha = k ** alpha
            best_val = -np.inf
            best_j = 0
            for j in range(n):
                kp = grid[j]
                c = k_alpha - kp
                val = utility(c) + beta * V[j]
                if val > best_val:
                    best_val = val
                    best_j = j
            pol_new[i] = best_j

        # If the policy hasn't changed (after first iteration), stop
        if outer > 0 and np.array_equal(pol_new, pol_idx):
            break
        pol_idx = pol_new

        # 2) Policy evaluation: solve V = u(i,pol[i]) + beta * V[pol[i]]
        #    using Gauss–Seidel iteration
        it_eval = 0
        while it_eval < max_eval:
            V_old = V.copy()
            for i in range(n):
                j = pol_idx[i]
                k = grid[i]
                kp = grid[j]
                c = k ** alpha - kp
                V[i] = utility(c) + beta * V[j]
            max_abs_V = max(abs(x) for x in V_old)
            err_eval = max(abs(V[i] - V_old[i]) for i in range(n)) / (1.0 + max_abs_V)
            it_eval += 1
            if err_eval < eval_tol:
                break

        outer += 1

        # 3) Check Bellman residual for convergence
        max_resid = 0.0
        for i in range(n):
            k = grid[i]
            k_alpha = k ** alpha
            best = -np.inf
            for j in range(n):
                kp = grid[j]
                c = k_alpha - kp
                val = utility(c) + beta * V[j]
                if val > best:
                    best = val
            resid = abs(best - V[i])
            if resid > max_resid:
                max_resid = resid
        if max_resid < tol:
            break

    t1 = time.perf_counter()
    return V.copy(), pol_idx.copy(), outer, (t1 - t0)

# === Run both versions for comparison ===
V_h, pol_h, it_outer, time_h = howard_loops(grid, alpha, beta, tol, eval_tol)
print(f"Howard improvement:     outer loops={it_outer:>5d}, time={time_h:.4f} s")

# === Policy and value comparison ===
g_num_h = grid[pol_h]

B = alpha / (1 - alpha * beta)
A = (np.log(1 - alpha * beta) + beta * B * np.log(alpha * beta)) / (1 - beta)
V_sol = A + B * np.log(grid)
g_ana = a * (grid ** alpha)

# === Plot results ===


plt.subplot(1, 2, 2)
plt.plot(grid, g_num_h,  '--', label="Howard g(k)")
plt.plot(grid, g_ana,    ':', label="Analytical g(k)")
plt.title("Policy Function Comparison")
plt.xlabel("Capital k"); plt.ylabel("Next-period capital k'")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()