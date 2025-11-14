# ------------------------------------------------------------------------------
# Howard Policy Improvement in Value Function Iteration (VFI)
# ------------------------------------------------------------------------------

import numpy as np
import time

def solve_ramsey_vfi_howard(
    beta: float ,
    alpha: float,
    delta: float,
    theta: float,
    n: int = 2000,
    tol: float = 1e-9,
    big_neg: float = -1e16,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve the deterministic Ramsey model via Value Function Iteration with Howard improvement.

    Parameters
    ----------
    beta : float
        Discount factor.
    alpha : float
        Capital share in production.
    delta : float
        Depreciation rate of capital.
    theta : float
        Productivity parameter.
    n : int
        Number of grid points for capital.
    tol : float
        Tolerance level for convergence.
    big_neg : float
        Large negative number to penalize infeasible consumption.
    verbose : bool
        If True, print convergence information.

    Returns
    -------
    g_num : np.ndarray
        Optimal policy function k'(k) evaluated on the grid (length n).
    """
    # Build capital grid around the steady state (valid for delta < 1)
    k_ss = ((alpha * theta) / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    k_min = 0.2 * k_ss
    k_max = 3.0 * k_ss
    grid = np.linspace(k_min, k_max, n)

    # Pre-compute utility matrix u(c) with c = theta * k^alpha + (1 - delta) * k - k'
    # If c <= 0, assign a large negative number to exclude infeasible choices
    k_col = grid[:, None]   # shape (n, 1)
    kp_row = grid[None, :]  # shape (1, n)
    c = theta * (k_col ** alpha) + (1.0 - delta) * k_col - kp_row
    util_mat = np.where(c > 0.0, np.log(c), big_neg)

    # Initialize value and policy
    V = np.zeros(n)
    val_mat = util_mat + beta * V[None, :]   # broadcast V over rows
    best_idx = np.argmax(val_mat, axis=1)    # initial policy indices
    err_g = np.inf

    start = time.time()

    # Outer loop: policy iteration with Howard improvement
    while err_g > tol:
        # Evaluate the value function under the current policy
        V_new = val_mat[np.arange(n), best_idx]
        g_num = grid[best_idx]
        err_V = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

        # Howard step: improve value by iterating with fixed policy
        while err_V > tol:
            V = V_new
            V_new = util_mat[np.arange(n), best_idx] + beta * V[best_idx]
            err_V = np.max(np.abs(V_new - V)) / (1.0 + np.max(np.abs(V)))

        # Policy improvement step using updated values
        val_mat = util_mat + beta * V_new[None, :]
        new_best_idx = np.argmax(val_mat, axis=1)
        new_g = grid[new_best_idx]

        # Check convergence of the policy
        err_g = np.max(np.abs(new_g - g_num)) / (1.0 + np.max(np.abs(g_num)))

        # Prepare for next iteration
        best_idx = new_best_idx
        V = V_new

    if verbose:
        print(f"VFI with Howard Improvement converged in {time.time() - start:.4f} seconds")

    return grid[best_idx]  # g_num


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
# if __name__ == "__main__":
#     beta = 0.984
#     alpha = 0.323
#     delta = 1.0
#     theta = 1.0

#     g_num = solve_ramsey_vfi_howard(
#         beta=beta,
#         alpha=alpha,
#         delta=delta,
#         theta=theta,
#         verbose=True
#     )

#     # Print first 10 values of the optimal policy function
#     print("Optimal policy function k'(k) (first 10 values):", g_num[:10])

#     # analytical solution for comparison
#     k_ss = ((alpha * theta) / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
#     k_min = 0.2 * k_ss
#     k_max = 3.0 * k_ss
#     grid = np.linspace(k_min, k_max, 2000)
#     g_ana = alpha * beta * (grid ** alpha)
#     print("Analytical policy function k'(k) (first 10 values):", g_ana[:10])
