import numpy as np
import time
from scipy.optimize import fsolve, brentq
from tauchen_rouwenhorst import rouwenhorst

def solve_random_ramsey_egm(
    beta: float ,
    alpha: float ,
    delta: float,
    n: int, # number of grid points for capital 200
    rho: float,
    mu: float,
    sigma_es: float,
    N: int,
    grid_style: str = "log",   # "linear" or "log"
    tol: float = 1e-9,
    max_iter: int = 1000,
    verbose: bool = False,
):
    """
    Solve the Ramsey model with AR(1) productivity shocks using the Endogenous Grid Method (EGM).

    Parameters
    ----------
    beta : float
        Discount factor.
    alpha : float
        Capital share in production.
    delta : float
        Depreciation rate of capital.
    n : int
        Number of grid points for capital.
    rho : float
        Persistence parameter of the productivity shock.
    mu : float
        Mean of the productivity shock.
    sigma_es : float
        Standard deviation of the productivity shock.
    N : int
        Number of states for the productivity shock.
    grid_style : str
        Type of grid for capital: "linear" or "log".
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.
    verbose : bool
        If True, print iteration details.

    Returns
    -------
    g_on_k_theta : np.ndarray
        Optimal policy k'(k, theta) on the exogenous capital grid (length n).
    grid_k : np.ndarray
        Exogenous grid for capital k (length n).
    iters : int
        Number of iterations performed.
    theta_values : np.ndarray
        Discretized productivity states (length N).
    trans_prob : np.ndarray
        Transition probability matrix for productivity states (N x N).  
    """

    # Production and derivatives (allowing for 0 < delta <= 1)
    def f(k, theta_local):        # resources next period: x = f(k) = theta*k^alpha + (1-delta)k
        return theta_local * (k ** alpha) + (1.0 - delta) * k

    # def fprime(k, theta_local):   # derivative of f(k)
    #     return alpha * theta_local * (k ** (alpha - 1.0)) + (1.0 - delta)
    # # Marginal utility and its inverse for log utility
    # def uprime(c):
    #     return 1.0 / c

    def uprime_inv(mu):
        return 1.0 / mu
    
    # cash at hand function
    def cash_func(k, theta_local):
        return theta_local * (k ** alpha) + (1.0 - delta) * k 

    # Steady state and grids
    theta_bar = np.exp(mu)
    k_ss = ((alpha * theta_bar) / (1.0 / beta - 1.0 + delta)) ** (1.0 / (1.0 - alpha))
    k_min = 0.2 * k_ss
    k_max = 3.0 * k_ss

    if grid_style == "linear":
        grid_k = np.linspace(k_min, k_max, n)     # exogenous grid for k
    else:
        grid_k = np.exp(np.linspace(np.log(k_min), np.log(k_max), n))

    

    # Discretize the productivity shock using Rouwenhorst method
    log_theta_discrete, trans_prob = rouwenhorst(rho, mu, sigma_es, N) #P: (N,N)
    theta_values = np.exp(log_theta_discrete) #shape (N,)

    # Control grid for k' (use same as k grid)
    grid_kp = grid_k.copy()

    # Initial policy guess on k-grid and theta states
    g_on_k_theta = np.empty((n, N))  # g(k, theta)
    for s in range(N):
        g_on_k_theta[:, s] = 0.3 * f(grid_k, theta_values[s])  # initial guess

    # Helper: relative sup norm
    def sup_norm_rel(new, old):
        denom = 1.0 + np.max(np.abs(old))
        return np.max(np.abs(new - old)) / denom

    err = np.inf
    it = 0
    t0 = time.time()

    while err > tol and it < max_iter:

        it += 1
        g_new = np.empty_like(g_on_k_theta)  # new policy

        # compute next period consumption and return on k' for all states
        g_kp_all_next = np.vstack([
                np.interp(grid_kp, grid_k, g_on_k_theta[:, sp]) for sp in range(N)
            ])  # shape (N, n)
        x_next = theta_values.reshape(-1, 1) * (grid_kp ** alpha) + (1.0 - delta) * grid_kp  # (N, n)
        c_next = x_next - g_kp_all_next
        R_next = theta_values.reshape(-1, 1) * (alpha * grid_kp ** (alpha - 1.0)) + (1.0 - delta)          # (N, n)


        for s in range(N):

            grid_x = f(grid_k, theta_values[s])    # grid for cash-at-hand x = f(k) at mean theta
            
            P_row = trans_prob[s, :]  # transition probabilities from state s to all states, (N,)
            # 
            rhs = beta * np.sum(P_row.reshape(-1, 1) * (1.0 / c_next) * R_next, axis=0)      # (n,)

            # consumption and cash at today：c_t(k') 与 x_t(k')
            c_today = uprime_inv(rhs)                     # (n,)
            x_today = grid_kp + c_today                   # (n,)

            # k_endog = np.empty_like(x_today)  # endogenous grid for k

            
            # for j in range(n):
            #     root = fsolve(lambda k: cash_func(k, theta_values[s]) - x_today[j], x0=0)
            #     k_endog[j] = float(root[0])

            g_new_on_x = np.interp(grid_x, x_today, grid_kp)  # g_new(x) on x grid

            g_new[:, s] = np.interp(cash_func(grid_k, theta_values[s]), grid_x, g_new_on_x)

        err = sup_norm_rel(g_new, g_on_k_theta)
        g_on_k_theta = g_new.copy()

        if verbose:
            print(f"[EGM] iter={it}  sup-norm rel err = {err:.2e}")

    if verbose:
        print(f"Converged in {it} iterations, {time.time() - t0:.4f} seconds. Final err = {err:.2e}")


    return g_on_k_theta, grid_k, it, theta_values, trans_prob




# # Example usage:
# if __name__ == "__main__":
#     g_num, grid_k, iters, theta_values, trans_prob = solve_random_ramsey_egm(verbose=True)
#     print("g_num (first 10):", g_num[:10])
#     print("g_num shape:", g_num.shape)

#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 6))
#     for s in range(g_num.shape[1]):
#         plt.plot(grid_k, g_num[:, s], label=f'Theta state {theta_values[s]:.4f}')
#     plt.title(f'Optimal Policy Functions k\'(k) for Different Productivity States (EGM) with std = {0.20}')
#     plt.xlabel('Capital k')
#     plt.ylabel('Optimal Next Period Capital k\'')
#     plt.legend()
#     plt.grid()
#     plt.show()


#     # ----- Simulation parameters -----
#     T = 51000
#     burn_in = 1000
#     tail_len = 2000
#     rng = np.random.default_rng(42)

#     # Simulate Markov chain: we need indices of states
#     from tauchen_rouwenhorst import markov_simulation
#     # markov_simulation can return the realized (log) states; map to indices
#     log_chain = markov_simulation(trans_prob, np.log(theta_values), np.ones(len(theta_values))/len(theta_values), T)
#     # Map log_chain to nearest discrete index
#     log_states = np.log(theta_values)
#     state_idx = np.array([np.argmin(np.abs(log_states - v)) for v in log_chain])
#     theta_path = theta_values[state_idx]

#     # Capital path
#     k_path = np.empty(T+1)
#     # Start from steady state w.r.t mean theta
#     alpha, delta = 0.323, 0.025  # keep consistent (could read from solve_random_ramsey_egm args)
#     theta_bar = np.mean(theta_values)
#     k_ss = ((alpha * theta_bar) / (1/0.984 - 1 + delta)) ** (1/(1-alpha))
#     k_path[0] = k_ss

#     # Pre-allocate consumption and output
#     c_path = np.empty(T)
#     y_path = np.empty(T)

#     for t in range(T):
#         z_t = theta_path[t]
#         s_t = state_idx[t]
#         k_t = k_path[t]
#         # Interpolate next capital from policy for current state
#         k_next = np.interp(k_t, grid_k, g_num[:, s_t])
#         # Resources (cash at hand)
#         x_t = z_t * (k_t ** alpha) + (1 - delta) * k_t
#         c_t = x_t - k_next
#         # Guard against numerical negatives
#         if c_t <= 0:
#             c_t = 1e-12
#         y_t = z_t * (k_t ** alpha)
#         k_path[t+1] = k_next
#         c_path[t] = c_t
#         y_path[t] = y_t

#     # Discard burn-in
#     k_eff = k_path[burn_in:-1]
#     c_eff = c_path[burn_in:]
#     y_eff = y_path[burn_in:]
#     theta_eff = theta_path[burn_in:]
#     # Take last tail_len
#     k_plot = k_eff[-tail_len:]
#     c_plot = c_eff[-tail_len:]
#     y_plot = y_eff[-tail_len:]
#     theta_plot = theta_eff[-tail_len:]
#     t_axis = np.arange(tail_len)

#     # Plots
#     fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
#     axs[0].plot(t_axis, theta_plot, color='tab:purple')
#     axs[0].set_ylabel('Productivity z')
#     axs[0].grid(alpha=0.3)

#     axs[1].plot(t_axis, k_plot, color='tab:blue')
#     axs[1].set_ylabel('Capital k')
#     axs[1].grid(alpha=0.3)

#     axs[2].plot(t_axis, y_plot, color='tab:green')
#     axs[2].set_ylabel('Output y')
#     axs[2].grid(alpha=0.3)

#     axs[3].plot(t_axis, c_plot, color='tab:orange')
#     axs[3].set_ylabel('Consumption c')
#     axs[3].set_xlabel('Periods (last 2000)')
#     axs[3].grid(alpha=0.3)

#     fig.suptitle("Simulated paths (last 2000 periods after burn-in)")
#     plt.tight_layout()
#     plt.show()


#     N = len(theta_values)

#     grid_k_finer = np.linspace(grid_k[0], grid_k[-1], 500)

#     from stationary_distribution import v1_stationary_cdf_core


#     stationary_dist, mu_K = v1_stationary_cdf_core(g_num,
#     grid_k_finer,
#     theta_values,
#     trans_prob
# )
#     average_k = np.sum(grid_k_finer * mu_K)
#     print("Average capital from stationary distribution:", average_k)

#     print("Stationary marginal distribution over k (first 10):", mu_K)

#     import numpy as np
#     import matplotlib.pyplot as plt

# # 假设已有：
# # k_grid: (Nk,) 资本网格（递增）
# # mu_K:   (Nk,) 在 k_grid 上的稳态概率质量（和为1）

# # 计算每个点对应的“桶边界”（用相邻中点）
#     edges = np.empty(len(grid_k_finer) + 1)
#     edges[1:-1] = 0.5 * (grid_k_finer[1:] + grid_k_finer[:-1])
# # 两端外推
#     edges[0]  = grid_k_finer[0]  - 0.5 * (grid_k_finer[1]   - grid_k_finer[0])
#     edges[-1] = grid_k_finer[-1] + 0.5 * (grid_k_finer[-1] - grid_k_finer[-2])
#     bin_widths = np.diff(edges)              # 每个桶的宽度
#     density    = mu_K / bin_widths          # 高度=密度，确保面积=概率
#     plt.figure(figsize=(7,4))
#     plt.bar(grid_k_finer, density, width=bin_widths, align='center',
#         edgecolor='k', alpha=0.7)
#     plt.xlabel('Capital k')
#     plt.ylabel('Density (area = probability)')
#     plt.title('Stationary Distribution of Capital')
#     plt.tight_layout()
#     plt.show()




