import stationary_distribution as sd
import numpy as np
import matplotlib.pyplot as plt
import random_Ramsey_EGM as rregm
import tauchen_rouwenhorst as tr
import Ramsey_VFI_Howard_Improvement as rvfi

#------ parameters ------

beta = 0.984
alpha = 0.323
delta = 0.025
n = 500  # number of grid points for capital
rho = 0.979
mu = 0.0
sigma_es = 0.0072
N = 7
tol = 1e-9
max_iter = 1000
verbose = False


#(2a)------ discretize the productivity process ------

theta_grid, Pz = tr.rouwenhorst(rho, mu, sigma_es, N)
theta_grid = np.exp(theta_grid)  # convert log(z) to z
print("Productivity states (theta):", theta_grid)
print("Transition matrix (Pz):\n", Pz)


#(2b)------ solve the deterministic Ramsey model for different  ------

for theta in theta_grid:
    print(f"\nSolving deterministic Ramsey model for theta = {theta:.4f} ...")
    g_num = rvfi.solve_ramsey_vfi_howard(
        beta=beta,
        alpha=alpha,
        delta=delta,
        theta=theta,
        n=n,
        tol=tol,
        verbose=False,
    )
    print("Optimal policy function k'(k) for theta =", theta)
    print(g_num)
    plt.plot(g_num, label=f"theta={theta:.4f}")
plt.title("Optimal Policy Functions k'(k) for Different Productivity States")
plt.xlabel("Capital Grid Index")
plt.ylabel("Next Period Capital k'")
plt.legend()
plt.show()



#2(c)------ solve the stochastic Ramsey model with aggregate risk under two different sigma_es ------

for sigma_es in [0.0072, 0.20]:
    print(f"\nSolving stochastic Ramsey model with sigma_es = {sigma_es:.4f} ...")
    g_num, grid_k, iters, theta_values, trans_prob = rregm.solve_random_ramsey_egm(
    beta=beta,
    alpha=alpha,
    delta=delta,
    n=n,
    rho=rho,
    mu=mu,
    sigma_es=sigma_es,
    N=N,
    tol=tol,
    max_iter=max_iter,
    verbose=True
)
    print(f"Solved in {iters} iterations.")
    print("Optimal policy function k'(k, theta):")
    plt.figure(figsize=(10, 6))
    for s in range(g_num.shape[1]):
        plt.plot(grid_k, g_num[:, s], label=f'Theta state {theta_values[s]:.4f}')
    plt.title(f'Optimal Policy Functions k\'(k) for Different Productivity States (EGM) with std = {sigma_es:.4f}')
    plt.xlabel('Capital k')
    plt.ylabel('Optimal Next Period Capital k\'')
    plt.legend()
    plt.grid()
    plt.show()

#2(d)------ simulate the economy over T = 51000 periods under two different sigma_es ------

 # ----- Simulation parameters -----
    T = 51000
    burn_in = 1000
    tail_len = 2000
    rng = np.random.default_rng(42)

    # Simulate Markov chain: we need indices of states
    from tauchen_rouwenhorst import markov_simulation
    # markov_simulation can return the realized (log) states; map to indices
    log_chain = markov_simulation(trans_prob, np.log(theta_values), np.ones(len(theta_values))/len(theta_values), T)
    # Map log_chain to nearest discrete index
    log_states = np.log(theta_values)
    state_idx = np.array([np.argmin(np.abs(log_states - v)) for v in log_chain])
    theta_path = theta_values[state_idx]

    # Capital path
    k_path = np.empty(T+1)
    # Start from steady state w.r.t mean theta
    alpha, delta = 0.323, 0.025  # keep consistent (could read from solve_random_ramsey_egm args)
    theta_bar = np.mean(theta_values)
    k_ss = ((alpha * theta_bar) / (1/0.984 - 1 + delta)) ** (1/(1-alpha))
    k_path[0] = k_ss

    # Pre-allocate consumption and output
    c_path = np.empty(T)
    y_path = np.empty(T)

    for t in range(T):
        z_t = theta_path[t]
        s_t = state_idx[t]
        k_t = k_path[t]
        # Interpolate next capital from policy for current state
        k_next = np.interp(k_t, grid_k, g_num[:, s_t])
        # Resources (cash at hand)
        x_t = z_t * (k_t ** alpha) + (1 - delta) * k_t
        c_t = x_t - k_next
        # Guard against numerical negatives
        if c_t <= 0:
            c_t = 1e-12
        y_t = z_t * (k_t ** alpha)
        k_path[t+1] = k_next
        c_path[t] = c_t
        y_path[t] = y_t

    # Discard burn-in
    k_eff = k_path[burn_in:-1]
    c_eff = c_path[burn_in:]
    y_eff = y_path[burn_in:]
    theta_eff = theta_path[burn_in:]
    # Take last tail_len
    k_plot = k_eff[-tail_len:]
    c_plot = c_eff[-tail_len:]
    y_plot = y_eff[-tail_len:]
    theta_plot = theta_eff[-tail_len:]
    t_axis = np.arange(tail_len)

    # Plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t_axis, theta_plot, color='tab:purple')
    axs[0].set_ylabel('Productivity z')
    axs[0].grid(alpha=0.3)

    axs[1].plot(t_axis, k_plot, color='tab:blue')
    axs[1].set_ylabel('Capital k')
    axs[1].grid(alpha=0.3)

    axs[2].plot(t_axis, y_plot, color='tab:green')
    axs[2].set_ylabel('Output y')
    axs[2].grid(alpha=0.3)

    axs[3].plot(t_axis, c_plot, color='tab:orange')
    axs[3].set_ylabel('Consumption c')
    axs[3].set_xlabel('Periods (last 2000)')
    axs[3].grid(alpha=0.3)

    fig.suptitle("Simulated paths (last 2000 periods after burn-in)")
    plt.tight_layout()
    plt.show()



#2(e)------ compute and plot the stationary distribution of capital under two different sigma_es ------

for sigma_es in [0.0072, 0.20]:

    print(f"\nSolving stochastic Ramsey model with sigma_es = {sigma_es:.4f} ...")
    g_num, grid_k, iters, theta_values, trans_prob = rregm.solve_random_ramsey_egm(
    beta=beta,
    alpha=alpha,
    delta=delta,
    n=n,
    rho=rho,
    mu=mu,
    sigma_es=sigma_es,
    N=N,
    tol=tol,
    max_iter=max_iter,
    verbose=True
)
    print(f"\nComputing stationary distribution with sigma_es = {sigma_es:.4f} ...")
    
    stationary_dist, mu_K = sd.stationary_cdf_core(policy_array = g_num,
    k_grid_finer = grid_k,
    z_vals = theta_values,
    Pz= trans_prob
)
    average_k = np.sum(grid_k * mu_K)

    print(stationary_dist)


    print("Average capital from stationary distribution:", average_k)



    edges = np.empty(len(grid_k) + 1)
    edges[1:-1] = 0.5 * (grid_k[1:] + grid_k[:-1])

    edges[0]  = grid_k[0]  - 0.5 * (grid_k[1]   - grid_k[0])
    edges[-1] = grid_k[-1] + 0.5 * (grid_k[-1] - grid_k[-2])
    bin_widths = np.diff(edges)             
    density    = mu_K / bin_widths          
    plt.figure(figsize=(7,4))
    plt.bar(grid_k, density, width=bin_widths, align='center',
        edgecolor='k', alpha=0.7)
    plt.xlabel('Capital k')
    plt.ylabel('Density (area = probability)')
    plt.title('Stationary Distribution of Capital')
    plt.tight_layout()
    plt.show()


