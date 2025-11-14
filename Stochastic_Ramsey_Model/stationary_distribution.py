
import numpy as np

def stationary_cdf_core(policy_array, k_grid_finer, z_vals, Pz, tol=1e-9, max_iter=2000, verbose=False):
    """
    Compute stationary joint CDF F(k,z) for capital and shock using inverse-policy pushforward.

    Inputs
    - policy_array : (Nk, Nz) array; column s is k'(k, z_s) evaluated on k_grid_finer
    - k_grid_finer : (Nk,) capital grid where the policy is evaluated (ascending)
    - z_vals       : (Nz,) discrete shock states (only length is used)
    - Pz           : (Nz, Nz) Markov transition matrix over z
    - tol          : sup-norm tolerance on CDF updates
    - max_iter     : maximum iterations
    - verbose      : print iteration info if True

    Returns
    - F    : (Nk, Nz) stationary CDF over (k,z)
    - mu_K : (Nk,)    stationary marginal distribution over capital
    """
    k_grid_finer = np.asarray(k_grid_finer)
    policy_array = np.asarray(policy_array)

    Nk = k_grid_finer.shape[0]
    Nz = len(z_vals)

    # init CDF: each column increases linearly to a uniform shock mass
    pi = np.full(Nz, 1.0 / Nz)                   # current column top masses
    base = np.linspace(0.0, 1.0, Nk)[:, None]    # (Nk,1)
    F = base @ pi[None, :]                       # (Nk,Nz)

    for it in range(max_iter):
        
        inv_maps = np.empty((Nz, Nk))
        for s in range(Nz):
            
            # 1) inverse maps: inv_maps[r, j] = g^{-1}(k'_j | z_r) on k'-grid = k_grid_finer
            kprime = policy_array[:, s]
            inv_maps[s, :] = np.interp(
                k_grid_finer, kprime, k_grid_finer,
                left=k_grid_finer[0], right=k_grid_finer[-1]
            )


        # 2) push-forward CDF
        F_new = np.zeros_like(F)
        for s in range(Nz):
            acc = np.zeros(Nk)
            for r in range(Nz):
                acc += Pz[r, s] * np.interp(
                    inv_maps[r, :], k_grid_finer, F[:, r],
                    left=F[0, r], right=F[-1, r]
                )
            F_new[:, s] = acc

        # 3) update shock marginal and rescale each column to match π_next[s]
        pi = pi @ Pz
        for s in range(Nz):
            top = F_new[-1, s]
            F_new[:, s] *= (pi[s] / top)

        # 4) convergence check
        err = float(np.max(np.abs(F_new - F)))
        F = F_new
        if err < tol:
            break

    # 5) marginal over k: differencing along k and summing over z
    mu_K = np.zeros(Nk)
    mu_K[0]  = np.sum(F[0, :])
    mu_K[1:] = np.sum(F[1:, :] - F[:-1, :], axis=1)


    return F, mu_K
