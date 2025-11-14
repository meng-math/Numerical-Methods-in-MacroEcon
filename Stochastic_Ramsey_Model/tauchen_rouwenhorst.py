'''
In this module, we implement Tauchen's and Rouwenhorst's methods for discretizing a continuous AR(1) process, as well as a function to simulate a Markov chain given a transition probability matrix.
'''


import numpy as np

def tauchen(rho:float, mu:float, sigma_es:float, m:int, N:int):

    """
    Tauchen's method for discretizing a continuous AR(1) process x_t = rho * x_{t-1} + e_t, where e_t ~ N(0, sigma_es^2).

    Parameters
    ----------
    rho : float
        The persistence parameter.
    mu : float
        The mean of the process.
    sigma_es : float
        The standard deviation of innovations.
    m : int
        The Tauchen parameter.
    N : int
        The number of grid points.

    Returns
    -------
    x_grid : numpy.ndarray
        The discretized state space grid.
    P : numpy.ndarray
        The transition probability matrix.
    """

    from scipy.stats import norm

    # step 1: set up the equidistant grid
    sigma_x = sigma_es / ((1 - rho ** 2) ** 0.5)
    x_min = mu - m * sigma_x
    x_max = mu + m * sigma_x
    x_grid = np.linspace(x_min, x_max, N)
    step_size = x_grid[1] - x_grid[0]

    # step 2: compute the transition probabilities
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j == 0:
                P[i, j] = norm.cdf((x_grid[0] - rho * x_grid[i] + step_size / 2) / sigma_es)
            elif j == N - 1:
                P[i, j] = 1 - norm.cdf((x_grid[N - 1] - rho * x_grid[i] - step_size / 2) / sigma_es)
            else:
                upper_bound = (x_grid[j] - rho * x_grid[i] + step_size / 2) / sigma_es
                lower_bound = (x_grid[j] - rho * x_grid[i] - step_size / 2) / sigma_es
                P[i, j] = norm.cdf(upper_bound) - norm.cdf(lower_bound)

    return x_grid, P

def rouwenhorst(rho:float, mu:float, sigma_es:float, N:int):

    """
    Rouwenhorst's method for discretizing a continuous AR(1) process x_t = rho * x_{t-1} + e_t, where e_t ~ N(0, sigma_es^2).

    Parameters
    ----------
    rho : float
        The persistence parameter.
    mu : float
        The mean of the process.
    sigma_es : float
        The standard deviation of innovations.
    N : int
        The number of grid points.
    
    Returns
    -------
    x_grid : numpy.ndarray
        The discretized state space grid.
    P : numpy.ndarray
        The transition probability matrix.
    """


    # step 1: set up the equidistant grid

    sigma_x = sigma_es / ((1 - rho ** 2) ** 0.5)
    x_min = mu - ((N - 1) ** 0.5) * sigma_x
    x_max = mu + ((N - 1) ** 0.5) * sigma_x
    x_grid = np.linspace(x_min, x_max, N)

    # step 2: compute the transition probabilities by recursion

    p = (1 + rho) / 2
    if N == 2:
        P = np.array([[p, 1 - p],
                      [1 - p, p]])
    else:
        P_prev = rouwenhorst(rho, mu, sigma_es, N - 1)[1] # here we use the recursive property
        P = np.zeros((N, N))
        P[:-1, :-1] += p * P_prev
        P[:-1, 1:] += (1 - p) * P_prev
        P[1:, :-1] += (1 - p) * P_prev
        P[1:, 1:] += p * P_prev
        P[1:-1, :] *= 0.5
    return x_grid, P


def markov_simulation(P:np.ndarray, state_grid:np.ndarray, initial_dist:np.ndarray, T:int):
    '''
    
    Simulate a Markov chain given a transition probability matrix, state grid, initial distribution, and number of periods. 
    Parameters
    ----------
    P : numpy.ndarray
        Transition probability matrix.
    state_grid : numpy.ndarray
        Discretized state space grid.
    initial_dist : numpy.ndarray
        Initial distribution over states.
    T : int
        Number of periods to simulate.
        
    Returns
    -------
    chain : numpy.ndarray
        Simulated Markov chain of states over T periods.
    '''

    chain = np.zeros(T, dtype=float)
    x = np.random.rand(1,T)
    cumulative_P = np.cumsum(P, axis=1)
    cumulative_initial = np.cumsum(initial_dist)
    for i in range(T):
        state_idx = np.searchsorted(cumulative_initial, x[0,i]) if i == 0 else np.searchsorted(cumulative_P[state_idx], x[0,i])
        chain[i] = state_grid[state_idx]
    return chain




if __name__ == "__main__":

    # Example usage

    '''parameters'''

    rho = 0.979
    mu = 0.0
    sigma_es = 0.0072
    m = 3
    N = 7

    '''AR(1) discretization by Tauchen and Rouwenhorst methods'''

    x_grid_tauchen, P_tauchen = tauchen(rho, mu, sigma_es, m, N)
    print("State Space Grid of log:\n", x_grid_tauchen)
    print("Transition Probability Matrix:\n", P_tauchen)

    x_grid_rouwenhorst, P_rouwenhorst = rouwenhorst(rho, mu, sigma_es, N)
    print("State Space Grid of log:\n", x_grid_rouwenhorst)
    print("Transition Probability Matrix:\n", P_rouwenhorst)

    '''Simulate Markov Chain and visualization'''

    T = 25000
    np.random.seed(42)

    initial_dist = np.ones(N) / N
    chain_tauchen = markov_simulation(P_tauchen, x_grid_tauchen, initial_dist, T)
    print(chain_tauchen)

    import matplotlib.pyplot as plt

    plt.plot(chain_tauchen)
    plt.title("Simulated Markov Chain using Tauchen's Method")
    plt.show()

    chain_rouwenhorst = markov_simulation(P_rouwenhorst, x_grid_rouwenhorst, initial_dist, T)
    plt.plot(chain_rouwenhorst)
    plt.title("Simulated Markov Chain using Rouwenhorst's Method")
    plt.show()

    '''Estimate AR(1) parameters from simulated chains by OLS regression'''

    import statsmodels.api as sm

    y = chain_tauchen[1:]
    X = chain_tauchen[:-1]
    ols = sm.OLS(y, X).fit()
    print("Estimated rho (Tauchen):", ols.summary())

    y_r = chain_rouwenhorst[1:]
    X_r = chain_rouwenhorst[:-1]
    ols_r = sm.OLS(y_r, X_r).fit()
    print("Estimated rho (Rouwenhorst):", ols_r.summary())


    print( f" estimated std (Rouwenhorst): {np.sqrt((1 - (ols_r.params[0]) ** 2) * np.var(chain_rouwenhorst))}")
    print( f" estimated std (Tauchen): {np.sqrt((1 - (ols.params[0]) ** 2) * np.var(chain_tauchen))}")