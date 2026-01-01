import numpy as np
from scipy.stats import norm
from typing import Tuple

def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Formule fermée de Black-Scholes pour un Call Européen."""
    if T <= 0: return max(S0 - K, 0.0)
    if sigma * np.sqrt(T) == 0: return max(S0 - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_call_price_simple(S0, K, T, r, sigma, n_paths=20_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    return payoffs.mean(), payoffs.std(ddof=1) / np.sqrt(n_paths)

def mc_call_price_control_variate(S0, K, T, r, sigma, n_paths=20_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    X = np.exp(-r * T) * np.maximum(ST - K, 0.0) # Payoff actualisé
    Y = np.exp(-r * T) * ST                    # Variable de contrôle (Prix actualisé)

    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    var_Y = Y.var(ddof=1)
    a_star = cov_XY / var_Y if var_Y > 0 else 0.0

    Z_cv = X - a_star * (Y - S0)
    return Z_cv.mean(), Z_cv.std(ddof=1) / np.sqrt(n_paths), a_star

def mc_call_price_importance_sampling(S0, K, T, r, sigma, n_paths=20_000, theta=1.0, seed=42):
    rng = np.random.default_rng(seed)
    Z_tilted = rng.standard_normal(n_paths) + theta
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_tilted)

    payoffs = np.maximum(ST - K, 0.0)
    w = np.exp(0.5 * theta**2 - theta * Z_tilted)
    discounted_weighted = np.exp(-r * T) * payoffs * w
    return discounted_weighted.mean(), discounted_weighted.std(ddof=1) / np.sqrt(n_paths)