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

def mc_call_price_antithetic(S0, K, T, r, sigma, n_paths=20_000, seed=42):
    rng = np.random.default_rng(seed)
    # On génère n/2 paires de variables antithétiques
    n_pairs = n_paths // 2
    Z = rng.standard_normal(n_pairs)
    
    # Trajectoires standards (+) et antithétiques (-)
    ST_plus = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ST_minus = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-Z))
    
    payoffs_plus = np.exp(-r * T) * np.maximum(ST_plus - K, 0.0)
    payoffs_minus = np.exp(-r * T) * np.maximum(ST_minus - K, 0.0)
    
    # On fait la moyenne des paires
    W = (payoffs_plus + payoffs_minus) / 2.0
    
    return W.mean(), W.std(ddof=1) / np.sqrt(n_pairs)

def get_finite_difference_grid(S0, K, T, r, sigma, M=100, N=2000):
    """Calcule la grille de diffusion en espace Log (beaucoup plus stable)."""
    # 1. Définition de la grille en log-espace
    x_min = np.log(S0 / 3) # On couvre de S0/3 à S0*3
    x_max = np.log(S0 * 3)
    dx = (x_max - x_min) / M
    dt = T / N
    
    x = np.linspace(x_min, x_max, M + 1)
    S = np.exp(x) # Pour revenir aux prix réels sur l'axe du graphique
    time_grid = np.linspace(0, T, N + 1)
    
    grid = np.zeros((N + 1, M + 1))
    grid[0, :] = np.maximum(S - K, 0) # Condition à maturité

    # 2. Coefficients de l'équation de la chaleur transformée
    # Ces coefficients ne dépendent plus de S, ce qui évite l'explosion
    sig2 = sigma**2
    a = dt * (sig2 / (2 * dx**2) - (r - 0.5 * sig2) / (2 * dx))
    b = 1 - dt * (sig2 / dx**2 + r)
    c = dt * (sig2 / (2 * dx**2) + (r - 0.5 * sig2) / (2 * dx))

    # 3. Calcul de la diffusion (Remontée dans le temps)
    for j in range(N):
        for i in range(1, M):
            grid[j+1, i] = a * grid[j, i-1] + b * grid[j, i] + c * grid[j, i+1]
        
        # Conditions aux limites (Boundary conditions)
        grid[j+1, 0] = 0
        grid[j+1, M] = S[M] - K * np.exp(-r * (j+1) * dt)
        
    return S, time_grid, grid