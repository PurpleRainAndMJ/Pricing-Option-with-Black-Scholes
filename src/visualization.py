import matplotlib.pyplot as plt
import numpy as np
from src.pricing_engine import mc_call_price_simple, mc_call_price_control_variate, mc_call_price_importance_sampling

def plot_convergence(S0, K, T, r, sigma, true_price, max_paths=5000):
    Ns = np.linspace(50, max_paths, 100, dtype=int)
    est_simple, est_cv = [], []

    for n in Ns:
        p_s, _ = mc_call_price_simple(S0, K, T, r, sigma, n_paths=n)
        p_cv, _, _ = mc_call_price_control_variate(S0, K, T, r, sigma, n_paths=n)
        est_simple.append(p_s)
        est_cv.append(p_cv)

    plt.figure(figsize=(12, 6))
    plt.plot(Ns, est_simple, label="MC Simple", alpha=0.7)
    plt.plot(Ns, est_cv, label="MC Variable de Contrôle", color="green")
    plt.axhline(true_price, color="red", linestyle="--", label="Prix BS (Exact)")
    plt.title(f"Convergence du pricing : Simple vs Variable de Contrôle (K={K})")
    plt.xlabel("Nombre de simulations")
    plt.ylabel("Prix estimé")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()