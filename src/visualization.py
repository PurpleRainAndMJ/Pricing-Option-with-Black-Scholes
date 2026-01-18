from src.pricing_engine import mc_call_price_simple, mc_call_price_control_variate, mc_call_price_importance_sampling, mc_call_price_antithetic 
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from src.pricing_engine import (
    mc_call_price_simple, 
    mc_call_price_antithetic, 
    mc_call_price_importance_sampling, 
    mc_call_price_control_variate,
    get_finite_difference_grid
)

def plot_convergence(S0, K, T, r, sigma, true_price, theta, max_paths=10000):
    Ns = np.arange(100, max_paths + 100, 100)
    est_simple, est_cv, est_is, est_av = [], [], [], []

    for n in Ns:
        # Simulation sans graine fixe pour chaque point (bruit indépendant)
        p_s, _ = mc_call_price_simple(S0, K, T, r, sigma, n_paths=n, seed=None)
        p_cv, _, _ = mc_call_price_control_variate(S0, K, T, r, sigma, n_paths=n, seed=None)
        p_is, _ = mc_call_price_importance_sampling(S0, K, T, r, sigma, n_paths=n, theta=theta, seed=None)
        p_av, _ = mc_call_price_antithetic(S0, K, T, r, sigma, n_paths=n, seed=None)
        
        est_simple.append(p_s)
        est_cv.append(p_cv)
        est_is.append(p_is)
        est_av.append(p_av)

    plt.figure(figsize=(14, 8))
    plt.plot(Ns, est_simple, label="MC Simple", color="gray", alpha=0.3)
    plt.plot(Ns, est_av, label="MC Anthitétique", color="blue", alpha=0.6)
    plt.plot(Ns, est_is, label="MC Importance Sampling", color="orange", alpha=0.7)
    plt.plot(Ns, est_cv, label="MC Variable de Contrôle", color="green", linewidth=2)
    plt.axhline(true_price, color="red", linestyle="--", label="Prix Black-Scholes")
    
    plt.title("Comparaison des oscillations et réduction de variance")
    plt.xlabel("Nombre de simulations (N)")
    plt.ylabel("Prix estimé")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


import time

def plot_mc_simple_convergence(S0, K, T, r, sigma, true_price, max_paths=20000):
    """Génère la courbe de convergence réelle et mesure le temps d'exécution."""
    Ns = np.arange(100, max_paths + 200, 200)
    prices = []
    times = []

    for n in Ns:
        start = time.time()
        # On ne fixe pas le seed pour voir le bruit réel
        p, _ = mc_call_price_simple(S0, K, T, r, sigma, n_paths=n, seed=None)
        times.append(time.time() - start)
        prices.append(p)

    # Graphique de Convergence
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, prices, label="Estimation MC Simple", color="#3498db", alpha=0.7)
    plt.axhline(true_price, color="#e74c3c", linestyle="--", linewidth=2, label=f"Asymptote (Prix BS: {true_price:.2f})")
    plt.title("Convergence de l'estimateur vers la valeur théorique")
    plt.xlabel("Nombre de simulations (N)")
    plt.ylabel("Prix de l'Option")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("mc_convergence_asymptote.png")

    # Graphique du temps d'exécution
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, times, color="#2ecc71", label="Temps de calcul")
    plt.title("Analyse de la complexité temporelle O(N)")
    plt.xlabel("Nombre de simulations (N)")
    plt.ylabel("Temps (secondes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("mc_execution_time.png")

def plot_diffusion_3d(S, time_grid, grid, S0, K):
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. On inverse l'axe du temps pour que 0 = Aujourd'hui et T = Maturité
    # On affiche le temps restant réel (T - t)
    tau_grid = time_grid[::-1] 
    S_mesh, Tau_mesh = np.meshgrid(S, tau_grid)
    
    # 2. On affiche la surface
    # grid[::-1] permet d'avoir le payoff (grid[0]) à Tau=0 (Maturité)
    surface = ax.plot_surface(S_mesh, Tau_mesh, grid[::-1], cmap='magma', 
                              antialiased=True, alpha=0.9, rstride=1, cstride=1)
    
    # 3. ZOOM : On limite l'affichage à la zone intéressante [0.5K, 1.5K]
    ax.set_xlim(K * 0.5, K * 1.5)
    ax.set_zlim(0, K * 0.5) # On évite que l'axe Z n'écrase la courbure
    
    ax.set_title(f"Diffusion de la valeur (K={K:.0f})")
    ax.set_xlabel("Prix du Sous-jacent (S)")
    ax.set_ylabel("Temps avant Maturité (Tau)")
    ax.set_zlabel("Prix de l'Option")
    
    # Position du point actuel (Aujourd'hui)
    price_today = np.interp(S0, S, grid[-1, :])
    ax.scatter(S0, tau_grid[0], price_today, color='cyan', s=100, label="Prix actuel (S0)")
    
    ax.view_init(elev=25, azim=-120) # Meilleur angle pour voir la "vague"
    plt.colorbar(surface, shrink=0.5)
    plt.legend()
    plt.show()

def plot_complexity_comparison(S0, K, T, r, sigma, theta, max_n=20000):
    """Compare le temps d'exécution (Complexité) de tous les algorithmes."""
    # On définit une échelle de N (Nombre de chemins pour MC / Pas de temps pour FD)
    Ns = np.arange(500, max_n + 500, 1000)
    
    times_simple = []
    times_anti = []
    times_is = []
    times_cv = []
    times_fd = []

    print("Début des tests de performance...")
    for n in Ns:
        # 1. MC Simple
        start = time.time()
        mc_call_price_simple(S0, K, T, r, sigma, n_paths=n)
        times_simple.append(time.time() - start)

        # 2. MC Anthitétique
        start = time.time()
        mc_call_price_antithetic(S0, K, T, r, sigma, n_paths=n)
        times_anti.append(time.time() - start)

        # 3. MC Importance Sampling
        start = time.time()
        mc_call_price_importance_sampling(S0, K, T, r, sigma, n_paths=n, theta=theta)
        times_is.append(time.time() - start)

        # 4. MC Variable de Contrôle
        start = time.time()
        mc_call_price_control_variate(S0, K, T, r, sigma, n_paths=n)
        times_cv.append(time.time() - start)

        # 5. Différences Finies (On fait varier le nombre de pas de temps N)
        start = time.time()
        get_finite_difference_grid(S0, K, T, r, sigma, M=100, N=n)
        times_fd.append(time.time() - start)

    plt.figure(figsize=(12, 7))
    plt.plot(Ns, times_simple, label="MC Simple", color="gray", linestyle='--')
    plt.plot(Ns, times_anti, label="MC Antithétique", color="blue")
    plt.plot(Ns, times_is, label="MC Importance Sampling", color="orange")
    plt.plot(Ns, times_cv, label="MC Variable de Contrôle", color="green", linewidth=2)
    plt.plot(Ns, times_fd, label="Différences Finies (EDP)", color="red", linewidth=1.5)

    plt.title("Comparaison de la Complexité Temporelle : Temps vs N")
    plt.xlabel("Nombre de simulations / Pas de temps (N)")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_complexities.png")
    plt.show()

def plot_mc_complexity_comparison(S0, K, T, r, sigma, theta, max_n=30000):
    Ns = np.arange(1000, max_n + 1000, 2000)
    
    t_simple, t_anti, t_is, t_cv = [], [], [], []

    for n in Ns:
        # MC Simple
        start = time.time()
        mc_call_price_simple(S0, K, T, r, sigma, n_paths=n)
        t_simple.append(time.time() - start)

        # MC Anthitétique
        start = time.time()
        mc_call_price_antithetic(S0, K, T, r, sigma, n_paths=n)
        t_anti.append(time.time() - start)

        # MC Importance Sampling
        start = time.time()
        mc_call_price_importance_sampling(S0, K, T, r, sigma, n_paths=n, theta=theta)
        t_is.append(time.time() - start)

        # MC Variable de Contrôle
        start = time.time()
        mc_call_price_control_variate(S0, K, T, r, sigma, n_paths=n)
        t_cv.append(time.time() - start)

    plt.figure(figsize=(10, 6))
    plt.plot(Ns, t_simple, label="Simple", color="gray", linestyle='--')
    plt.plot(Ns, t_anti, label="Antithétique", color="blue")
    plt.plot(Ns, t_is, label="Importance Sampling", color="orange")
    plt.plot(Ns, t_cv, label="Variable de Contrôle", color="green", linewidth=2)

    plt.title("Comparaison de la Complexité Temporelle (O(N))")
    plt.xlabel("Nombre de simulations (N)")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("mc_complexities_only.png")