from src.data_loader import download_price_series_binance
from src.utils import estimate_bs_parameters
from src.pricing_engine import bs_call_price, mc_call_price_simple, mc_call_price_control_variate, mc_call_price_importance_sampling, get_finite_difference_grid
from src.visualization import plot_convergence, plot_diffusion_3d, plot_mc_simple_convergence, plot_complexity_comparison, plot_mc_complexity_comparison
from src.pricing_engine import mc_call_price_antithetic
import numpy as np


def main():
    # 1. Données et Paramètres
    # Récupération des prix via l'API Binance
    prices = download_price_series_binance("BTCUSDT") 
    S0 = prices.iloc[-1]
    
    # Estimation de la volatilité avec 365 jours de trading pour la crypto
    _, sigma = estimate_bs_parameters(prices, trading_days=365) 
    K, T, r = S0 * 1.05, 1.0, 0.05
    
    # Calcul du paramètre de décalage optimal pour l'Importance Sampling
    theta_opt = (np.log(K/S0) - (r - 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    print("\nAnalyse des temps de calcul...")
    plot_complexity_comparison(S0, K, T, r, sigma, theta_opt, max_n=30000)

    print("\nAnalyse des temps de calcul...")
    plot_mc_complexity_comparison(S0, K, T, r, sigma, theta_opt, max_n=30000)
    # 2. Calcul des prix de référence
    # Prix théorique exact de Black-Scholes
    bs_p = bs_call_price(S0, K, T, r, sigma) 
    
    # Prix par l'analogie de la chaleur (EDP)
    # On effectue le calcul avant le tableau pour extraire la valeur
    S_grid, T_grid, full_grid = get_finite_difference_grid(S0, K, T, r, sigma, M=60, N=500)
    price_fd_value = np.interp(S0, S_grid, full_grid[-1, :]) # Extraction du prix à t=0
    print("Génération des courbes de convergence et de temps d'exécution...")
    plot_mc_simple_convergence(S0, K, T, r, sigma, bs_p, max_paths=30000)
    # 3. Calculs Monte Carlo (avec N=50 000 pour la précision finale)
    n_final = 50_000
    mc_p, mc_se = mc_call_price_simple(S0, K, T, r, sigma, n_paths=n_final)
    av_p, av_se = mc_call_price_antithetic(S0, K, T, r, sigma, n_paths=n_final)
    cv_p, cv_se, _ = mc_call_price_control_variate(S0, K, T, r, sigma, n_paths=n_final)
    is_p, is_se = mc_call_price_importance_sampling(S0, K, T, r, sigma, n_paths=n_final, theta=theta_opt)
    
    # 4. Affichage des résultats comparatifs
    print(f"\n{'Méthode':<25} | {'Prix':<10} | {'Erreur Std':<10}")
    print("-" * 50)
    print(f"{'Black-Scholes':<25} | {bs_p:<10.2f} | {'0.0000':<10}")
    print(f"{'Différences Finies':<25} | {price_fd_value:<10.2f} | {'N/A':<10}")
    print(f"{'MC Simple':<25} | {mc_p:<10.2f} | {mc_se:<10.4f}")
    print(f"{'MC Anthitétique':<25} | {av_p:<10.2f} | {av_se:<10.4f}")
    print(f"{'MC Importance Sampling':<25} | {is_p:<10.2f} | {is_se:<10.4f}")
    print(f"{'MC Variable de Contrôle':<25} | {cv_p:<10.2f} | {cv_se:<10.4f}")
    print("-" * 50)
    print("Génération des courbes de convergence et de temps d'exécution...")

    
    # 5. Visualisations
    # Graphique de convergence des méthodes de Monte Carlo
    plot_convergence(S0, K, T, r, sigma, bs_p, theta=theta_opt, max_paths=15000)
    
    # Graphique 3D de la diffusion de la chaleur (EDP)
    print("\nAffichage de la surface de diffusion 3D...")
    plot_diffusion_3d(S_grid, T_grid, full_grid, S0, K)

if __name__ == "__main__":
    main()