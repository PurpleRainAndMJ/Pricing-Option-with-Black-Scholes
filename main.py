from src.data_loader import download_price_series_binance
from src.utils import estimate_bs_parameters
from src.pricing_engine import bs_call_price, mc_call_price_simple, mc_call_price_control_variate
from src.visualization import plot_convergence

def main():
    # 1. Data
    prices = download_price_series_binance("BTCUSDT")
    S0 = prices.iloc[-1]
    _, sigma = estimate_bs_parameters(prices)
    
    # 2. Params
    K, T, r = S0 * 1.05, 1.0, 0.02
    
    # 3. Pricing
    bs_p = bs_call_price(S0, K, T, r, sigma)
    mc_p, mc_se = mc_call_price_simple(S0, K, T, r, sigma)
    cv_p, cv_se, _ = mc_call_price_control_variate(S0, K, T, r, sigma)
    
    print(f"Prix Black-Scholes : {bs_p:.2f}")
    print(f"Prix MC Simple : {mc_p:.2f} (SE: {mc_se:.4f})")
    print(f"Prix MC Control Variate : {cv_p:.2f} (SE: {cv_se:.4f})")
    
    # 4. Visualisation
    plot_convergence(S0, K, T, r, sigma, bs_p)

if __name__ == "__main__":
    main()