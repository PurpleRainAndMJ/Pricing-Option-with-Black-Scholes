import numpy as np
import pandas as pd
from binance.client import Client
from scipy.stats import norm
import matplotlib.pyplot as plt

def download_price_series_binance(symbol="BTCUSDT", interval="1d", lookback_days=5 * 365, api_key=None, api_secret=None,):
    client = Client(api_key, api_secret)

    start_str = f"{lookback_days} day ago UTC"
    klines = client.get_historical_klines(symbol, interval, start_str)

    if len(klines) == 0:
        raise ValueError("Aucune donnée récupérée depuis Binance. Vérifiez le symbole ou l'intervalle.")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df.set_index("open_time", inplace=True)

    close = df["close"].sort_index()
    return close

def estimate_bs_parameters(close_prices, trading_days=252):
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    mu_hat_daily = log_returns.mean()
    sigma_hat_daily = log_returns.std(ddof=1)

    mu_hat_annual = mu_hat_daily * trading_days
    sigma_hat_annual = sigma_hat_daily * np.sqrt(trading_days)

    return mu_hat_annual, sigma_hat_annual


def bs_call_price(S0, K, T, r, sigma):
    if T <= 0:
        return max(S0 - K, 0.0)

    if sigma * np.sqrt(T) == 0:
        return max(S0 - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call


def mc_call_price_simple(S0, K, T, r, sigma, n_paths=20_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs

    price = discounted_payoffs.mean()
    std_est = discounted_payoffs.std(ddof=1)
    std_error = std_est / np.sqrt(n_paths)
    return price, std_est, std_error

def mc_call_price_control_variate(S0, K, T, r, sigma, n_paths=20_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    payoff = np.maximum(ST - K, 0.0)
    X = np.exp(-r * T) * payoff       
    Y = np.exp(-r * T) * ST          

    mean_X = X.mean()
    mean_Y = Y.mean()

    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    var_Y = Y.var(ddof=1)
    a_star = cov_XY / var_Y if var_Y > 0 else 0.0

    Z_cv = X - a_star * (Y - S0)
    price_cv = Z_cv.mean()

    std_est_cv = Z_cv.std(ddof=1)
    std_error_cv = std_est_cv / np.sqrt(n_paths)
    return price_cv, std_est_cv, std_error_cv, a_star


def mc_call_price_importance_sampling(S0, K, T, r, sigma, n_paths=20_000,theta=1.0, seed=42):
    rng = np.random.default_rng(seed)
    Z_tilted = rng.standard_normal(n_paths) + theta
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T +
                     sigma * np.sqrt(T) * Z_tilted)

    payoffs = np.maximum(ST - K, 0.0)
    w = np.exp(0.5 * theta**2 - theta * Z_tilted)
    discounted_weighted = np.exp(-r * T) * payoffs * w
    price = discounted_weighted.mean()
    std_est = discounted_weighted.std(ddof=1)
    std_error = std_est / np.sqrt(n_paths)
    return price, std_est, std_error


def plot_convergence_mc_vs_cv(S0, K, T, r, sigma, true_price=None, max_paths=10_000, seed=42):
    
    Ns = np.arange(20, max_paths + 1)
    est_plain = []
    est_cv = []

    for n in Ns:
        rng_n = np.random.default_rng(seed + n)
        Z = rng_n.standard_normal(n)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T +
                         sigma * np.sqrt(T) * Z)
        payoff = np.maximum(ST - K, 0.0)

        X = np.exp(-r * T) * payoff      
        Y = np.exp(-r * T) * ST          

        price_plain = X.mean()

        cov_XY = np.cov(X, Y, ddof=1)[0, 1]
        var_Y = Y.var(ddof=1)
        a_star = cov_XY / var_Y if var_Y > 0 else 0.0

        Z_cv = X - a_star * (Y - S0)
        price_cv = Z_cv.mean()

        est_plain.append(price_plain)
        est_cv.append(price_cv)

    plt.figure(figsize=(10, 5))
    plt.plot(Ns, est_plain, label="MC simple", linewidth=0.8)
    plt.plot(Ns, est_cv, label="MC variable de contrôle", linewidth=0.8)

    if true_price is not None:
        plt.axhline(true_price, linestyle="--", label="Prix théorique (BS)")

    plt.xlabel("Nombre de simulations n")
    plt.ylabel("Prix estimé du call")
    plt.title("Convergence du prix de l'option par Monte-Carlo")
    plt.grid(True)
    plt.legend()


def plot_convergence_mc_vs_is(S0, K, T, r, sigma, true_price=None, max_paths=10_000, theta=1.0, seed=42):
    Ns = np.arange(20, max_paths + 1)  
    est_plain = []
    est_is = []

    for n in Ns:
        rng_n = np.random.default_rng(seed + n)

        Z_plain = rng_n.standard_normal(n)
        ST_plain = S0 * np.exp((r - 0.5 * sigma**2) * T +
                               sigma * np.sqrt(T) * Z_plain)
        payoff_plain = np.maximum(ST_plain - K, 0.0)
        X_plain = np.exp(-r * T) * payoff_plain
        est_plain.append(X_plain.mean())

        Z_tilted = rng_n.standard_normal(n) + theta
        ST_is = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_tilted)
        payoff_is = np.maximum(ST_is - K, 0.0)
        w = np.exp(0.5 * theta**2 - theta * Z_tilted)
        X_is = np.exp(-r * T) * payoff_is * w
        est_is.append(X_is.mean())

    plt.figure(figsize=(10, 5))
    plt.plot(Ns, est_plain, label="MC simple", linewidth=0.8)
    plt.plot(Ns, est_is, label=f"MC importance sampling (theta={theta:.1f})", linewidth=0.8)
    if true_price is not None:
        plt.axhline(true_price, linestyle="--", label="Prix théorique (BS)")

    plt.xlabel("Nombre de simulations n")
    plt.ylabel("Prix estimé du call")
    plt.title("Convergence : MC simple vs importance sampling")
    plt.grid(True)
    plt.legend()


if __name__ == "__main__":
    symbol = "BTCUSDT"
    print(f"=== Téléchargement des données Binance pour {symbol} ===")

    close = download_price_series_binance(symbol=symbol, interval="1d", lookback_days=5 * 365)
    S0 = close.iloc[-1]

    mu_hat, sigma_hat = estimate_bs_parameters(close)
    print(f"Prix spot S0 = {S0:.4f}")
    print(f"Estimation µ annuel       = {mu_hat:.4%}")
    print(f"Estimation sigma annuel   = {sigma_hat:.4%}")

    K = 1.05 * S0
    T = 1.0
    r = 0.02

    call_bs = bs_call_price(S0, K, T, r, sigma_hat)
    print("\n=== Prix du call par Black–Scholes (formule fermée) ===")
    print(f"Call BS (K={K:.2f}, T={T:.2f} an, r={r:.2%}, sigma={sigma_hat:.2%}) = {call_bs:.4f}")

    # Monte-Carlo simple
    n_paths = 20_000
    call_mc, std_mc, se_mc = mc_call_price_simple(S0, K, T, r, sigma_hat, n_paths=n_paths)
    print("\n=== Monte-Carlo simple ===")
    print(f"Prix MC       = {call_mc:.4f}")
    print(f"Écart-type    = {std_mc:.4f}")
    print(f"Erreur std    = {se_mc:.4f}")


    # Variable de contrôle
    call_cv, std_cv, se_cv, a_star = mc_call_price_control_variate(S0, K, T, r, sigma_hat, n_paths=n_paths)
    print("\n=== Monte-Carlo avec variable de contrôle ===")
    print(f"Prix MC (control variate) = {call_cv:.4f}")
    print(f"Écart-type                 = {std_cv:.4f}")
    print(f"Erreur std                 = {se_cv:.4f}")
    print(f"Coefficient optimal a*     = {a_star:.4f}")
    print(f"Réduction de variance par Control Variate (vs MC simple) ≈ {(1 - (std_cv**2 / std_mc**2)) * 100:.2f}%")

    # Importance Sampling
    call_im, std_im, se_im = mc_call_price_importance_sampling(S0, K, T, r, sigma_hat, n_paths=n_paths)
    print("\n=== Monte-Carlo avec variable de contrôle ===")
    print(f"Prix MC (control variate) = {call_im:.4f}")
    print(f"Écart-type                 = {std_im:.4f}")
    print(f"Erreur std                 = {se_im:.4f}")
    print(f"Réduction de variance par Importance Sampling (vs MC simple) ≈ {(1 - (std_cv**2 / std_mc**2)) * 100:.2f}%")


    print("\n=== Comparaison finale ===")
    print(f"Call BS (exact)          : {call_bs:.4f}")
    print(f"MC simple                : {call_mc:.4f}  (SE ≈ {se_mc:.4f})")
    print(f"MC variable de contrôle  : {call_cv:.4f}  (SE ≈ {se_cv:.4f})")

    plot_convergence_mc_vs_cv(S0, K, T, r, sigma_hat, true_price=call_bs, max_paths=10_000)
    

    plot_convergence_mc_vs_is(S0, K, T, r, sigma_hat, true_price=call_bs, max_paths=10_000, theta=1)
    
    plot_convergence_mc_vs_is(S0, 3 * K, T, r, sigma_hat, true_price=call_bs, max_paths=10_000, theta=1)
    plt.show()

