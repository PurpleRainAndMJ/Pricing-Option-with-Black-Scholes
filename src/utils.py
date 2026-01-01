import numpy as np
import pandas as pd
from typing import Tuple

def estimate_bs_parameters(close_prices: pd.Series, trading_days: int = 252) -> Tuple[float, float]:
    """Calcule le drift (mu) et la volatilité (sigma) annuels à partir des prix."""
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    mu_hat_daily = log_returns.mean()
    sigma_hat_daily = log_returns.std(ddof=1)

    mu_hat_annual = mu_hat_daily * trading_days
    sigma_hat_annual = sigma_hat_daily * np.sqrt(trading_days)

    return mu_hat_annual, sigma_hat_annual