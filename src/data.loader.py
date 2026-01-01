import pandas as pd
from binance.client import Client

def download_price_series_binance(
    symbol: str = "BTCUSDT", 
    interval: str = "1d", 
    lookback_days: int = 5 * 365, 
    api_key: str = None, 
    api_secret: str = None
) -> pd.Series:
    """Télécharge l'historique des prix de clôture depuis Binance."""
    client = Client(api_key, api_secret)
    start_str = f"{lookback_days} day ago UTC"
    
    klines = client.get_historical_klines(symbol, interval, start_str)
    if not klines:
        raise ValueError(f"Aucune donnée pour {symbol}. Vérifiez vos accès ou le symbole.")

    cols = ["open_time", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base", "taker_buy_quote", "ignore"]
    
    df = pd.DataFrame(klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df.set_index("open_time", inplace=True)
    
    return df["close"].sort_index()