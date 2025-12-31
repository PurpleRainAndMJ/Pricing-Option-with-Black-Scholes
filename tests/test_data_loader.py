import pytest
from unittest.mock import MagicMock, patch
from src.data_loader import download_price_series_binance

@patch('src.data_loader.Client')
def test_download_price_series_mock(mock_client):
    """Vérifie la transformation des données sans appeler l'API Binance."""
    # Simulation d'une réponse de l'API (klines)
    mock_klines = [
        [1609459200000, "29000.0", "29500.0", "28800.0", "29300.0", "100", 1609459260000, "2900000", 10, "50", "1450000", "0"]
    ]
    
    instance = mock_client.return_value
    instance.get_historical_klines.return_value = mock_klines
    
    res = download_price_series_binance("BTCUSDT", lookback_days=1)
    
    assert len(res) == 1
    assert res.iloc[0] == 29300.0
    assert res.name == "close"