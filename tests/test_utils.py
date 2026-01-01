import pytest
import numpy as np
import pandas as pd
from src.utils import estimate_bs_parameters

def test_estimate_parameters_synthetic():
    """Vérifie si l'estimateur retrouve la vol injectée dans un GBM simulé."""
    true_sigma = 0.30
    np.random.seed(42)
    
    # Simulation d'une série de prix avec 30% de vol
    returns = np.random.normal(0, true_sigma / np.sqrt(252), 1000)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    
    _, estimated_sigma = estimate_bs_parameters(prices)
    
    # On autorise une marge d'erreur due à l'échantillonnage aléatoire
    assert pytest.approx(estimated_sigma, rel=0.1) == true_sigma