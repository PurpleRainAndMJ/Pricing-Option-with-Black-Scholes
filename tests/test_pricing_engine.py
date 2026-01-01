import pytest
import numpy as np
from src.pricing_engine import bs_call_price, mc_call_price_simple, mc_call_price_control_variate

# Constantes pour les tests (cas d'école)
S0, K, T, r, SIGMA = 100, 100, 1, 0.05, 0.2

def test_bs_call_known_value():
    """Vérifie BS avec une valeur connue (Calculateur standard)."""
    expected_price = 10.45058
    calculated_price = bs_call_price(S0, K, T, r, SIGMA)
    assert pytest.approx(calculated_price, rel=1e-4) == expected_price

def test_mc_convergence():
    """Vérifie que le MC simple est proche du prix BS (Loi des grands nombres)."""
    bs_p = bs_call_price(S0, K, T, r, SIGMA)
    mc_p, mc_se = mc_call_price_simple(S0, K, T, r, SIGMA, n_paths=50_000)
    
    # Le prix MC doit être à moins de 3 erreurs standards du prix BS (IC 99.7%)
    assert abs(mc_p - bs_p) < 3 * mc_se

def test_variance_reduction():
    """Vérifie que la Variable de Contrôle réduit effectivement la variance."""
    _, se_simple = mc_call_price_simple(S0, K, T, r, SIGMA, n_paths=10_000)
    _, se_cv, _ = mc_call_price_control_variate(S0, K, T, r, SIGMA, n_paths=10_000)
    
    # L'erreur standard du CV doit être significativement plus faible
    assert se_cv < se_simple