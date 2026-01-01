# Pricing d'Options Monte Carlo & Réduction de Variance

Projet de finance quantitative implémentant le modèle de Black-Scholes et des simulations de Monte Carlo avec des données réelles de marché (via l'API Binance).

## 🚀 Fonctionnalités
- **Récupération de données** : Intégration API Binance pour le BTC/USDT.
- **Modèle Théorique** : Calcul du prix via la formule fermée de Black-Scholes.
- **Simulation Monte Carlo** :
  - MC Simple.
  - **Variable de Contrôle** (Control Variate) : Réduction drastique de la variance.
  - **Importance Sampling** : Optimisation pour les options "Out-of-the-money".
- **Tests Unitaires** : Couverture des calculs mathématiques et simulation de l'API (Mocking).

## 📊 Performance de la réduction de variance
L'implémentation de la **Variable de Contrôle** permet de réduire l'erreur standard de l'estimation de près de 90% par rapport à un Monte Carlo classique.

## 🛠 Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .