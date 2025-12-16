# -*- coding: utf-8 -*-
"""
Module playground - Bac à sable pour tester les modèles ML sur données synthétiques.

Structure:
  - layouts/          # Composants de layout (génération courbe, hyperparamètres, etc.)
  - callbacks/        # Callbacks Dash (génération, entraînement, backtest, etc.)
  - __init__.py       # Point d'entrée, expose le layout
"""

# Les callbacks sont enregistrés au moment où `main` est importé
# (car `main.py` importe `playground_callbacks.callbacks`).
from .main import layout, layout_content  # noqa: F401

__all__ = [
    'layout',
    'layout_content',
]

