# -*- coding: utf-8 -*-
"""
Callbacks du playground - logique interactive.

Importer ce module enregistre automatiquement tous les callbacks.
"""

# Importer les modules de callbacks pour les enregistrer
from . import generation
from . import training
from . import backtest
from . import ui_toggles
from . import generalization
from . import saved_models

__all__ = [
    'generation',
    'training',
    'backtest',
    'ui_toggles',
    'generalization',
    'saved_models',
]

