# -*- coding: utf-8 -*-
"""
Layouts du playground - composants de la page.
"""

from .curve_generation import create_curve_generation_panel
from .model_params import create_model_params_panel
from .backtest_params import create_backtest_params_panel
from .results import create_results_panel

__all__ = [
    'create_curve_generation_panel',
    'create_model_params_panel',
    'create_backtest_params_panel',
    'create_results_panel',
]

