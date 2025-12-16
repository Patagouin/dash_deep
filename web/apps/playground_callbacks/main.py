# -*- coding: utf-8 -*-
"""
Module principal playground - Point d'entrée et layout.

Ce fichier importe:
- Les layouts depuis playground/layouts/
- Les callbacks depuis playground/callbacks/ (enregistrés automatiquement via import)
"""

from dash import dcc, html
import pandas as pd

from app import shM
from web.components.navigation import create_navigation, create_page_help
from web.components.help_texts import get_playground_help

# Import des layouts
from .layouts.curve_generation import create_curve_generation_panel
from .layouts.model_params import create_model_params_panel
from .layouts.backtest_params import create_backtest_params_panel
from .layouts.results import create_results_panel

# Import des callbacks (l'import suffit à les enregistrer)
from . import callbacks  # noqa: F401


def _get_symbols_options():
    """Retourne les options de symboles pour les dropdowns."""
    try:
        df = shM.getAllShares()
        symbols = list(df['symbol'].values) if not df.empty else []
        return [{'label': s, 'value': s} for s in symbols]
    except Exception:
        return []


def _default_dates():
    """Retourne les dates par défaut (aujourd'hui - 20 jours, aujourd'hui)."""
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=20)
    return start.date(), today.date()


def layout_content():
    """Construit le layout complet de la page Playground."""
    start, end = _default_dates()
    help_text = get_playground_help()

    return html.Div([
        create_page_help("Aide Playground", help_text),
        html.H3('Playground', style={'color': '#FF8C00'}),

        # Store pour les données
        dcc.Store(id='play_df_store', storage_type='session'),

        # Panels gauche/droite
        html.Div([
            # Panel Génération de courbe
            create_curve_generation_panel(start, end),
            
            # Panel Modèle et backtest
            html.Div([
                create_model_params_panel(_get_symbols_options),
                create_backtest_params_panel(),
            ], style={'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px'}),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))', 'gap': '12px'}),

        # Panel Résultats
        create_results_panel(),

        create_navigation()
    ], style={'backgroundColor': 'black', 'padding': '20px', 'minHeight': '100vh'})


layout = layout_content()

