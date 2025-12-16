# -*- coding: utf-8 -*-
"""
Panel de résultats (graphiques et tableaux).
"""

from dash import dcc, html


def create_results_panel():
    """
    Crée le panneau de résultats avec les graphiques.
    
    Returns:
        html.Div contenant les graphiques
    """
    return html.Div([
        # Graphe segments
        html.Div([
            html.H4('Série synthétique & Segments', style={'color': '#FF8C00'}),
            dcc.Graph(
                id='play_segments_graph',
                style={'height': '450px'},
                config={'responsive': False},
                figure={
                    'data': [],
                    'layout': {
                        'template': 'plotly_dark',
                        'paper_bgcolor': '#000',
                        'plot_bgcolor': '#000',
                        'font': {'color': '#FFF'},
                        'title': 'Cliquer sur "Générer la courbe"',
                        'height': 420
                    }
                }
            ),
        ], style={'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px'}),
        
        # Graphe équité
        html.Div([
            html.H4('Équité', style={'color': '#FF8C00'}),
            dcc.Graph(
                id='play_equity_graph',
                style={'height': '420px'},
                config={'responsive': False},
                figure={
                    'data': [],
                    'layout': {
                        'template': 'plotly_dark',
                        'paper_bgcolor': '#000',
                        'plot_bgcolor': '#000',
                        'font': {'color': '#FFF'},
                        'title': 'En attente de backtest...',
                        'height': 400
                    }
                }
            ),
            html.Div(id='play_trades_table', style={'marginTop': '8px'}),
            html.Div(id='play_summary', style={'marginTop': '8px'}),
        ], style={'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px'}),
    ], style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px', 'marginTop': '12px'})

