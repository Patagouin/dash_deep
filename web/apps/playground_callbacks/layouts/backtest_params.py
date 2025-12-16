# -*- coding: utf-8 -*-
"""
Panel de paramètres de simulation financière (backtest).
"""

from dash import dcc, html


def create_backtest_params_panel():
    """
    Crée le panneau de paramètres de backtest.
    
    Returns:
        html.Div contenant le panneau
    """
    # Tooltips
    t_cash = 'Capital de départ pour la simulation'
    t_trade_amt = 'Montant engagé par trade'
    t_ktrades = 'Nombre maximum de trades simultanés/journaliers'
    t_spread = 'Spread bid-ask en % appliqué à chaque trade (coût de transaction)'
    
    return html.Div([
        html.Hr(),
        
        # Titre
        html.Div([
            html.Label('💰 Simulation Financière (Backtest)', style={'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '4px'}),
        ]),
        
        # Paramètres
        html.Div([
            html.Div([
                html.Label('Capital initial (€)', title=t_cash),
                html.Div(dcc.Input(id='play_initial_cash', value=10_000, type='number', step=100, min=0, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_cash),
            ]),
            html.Div([
                html.Label('Montant par trade (€)', title=t_trade_amt),
                html.Div(dcc.Input(id='play_trade_amount', value=1_000, type='number', step=50, min=0, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_trade_amt),
            ]),
            html.Div([
                html.Label('K trades/jour', title=t_ktrades),
                html.Div(dcc.Input(id='play_k_trades', value=2, type='number', step=1, min=1, max=10, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_ktrades),
            ]),
            html.Div([
                html.Label('Spread (%)', title=t_spread),
                html.Div(dcc.Input(id='play_spread_pct', value=0.0, type='number', step=0.01, min=0.0, max=1.0, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_spread),
            ]),
            html.Div([
                html.Label('Stratégie', title='LONG = acheter puis vendre (gagner si hausse). SHORT = vendre puis racheter (gagner si baisse). LONG&SHORT = les deux selon la prédiction.'),
                dcc.Dropdown(
                    id='play_strategy',
                    options=[
                        {'label': '📈 LONG (hausse)', 'value': 'long'},
                        {'label': '📉 SHORT (baisse)', 'value': 'short'},
                        {'label': '📊 LONG & SHORT', 'value': 'both'},
                    ],
                    value='long',
                    persistence=True, persistence_type='session',
                    style={'width': '100%', 'color': '#FF8C00'}
                ),
            ]),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))', 'gap': '8px'}),

        # Boutons
        html.Div([
            html.Button(
                '📈 Lancer le Backtest',
                id='play_run_backtest',
                n_clicks=0,
                style={
                    'width': '100%',
                    'backgroundColor': '#2196F3',
                    'color': '#000000',
                    'padding': '12px',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'border': '1px solid #FFFFFF',
                },
                disabled=True,
            ),
        ], id='panel_play_btn_new', style={'marginTop': '12px'}),

        html.Div([
            html.Button('Backtester modèle sauvegardé', id='play_backtest_saved', n_clicks=0, style={'width': '100%'}),
        ], id='panel_play_btn_saved', style={'display': 'none'}),
        
        # Stores
        dcc.Store(id='play_predictions_store', storage_type='memory'),
        # Store pour réinitialiser l’état des boutons (compat ancienne implémentation)
        dcc.Store(id='play_reset_buttons', storage_type='memory', data=True),
        
        # Suivi entraînement
        html.Hr(),
        html.Div([
            html.H4('Suivi entraînement', style={'color': '#FF8C00'}),
            html.Div(id='play_training_progress', style={'marginBottom': '8px'}),
            dcc.Graph(
                id='play_training_history',
                style={'height': '300px'},
                config={'responsive': False},
                figure={
                    'data': [],
                    'layout': {
                        'template': 'plotly_dark',
                        'paper_bgcolor': '#000',
                        'plot_bgcolor': '#000',
                        'font': {'color': '#FFF'},
                        'title': "En attente d'entraînement...",
                        'height': 280
                    }
                }
            ),
        ], style={'marginTop': '12px'}),
        
        # Prédiction sur la courbe actuelle (réutilise le modèle entraîné)
        html.Div([
            html.H4('Prédiction (courbe actuelle)', style={'color': '#FF8C00', 'marginTop': '12px'}),
            html.Button(
                '🔮 Prédire avec le modèle entraîné (courbe actuelle)',
                id='play_test_generalization',
                n_clicks=0,
                disabled=True,
                style={
                    'width': '100%',
                    'backgroundColor': '#FF8C00',
                    'color': '#000000',
                    'padding': '10px',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'border': 'none',
                    'borderRadius': '8px',
                    'marginTop': '8px'
                }
            ),
            html.Div(
                id='play_gen_summary',
                style={
                    'marginTop': '8px',
                    'color': '#CCCCCC',
                    'fontSize': '12px'
                }
            )
        ], style={'marginTop': '4px'}),
        
        # Stores pour modèle en mémoire
        dcc.Store(id='play_model_ready', storage_type='memory', data=False),
        dcc.Store(id='play_model_path', storage_type='memory'),
    ])

