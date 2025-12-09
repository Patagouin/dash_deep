from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from app import app, shM
from web.services.timeseries import fetch_intraday_series, align_minute, fetch_intraday_series_with_perf, align_minute_with_perf, fetch_intraday_dataframe
from web.services.backtest import backtest_lag_correlation, backtest_time_window
from web.services.model_strategy import backtest_model_intraday
from web.components.navigation import create_navigation, create_page_help
from web.services.sim_builders import build_equity_figure, build_daily_outputs, build_trades_table, build_summary
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import dash
from dash import callback_context as ctx
import time
import json


def _get_symbols_options():
    try:
        df = shM.dfShares
        sorted_df = df.sort_values(by='symbol')
        return [
            {
                'label': f"{row.symbol}",
                'value': row.symbol
            }
            for row in sorted_df.itertuples()
        ]
    except Exception as e:
        logging.warning(f"Impossible de charger la liste des actions: {e}")
        return []


CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '24px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'marginBottom': '24px'
}

SECTION_TITLE_STYLE = {
    'fontSize': '0.8125rem',
    'fontWeight': '500',
    'color': '#94a3b8',
    'textTransform': 'uppercase',
    'letterSpacing': '0.05em',
    'marginBottom': '8px'
}

def layout_content():
    today = pd.Timestamp.today().normalize()
    fifteen_days_ago = today - pd.Timedelta(days=15)

    help_text = """
### Simulation (Backtesting)

Cette page est dédiée au test de stratégies de trading sur des données historiques pour évaluer leur rentabilité potentielle avant de les utiliser en réel.

#### Modes de Simulation

1.  **Lead-Lag (Corrélation)** :
    *   Stratégie basée sur des paires d'actions.
    *   On suppose que l'action A (Leader) influence l'action B (Suiveuse) avec un certain retard.
    *   Si A monte, on achète B. Si A baisse, on vend B.
    *   **Paramètres** :
        *   `Lag` : Retard en minutes entre A et B.
        *   `Seuil` : Variation minimum de A pour déclencher un ordre sur B.

2.  **Fenêtre Horaire** :
    *   Stratégie simple et systématique.
    *   Achète une action à une heure fixe (ex: 09:30) et la revend à une autre heure fixe (ex: 16:00).
    *   Permet de tester la tendance intraday moyenne d'un actif.

3.  **Modèle IA** :
    *   Utilise un modèle entraîné dans la page "Prédiction".
    *   Le modèle analyse le marché minute par minute et décide d'acheter ou vendre.
    *   C'est le test ultime pour valider votre IA.
    *   **Généralisation** : vous pouvez réutiliser un modèle sauvegardé sur d'autres fenêtres temporelles pour vérifier s'il généralise bien hors de sa période d'entraînement.

#### Paramètres Généraux
*   **Période** : Dates de début et fin du backtest.
*   **Horaires** : Heures autorisées pour trader (pour éviter les heures creuses ou la nuit).
*   **Capital initial** : Somme d'argent virtuelle de départ.
*   **Montant par trade** : Taille de chaque position.

#### Résultats
*   **Courbe de capital** : Graphique montrant l'évolution de votre portefeuille. Si ça monte, c'est gagné !
*   **Tableau journalier** : Détail des gains/pertes jour par jour.
*   **Transactions** : Liste complète de tous les ordres d'achat et de vente exécutés par la simulation.
"""

    return html.Div([
        create_page_help("Aide Simulation", help_text),
        
        # En-tête de page
        html.Div([
            html.H3('Simulation', style={
                'margin': '0',
                'textAlign': 'center'
            }),
            html.P('Backtesting & Test de Stratégies', style={
                'textAlign': 'center',
                'color': '#94a3b8',
                'marginTop': '8px',
                'marginBottom': '0'
            })
        ], style={'marginBottom': '32px'}),

        # Store persistant (session) pour conserver les résultats de simulation
        dcc.Store(id='sim_results_store', storage_type='session'),

        # Section des modes de simulation
        html.Div([
            html.Div([
                html.Span('🎯', style={'fontSize': '1.25rem'}),
                html.Span('Mode de Simulation', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '16px', 'display': 'flex', 'alignItems': 'center'}),
            
            dcc.Tabs(id='sim_tabs', value='leadlag', children=[
                dcc.Tab(label='📊 Lead-lag (2 actions)', value='leadlag'),
                dcc.Tab(label='⏰ Fenêtre horaire (1 action)', value='timewindow'),
                dcc.Tab(label='🤖 Modèle IA (1 action)', value='model')
            ], className='custom-tabs')
        ], style={**CARD_STYLE, 'marginBottom': '16px'}),
        
        # Section des paramètres
        html.Div([
            html.Div([
                html.Span('⚙️', style={'fontSize': '1.25rem'}),
                html.Span('Paramètres', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label('Action référence (A)', style=SECTION_TITLE_STYLE),
                    dcc.Dropdown(
                        id='sim_reference_symbol',
                        options=_get_symbols_options(),
                        placeholder='Choisir A (référence)',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ]),
                html.Div([
                    html.Label('Action tradée (B)', style=SECTION_TITLE_STYLE),
                    dcc.Dropdown(
                        id='sim_trade_symbol',
                        options=_get_symbols_options(),
                        placeholder='Choisir B (achetée/vendue)',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ], id='panel_trade_symbol'),
                html.Div([
                    html.Label('Modèle sauvegardé', style=SECTION_TITLE_STYLE),
                    dcc.Dropdown(
                        id='sim_saved_model',
                        options=[],
                        placeholder="Choisir un modèle pour l'action référence",
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ], id='panel_model', style={'display': 'none'}),
                html.Div(id='sim_model_meta', style={
                    'gridColumn': '1 / -1',
                    'color': '#94a3b8',
                    'backgroundColor': '#12121a',
                    'padding': '12px',
                    'borderRadius': '8px',
                    'fontSize': '0.8125rem'
                }),
                html.Div([
                    html.Label('Période', style=SECTION_TITLE_STYLE),
                    dcc.DatePickerRange(
                        id='sim_date_range',
                        start_date=fifteen_days_ago,
                        end_date=today,
                        display_format='YYYY-MM-DD',
                        persistence=True, persistence_type='session'
                    )
                ]),
                html.Div([
                    html.Label("Heure début d'achat", style=SECTION_TITLE_STYLE),
                    dcc.Input(
                        id='sim_buy_start_time',
                        type='text',
                        value='09:30',
                        placeholder='HH:MM',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ]),
                html.Div([
                    html.Label('Heure fin de vente', style=SECTION_TITLE_STYLE),
                    dcc.Input(
                        id='sim_sell_end_time',
                        type='text',
                        value='16:00',
                        placeholder='HH:MM',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ]),
                html.Div([
                    html.Label('Seuil hausse A sur 30 min (%)', style=SECTION_TITLE_STYLE),
                    dcc.Input(
                        id='sim_threshold_pct',
                        type='number',
                        value=0.1,
                        step=0.05,
                        min=0,
                        max=10,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    html.Small("Ex: 0.1 = 0,1%", style={'color': '#64748b', 'fontSize': '0.75rem'})
                ], id='panel_threshold'),
                html.Div([
                    html.Label('Sens du signal', style=SECTION_TITLE_STYLE),
                    dcc.RadioItems(
                        id='sim_direction',
                        options=[
                            {'label': ' ↑ Hausse', 'value': 'up'},
                            {'label': ' ↓ Baisse', 'value': 'down'},
                            {'label': ' ↕ Les deux', 'value': 'both'}
                        ],
                        value='up',
                        inline=True,
                        persistence=True, persistence_type='session',
                        labelStyle={'marginRight': '16px', 'color': '#f8fafc'}
                    )
                ], id='panel_direction'),
                html.Div([
                    html.Label('Décalage (minutes)', style=SECTION_TITLE_STYLE),
                    dcc.Input(
                        id='sim_lag_minutes',
                        type='number',
                        value=30,
                        step=1,
                        min=1,
                        max=240,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ], id='panel_lag'),
                html.Div([
                    html.Label('Capital initial (€)', style=SECTION_TITLE_STYLE),
                    dcc.Input(
                        id='sim_initial_cash',
                        type='number',
                        value=10000,
                        step=100,
                        min=0,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ]),
                html.Div([
                    html.Label('Montant par trade (€)', style=SECTION_TITLE_STYLE),
                    dcc.Input(
                        id='sim_trade_amount',
                        type='number',
                        value=1000,
                        step=50,
                        min=0,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ]),
                html.Div([
                    html.Button('🚀 Lancer la simulation', id='sim_run', n_clicks=0, style={
                        'width': '100%',
                        'background': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
                        'color': 'white',
                        'fontWeight': '600',
                        'fontFamily': 'Outfit, sans-serif',
                        'padding': '14px',
                        'borderRadius': '10px',
                        'border': 'none',
                        'fontSize': '1rem',
                        'cursor': 'pointer',
                        'boxShadow': '0 4px 15px rgba(99, 102, 241, 0.3)',
                        'transition': 'all 0.25s ease'
                    })
                ], style={'gridColumn': '1 / -1'})
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(240px, 1fr))',
                'gap': '16px'
            })
        ], style=CARD_STYLE),

        # Résultats de simulation
        html.Div([
            html.Div([
                html.Span('📈', style={'fontSize': '1.25rem'}),
                html.Span('Résultats', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                # Courbe de capital
                html.Div([
                    html.H5('Courbe de capital', style={
                        'color': '#f8fafc',
                        'marginBottom': '12px'
                    }),
                    dcc.Loading(
                        dcc.Graph(id='simulation_equity_graph', style={'height': '45vh'}),
                        type='default'
                    )
                ], style={
                    'backgroundColor': '#12121a',
                    'padding': '16px',
                    'borderRadius': '12px',
                    'border': '1px solid rgba(148, 163, 184, 0.1)'
                }),
                
                # Résumé
                html.Div([
                    html.H5('Résumé', style={
                        'color': '#f8fafc',
                        'marginBottom': '12px'
                    }),
                    html.Div(id='simulation_summary', style={'color': '#f8fafc'})
                ], style={
                    'backgroundColor': '#12121a',
                    'padding': '16px',
                    'borderRadius': '12px',
                    'border': '1px solid rgba(148, 163, 184, 0.1)'
                })
            ], style={
                'display': 'grid',
                'gridTemplateColumns': '2fr 1fr',
                'gap': '16px',
                'marginBottom': '16px'
            }),
            
            html.Div([
                # Courbe journalière
                html.Div([
                    html.H5('Courbe de capital (journalier)', style={
                        'color': '#f8fafc',
                        'marginBottom': '12px'
                    }),
                    dcc.Loading(
                        dcc.Graph(id='simulation_daily_equity_graph', style={'height': '45vh'}),
                        type='default'
                    )
                ], style={
                    'backgroundColor': '#12121a',
                    'padding': '16px',
                    'borderRadius': '12px',
                    'border': '1px solid rgba(148, 163, 184, 0.1)'
                }),
                
                # Résumé journalier
                html.Div([
                    html.H5('Résumé journalier', style={
                        'color': '#f8fafc',
                        'marginBottom': '12px'
                    }),
                    html.Div(id='simulation_daily_table', style={'color': '#f8fafc'})
                ], style={
                    'backgroundColor': '#12121a',
                    'padding': '16px',
                    'borderRadius': '12px',
                    'border': '1px solid rgba(148, 163, 184, 0.1)'
                })
            ], style={
                'display': 'grid',
                'gridTemplateColumns': '2fr 1fr',
                'gap': '16px',
                'marginBottom': '16px'
            }),
            
            # Transactions
            html.Div([
                html.H5('📋 Transactions', style={
                    'color': '#f8fafc',
                    'marginBottom': '12px'
                }),
                html.Div(id='simulation_trades_table')
            ], style={
                'backgroundColor': '#12121a',
                'padding': '16px',
                'borderRadius': '12px',
                'border': '1px solid rgba(148, 163, 184, 0.1)'
            })
        ], style=CARD_STYLE),

        # Spacer pour navigation
        html.Div(style={'height': '100px'}),

        create_navigation()
    ], style={
        'backgroundColor': '#0a0a0f',
        'minHeight': '100vh',
        'padding': '24px 32px',
        'width': '100%',
        'maxWidth': '100%',
        'margin': '0'
    })


# Pour compatibilité avec index.py qui attend `layout`
layout = layout_content()


@app.callback(
    [
        Output('panel_trade_symbol', 'style'),
        Output('panel_threshold', 'style'),
        Output('panel_lag', 'style'),
        Output('panel_model', 'style'),
        Output('panel_direction', 'style')
    ],
    [
        Input('sim_tabs', 'value')
    ]
)
def toggle_panels(sim_mode):
    show_style = { 'display': 'block' }
    hide_style = { 'display': 'none' }
    if sim_mode == 'timewindow':
        return hide_style, hide_style, hide_style, hide_style, hide_style
    if sim_mode == 'model':
        # Modèle: une seule action + modèle, pas de seuil/lag
        return hide_style, hide_style, hide_style, show_style, hide_style
    return show_style, show_style, show_style, hide_style, show_style


@app.callback(
    Output('sim_saved_model', 'options'),
    [Input('sim_reference_symbol', 'value')]
)
def populate_sim_models(symbol):
    try:
        if not symbol:
            return []
        rows = shM.list_models_for_symbol(symbol)
        options = []
        for r in rows or []:
            model_id = r[0]
            date_val = r[1]
            train_s = r[2]
            test_s = r[3]
            label = f"{model_id} — {str(date_val) if date_val else ''} — train={train_s if train_s is not None else '-'} val={test_s if test_s is not None else '-'}"
            options.append({'label': label, 'value': model_id})
        return options
    except Exception:
        return []


@app.callback(
    Output('sim_model_meta', 'children'),
    [Input('sim_saved_model', 'value')]
)
def show_sim_model_metadata(model_id):
    try:
        if not model_id:
            return ''
        meta = shM.get_model_metadata(model_id) or {}
        lines = []
        # Champs principaux
        lines.append(f"Modèle: {model_id}")
        if meta.get('date'):
            lines.append(f"Date: {meta.get('date')}")
        if meta.get('trainScore') is not None or meta.get('testScore') is not None:
            lines.append(f"Scores: train={meta.get('trainScore', '-')}, val={meta.get('testScore', '-')}")
        # Symbols
        symbols = meta.get('symbols') or (meta.get('data_info') or {}).get('symbols')
        if symbols:
            if isinstance(symbols, list):
                lines.append("Symbols: " + ", ".join([str(s) for s in symbols]))
            else:
                lines.append(f"Symbols: {symbols}")
        # HPS bref
        hps = meta.get('hps') or {}
        if hps:
            arch = hps.get('architecture'); layers = hps.get('layers'); units = hps.get('nb_units'); lr = hps.get('learning_rate'); loss = hps.get('loss')
            parts = []
            if arch: parts.append(f"arch={arch}")
            if layers is not None: parts.append(f"layers={layers}")
            if units is not None: parts.append(f"units={units}")
            if lr is not None: parts.append(f"lr={lr}")
            if loss is not None: parts.append(f"loss={loss}")
            if parts:
                lines.append("HPS: " + " | ".join([str(p) for p in parts]))
        # Data bref
        data_info = meta.get('data_info') or {}
        if data_info:
            lb = data_info.get('look_back_x'); st = data_info.get('stride_x'); ny = data_info.get('nb_y'); split = data_info.get('percent_train_test')
            parts = []
            if lb is not None: parts.append(f"look_back={lb}")
            if st is not None: parts.append(f"stride={st}")
            if ny is not None: parts.append(f"nb_y={ny}")
            if split is not None: parts.append(f"split={split}%")
            if parts:
                lines.append("Data: " + " | ".join([str(p) for p in parts]))
        return html.Pre("\n".join([str(x) for x in lines]), style={'whiteSpace': 'pre-wrap'})
    except Exception:
        return ''

@app.callback(
    [
        Output('simulation_equity_graph', 'figure'),
        Output('simulation_trades_table', 'children'),
        Output('simulation_summary', 'children'),
        Output('simulation_daily_equity_graph', 'figure'),
        Output('simulation_daily_table', 'children'),
        Output('sim_results_store', 'data')
    ],
    [
        Input('sim_run', 'n_clicks'),
        Input('sim_results_store', 'data'),
        Input('sim_tabs', 'value')
    ],
    [
        State('sim_reference_symbol', 'value'),
        State('sim_trade_symbol', 'value'),
        State('sim_date_range', 'start_date'),
        State('sim_date_range', 'end_date'),
        State('sim_threshold_pct', 'value'),
        State('sim_direction', 'value'),
        State('sim_lag_minutes', 'value'),
        State('sim_initial_cash', 'value'),
        State('sim_trade_amount', 'value'),
        State('sim_buy_start_time', 'value'),
        State('sim_sell_end_time', 'value'),
        State('sim_saved_model', 'value')
    ]
)
def run_simulation(n_clicks, stored_data, sim_mode, ref_symbol, trade_symbol, start_date, end_date, threshold_pct, sim_direction, lag_minutes, initial_cash, trade_amount, buy_start_time, sell_end_time, sim_model_id):
    cb_t0 = time.perf_counter()
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx and ctx.triggered else None
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=
        {
            'color': '#FFFFFF'
        },
        title='Simulation — en attente de paramètres'
    )
    fig_daily = go.Figure()
    fig_daily.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=
        {
            'color': '#FFFFFF'
        },
        title='Equity journalier — en attente de paramètres'
    )

    # Réhydratation depuis le store lors du retour sur la page
    if trigger_id and trigger_id.startswith('sim_results_store.data') and stored_data:
        try:
            # Figures
            fig = go.Figure(stored_data.get('equity_figure', {}))
            fig_daily = go.Figure(stored_data.get('daily_figure', {}))

            # Tableau transactions
            stored_trades = stored_data.get('trades', []) or []
            if stored_trades:
                columns = [
                    {"name": "time", "id": "time"},
                    {"name": "action", "id": "action"},
                    {"name": "side", "id": "side"},
                    {"name": "qty", "id": "qty"},
                    {"name": "price", "id": "price"},
                    {"name": "entry_time", "id": "entry_time"},
                    {"name": "entry_price", "id": "entry_price"},
                    {"name": "pnl", "id": "pnl"},
                    {"name": "A_price_t0-lag", "id": "ref_price_t0_lag"},
                    {"name": "A_price_t0", "id": "ref_price_t0"}
                ]
                table = dash_table.DataTable(
                    data=stored_trades,
                    columns=columns,
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "40vh",
                        "overflowY": "auto"
                    },
                    style_cell={
                        "textAlign": "center",
                        "minWidth": "80px",
                        "width": "120px",
                        "maxWidth": "200px"
                    },
                    style_header={"backgroundColor": "#000000", "color": "#ffffff"},
                    style_data={"backgroundColor": "#1E1E1E", "color": "#ffffff"}
                )
            else:
                table = html.Div('Aucune transaction effectuée sur la période.')

            # Tableau journalier
            daily_table_records = stored_data.get('daily_table_records', []) or []
            daily_columns = stored_data.get('daily_table_columns', []) or [
                { 'name': 'date', 'id': 'date' },
                { 'name': 'equity_end', 'id': 'equity_end' },
                { 'name': 'daily_return_%', 'id': 'daily_return_pct' },
                { 'name': 'buys', 'id': 'buys' },
                { 'name': 'sells', 'id': 'sells' },
                { 'name': 'realized_pnl_day', 'id': 'realized_pnl_day' },
                { 'name': 'cum_pnl', 'id': 'cum_pnl' }
            ]
            if daily_table_records:
                daily_table_component = dash_table.DataTable(
                    data=daily_table_records,
                    columns=daily_columns,
                    style_table={
                        'overflowX': 'auto',
                        'maxHeight': '40vh',
                        'overflowY': 'auto'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'minWidth': '80px',
                        'width': '120px',
                        'maxWidth': '200px'
                    },
                    style_header={
                        'backgroundColor': '#000000',
                        'color': '#ffffff'
                    },
                    style_data={
                        'backgroundColor': '#1E1E1E',
                        'color': '#ffffff'
                    }
                )
            else:
                daily_table_component = html.Div('')

            # Résumé
            summary_items = stored_data.get('summary_items', []) or []
            summary = html.Ul([html.Li(item) for item in summary_items], style={ 'color': '#FFFFFF' })

            return fig, table, summary, fig_daily, daily_table_component, stored_data
        except Exception:
            # En cas de problème de réhydratation, on retombe sur le placeholder
            pass

    if not n_clicks:
        return fig, html.Div('Aucune simulation effectuée.'), html.Div(), fig_daily, html.Div(''), None

    try:
        if sim_mode == 'leadlag' and (not ref_symbol or not trade_symbol):
            fig.update_layout(title='Veuillez sélectionner les actions A et B')
            return fig, html.Div('Actions non sélectionnées.'), html.Div(), fig_daily, html.Div(''), None
        if sim_mode == 'model' and not (ref_symbol and sim_model_id):
            fig.update_layout(title="Veuillez sélectionner l'action et le modèle")
            return fig, html.Div("Paramètres insuffisants."), html.Div(), fig_daily, html.Div(''), None
        # Permettre A et B identiques pour les tests
        # if sim_mode == 'leadlag' and ref_symbol == trade_symbol:
        #     fig.update_layout(title='A et B doivent être différentes')
        #     return fig, html.Div('Actions identiques.'), html.Div(), fig_daily, html.Div(''), None

        # Fenêtre temporelle
        if start_date is None or end_date is None:
            end_dt = pd.Timestamp.today().normalize()
            start_dt = end_dt - pd.Timedelta(days=15)
        else:
            start_dt = pd.to_datetime(start_date).normalize()
            end_dt = pd.to_datetime(end_date).normalize()
        # inclure la fin
        end_dt_exclusive = end_dt + pd.Timedelta(days=1)

        # Paramètres
        lag = int(lag_minutes) if lag_minutes and lag_minutes > 0 else 30
        threshold = float(threshold_pct) if threshold_pct is not None else 0.1
        cash = float(initial_cash) if initial_cash is not None else 10000.0
        per_trade = float(trade_amount) if trade_amount is not None else 1000.0

        # Charger données A et B via services
        symbols = [ref_symbol, trade_symbol] if sim_mode == 'leadlag' else [ref_symbol]
        t_fetch_start = time.perf_counter()
        series_map, perf_fetch = fetch_intraday_series_with_perf(shM, symbols, start_dt, end_dt_exclusive)
        t_fetch_end = time.perf_counter()
        map_keys = list(series_map.keys())
        logging.info(f"Simulation: fetched series keys={map_keys}")

        # Sélection robuste des deux séries (tolère désalignements de clés)
        if sim_mode == 'leadlag':
            series_A = series_map.get(ref_symbol)
            series_B = series_map.get(trade_symbol)
            if series_A is None or series_B is None:
                vals = list(series_map.values())
                if len(vals) >= 2:
                    series_A = vals[0]
                    series_B = vals[1]
                    logging.warning("Simulation: fallback ordre séries (clés manquantes)")

            if series_A is None or series_B is None:
                fig.update_layout(title='Données insuffisantes pour la période')
                detail = f"Intraday manquant: A in keys? {ref_symbol in series_map}, B in keys? {trade_symbol in series_map}, keys={map_keys}"
                logging.error(f"[SIMULATION ERROR] Données intraday insuffisantes. {detail}") # Ajout du log de debug
                return fig, html.Div(f'Données intraday insuffisantes. {detail}'), html.Div(), fig_daily, html.Div(''), None

            t_align_start = time.perf_counter()
            aligned, perf_align = align_minute_with_perf({ref_symbol: series_A, trade_symbol: series_B}, start_dt, end_dt_exclusive)
            t_align_end = time.perf_counter()
            # Filtrage aux heures de marché pour réduire le nombre de points traités
            if buy_start_time and sell_end_time and not aligned.empty:
                try:
                    bh, bm = map(int, str(buy_start_time).split(':'))
                    sh, sm = map(int, str(sell_end_time).split(':'))
                    mins = aligned.index.hour * 60 + aligned.index.minute
                    aligned = aligned[(mins >= (bh * 60 + bm)) & (mins <= (sh * 60 + sm))]
                except Exception:
                    pass
            if aligned.shape[0] < max(10, lag + 1):
                fig.update_layout(title='Trop peu de points minute sur la période')
                return fig, html.Div('Séries trop courtes.'), html.Div(), fig_daily, html.Div(''), None
        else:
            # timewindow ou model (préparation spécifique plus bas pour modèle)
            series_A = series_map.get(ref_symbol)
            if series_A is None:
                # Fallback: utiliser la première série disponible si la clé diffère
                if len(series_map) >= 1:
                    alt_symbol, alt_series = next(iter(series_map.items()))
                    logging.warning(f"Timewindow/Model: fallback série sur {alt_symbol}")
                    ref_symbol = alt_symbol
                    series_A = alt_series
                else:
                    fig.update_layout(title='Données insuffisantes pour la période (A)')
                    return fig, html.Div('Intraday insuffisant pour A.'), html.Div(), fig_daily, html.Div(''), None
            if sim_mode == 'model':
                # Pour le modèle: récupérer DataFrame complet (openPrice, volume) et resampler à la minute
                t_align_start = time.perf_counter()
                df_ref = fetch_intraday_dataframe(shM, ref_symbol, start_dt, end_dt_exclusive)
                if df_ref is None or df_ref.empty:
                    fig.update_layout(title='Aucune donnée (modèle) pour la période')
                    return fig, html.Div("Données insuffisantes."), html.Div(), fig_daily, html.Div(''), None
                df_ref = df_ref.resample('1min').last().ffill().bfill()
                aligned = df_ref[(df_ref.index >= start_dt) & (df_ref.index < end_dt_exclusive)].dropna(how='all')
                t_align_end = time.perf_counter()
                # Filtrage horaires
                if buy_start_time and sell_end_time and not aligned.empty:
                    try:
                        bh, bm = map(int, str(buy_start_time).split(':'))
                        sh, sm = map(int, str(sell_end_time).split(':'))
                        mins = aligned.index.hour * 60 + aligned.index.minute
                        aligned = aligned[(mins >= (bh * 60 + bm)) & (mins <= (sh * 60 + sm))]
                    except Exception:
                        pass
                if aligned.shape[0] < 10:
                    fig.update_layout(title='Trop peu de points minute sur la période (A)')
                    return fig, html.Div('Série A trop courte.'), html.Div(), fig_daily, html.Div(''), None
            else:
                t_align_start = time.perf_counter()
                aligned, perf_align = align_minute_with_perf({ref_symbol: series_A}, start_dt, end_dt_exclusive)
                t_align_end = time.perf_counter()
                # Filtrage aux heures de marché pour réduire le nombre de points traités
                if buy_start_time and sell_end_time and not aligned.empty:
                    try:
                        bh, bm = map(int, str(buy_start_time).split(':'))
                        sh, sm = map(int, str(sell_end_time).split(':'))
                        mins = aligned.index.hour * 60 + aligned.index.minute
                        aligned = aligned[(mins >= (bh * 60 + bm)) & (mins <= (sh * 60 + sm))]
                    except Exception:
                        pass
                if aligned.shape[0] < 10:
                    fig.update_layout(title='Trop peu de points minute sur la période (A)')
                    return fig, html.Div('Série A trop courte.'), html.Div(), fig_daily, html.Div(''), None

        # Backtest via service
        t_bt_start = time.perf_counter()
        if sim_mode == 'leadlag':
            bt = backtest_lag_correlation(
                aligned_prices=aligned,
                ref_symbol=ref_symbol,
                trade_symbol=trade_symbol,
                lag_minutes=lag,
                threshold_pct=threshold,
                initial_cash=cash,
                per_trade_amount=per_trade,
                minutes_before_close=10,
                signal_window_minutes=30,
                buy_start_time=buy_start_time,
                sell_end_time=sell_end_time,
                direction=sim_direction
            )
        elif sim_mode == 'timewindow':
            bt = backtest_time_window(
                aligned_prices=aligned,
                symbol=ref_symbol,
                initial_cash=cash,
                per_trade_amount=per_trade,
                buy_start_time=buy_start_time or '09:30',
                sell_end_time=sell_end_time or '16:00'
            )
        else:
            # Simulation par modèle
            try:
                model = shM.load_model_from_db(sim_model_id)
            except Exception as e:
                fig.update_layout(title=f"Erreur chargement modèle: {e}")
                return fig, html.Div('Erreur modèle.'), html.Div(), fig_daily, html.Div(''), None
            bt = backtest_model_intraday(
                day_aligned_df=aligned[['openPrice', 'volume']] if 'volume' in aligned.columns else aligned[['openPrice']],
                model=model,
                initial_cash=cash,
                per_trade_amount=per_trade,
                k_trades=2,
            )
        t_bt_end = time.perf_counter()
        equity_curve_times = bt['equity_times']
        equity_curve_values = bt['equity_values']
        trades = bt['trades']
        final_portfolio_value = bt['final_value']

        # Downsampling de la courbe d'equity pour l'affichage
        t_ds0 = time.perf_counter()
        total_points = len(equity_curve_times) if equity_curve_times else 0
        max_points = 1000
        step = max(1, total_points // max_points) if total_points > 0 else 1
        ds_times = equity_curve_times[::step] if equity_curve_times else []
        ds_values = equity_curve_values[::step] if equity_curve_values else []
        shown_points = len(ds_times)
        t_ds1 = time.perf_counter()

        # Figure equity via builder (mesurée)
        t_fig0 = time.perf_counter()
        fig = build_equity_figure(sim_mode, ref_symbol, trade_symbol, lag, threshold_pct, buy_start_time, sell_end_time, ds_times, ds_values)
        t_fig1 = time.perf_counter()
        if not equity_curve_times:
            fig.update_layout(title='Aucune donnée pour la courbe de capital')

        # Equity journalier et tableau via builder (mesurés)
        t_daily0 = time.perf_counter()
        fig_daily, daily_table_component, df_daily_reset, avg_daily_return_pct = build_daily_outputs(equity_curve_times, equity_curve_values, trades)
        t_daily1 = time.perf_counter()

        # Table des transactions via builder (mesurée)
        t_table0 = time.perf_counter()
        table = build_trades_table(trades)
        t_table1 = time.perf_counter()

        # Résumé via builder
        perf_build = {
            'downsample_s': float(t_ds1 - t_ds0),
            'equity_points_total': int(total_points),
            'equity_points_shown': int(shown_points),
            'equity_step': int(step),
            'equity_fig_build_s': float(t_fig1 - t_fig0),
        }
        summary_items = build_summary(sim_mode, lag, threshold, buy_start_time, sell_end_time, aligned, initial_cash, final_portfolio_value, trades, avg_daily_return_pct, perf_fetch=perf_fetch, perf_align=perf_align, fetch_s=(t_fetch_end - t_fetch_start), align_s=(t_align_end - t_align_start), bt_s=(t_bt_end - t_bt_start), perf_bt=bt.get('perf', {}), perf_build=perf_build)
        summary = html.Ul([html.Li(item) for item in summary_items], style={ 'color': '#FFFFFF' })

        # Préparer les données pour le store persistant
        try:
            df_trades_records = trades.to_dict('records') if trades else []
        except Exception:
            df_trades_records = []

        # Définition des colonnes du tableau journalier (schéma stable pour réhydratation)
        daily_columns = [
            {
                'name': 'date',
                'id': 'date'
            },
            {
                'name': 'equity_end',
                'id': 'equity_end'
            },
            {
                'name': 'daily_return_%',
                'id': 'daily_return_pct'
            },
            {
                'name': 'buys',
                'id': 'buys'
            },
            {
                'name': 'sells',
                'id': 'sells'
            },
            {
                'name': 'realized_pnl_day',
                'id': 'realized_pnl_day'
            },
            {
                'name': 'cum_pnl',
                'id': 'cum_pnl'
            }
        ]

        t_json0 = time.perf_counter()
        fig_dict = fig.to_dict() if fig else {}
        fig_daily_dict = fig_daily.to_dict() if fig_daily else {}
        t_json1 = time.perf_counter()

        store_payload = {
            'equity_figure': fig_dict,
            'daily_figure': fig_daily_dict,
            'trades': df_trades_records,
            'daily_table_records': df_daily_reset.to_dict('records') if 'df_daily_reset' in locals() else [],
            'daily_table_columns': daily_columns,
            'summary_items': summary_items
        }

        # Mesures post-build
        perf_build.update({
            'daily_outputs_s': float(t_daily1 - t_daily0),
            'trades_table_s': float(t_table1 - t_table0),
            'fig_to_dict_s': float(t_json1 - t_json0),
        })

        cb_t1 = time.perf_counter()
        perf_build['callback_total_s'] = float(cb_t1 - cb_t0)

        return fig, table, summary, fig_daily, daily_table_component, store_payload
    except Exception as e:
        logging.exception('Erreur simulation')
        fig.update_layout(title=f"Erreur simulation: {e}")
        return fig, html.Div('Erreur lors de la simulation.'), html.Div(), fig_daily, html.Div(''), None

