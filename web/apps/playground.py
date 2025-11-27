from dash import dcc, html, Input, Output, State
import dash
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from io import StringIO
import logging

from app import app, shM
import os
from web.services.synthetic import generate_synthetic_timeseries, estimate_nb_quotes_per_day
from web.components.navigation import create_navigation, create_page_help

# Configuration du logging pour afficher les messages en console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from web.services.model_strategy import backtest_model_intraday
from web.services.sim_builders import (
    build_equity_figure,
    build_trades_table,
    build_daily_outputs,
    build_summary,
)

# Constantes par défaut pour éviter les magic numbers
DEFAULT_EPOCHS = 5
DEFAULT_LOOK_BACK = 60
DEFAULT_STRIDE = 1
DEFAULT_NB_Y = 5
DEFAULT_FIRST_MINUTES = 60
DEFAULT_UNITS = 64
DEFAULT_LAYERS = 1
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_TRADE_AMOUNT = 1_000.0
DEFAULT_K_TRADES = 2
TRAINING_GRAPH_UPDATE_INTERVAL_SECONDS = 5.0


def _get_symbols_options():
    try:
        df = shM.getAllShares()
        symbols = list(df['symbol'].values) if not df.empty else []
        return [{ 'label': s, 'value': s } for s in symbols]
    except Exception:
        return []


def _default_dates():
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=20)
    return start, today


def layout_content():
    start, end = _default_dates()
    
    # Définition des infobulles (titles) pour réutilisation (Label + Input)
    t_curve = 'Choisir la forme de la série synthétique (tendance, saisonnalité, etc.)'
    t_period = 'Période de génération des données'
    t_open = "Heure d'ouverture du marché (HH:MM)"
    t_close = 'Heure de fermeture du marché (HH:MM)'
    t_price = 'Prix initial de la série'
    t_vol = 'Amplitude aléatoire minute à minute (volatilité)'
    t_trend = 'Force de la tendance directionnelle (pente)'
    t_seas = 'Amplitude de la saisonnalité intra‑journalière'
    t_sine = 'Période (en minutes) de la composante sinusoïdale'
    t_lunch = 'Intensité de l’effet de pause déjeuner (réduction de volatilité)'
    t_noise = 'Bruit additif supplémentaire (aléatoire)'
    t_seed = 'Seed aléatoire pour la reproductibilité (laisser vide pour aléatoire)'
    
    t_lookback = 'Taille de la fenêtre d’entrée (en points/minutes)'
    t_stride = "Pas d'échantillonnage pour la fenêtre d'entrée (ex: 5 = 1 point toutes les 5 min)"
    t_nby = 'Nombre de points futurs à prédire (répartis uniformément sur le reste de la journée)'
    t_predtype = "Type de cible à prédire : Variation (Return) ou Prix Normalisé (Price)"
    t_da = 'Activer la métrique Directional Accuracy (pourcentage de bonnes directions)'
    t_first = "Nombre de minutes d'observation en début de journée (Input du modèle)"
    t_units = 'Nombre de neurones par couche LSTM'
    t_layers = 'Nombre de couches LSTM empilées'
    t_lr = "Vitesse d'apprentissage (Learning Rate)"
    t_epochs = "Nombre d'itérations complètes sur le jeu d'entraînement"
    
    t_symbol = 'Filtrer les modèles sauvegardés par symbole'
    t_saved = 'Sélectionner un modèle déjà entraîné'
    
    t_cash = 'Capital de départ pour la simulation'
    t_trade_amt = 'Montant engagé par trade'
    t_ktrades = 'Nombre maximum de trades simultanés/journaliers'

    help_text = """
### Playground (Bac à Sable)

Cet outil est un laboratoire expérimental pour comprendre et tester le fonctionnement de l'IA sur des données de marché synthétiques.

#### 1. Génération de Courbe
Créez des séries temporelles artificielles pour voir si le modèle est capable d'apprendre des motifs simples.
*   **Type de courbe** : Forme générale de la courbe (Marche aléatoire, Tendance, Saisonnière, Sinusoïdale...).
*   **Période** : Dates de début et de fin de la simulation.
*   **Heure ouverture/fermeture** : Horaires de marché simulés.
*   **Prix initial** : Prix de départ de l'actif.
*   **Volatilité** : Amplitude des variations aléatoires minute à minute.
*   **Trend strength** : Force de la tendance haussière (positif) ou baissière (négatif).
*   **Seasonality amplitude** : Amplitude des cycles répétés chaque jour.
*   **Période sinusoïdale** : Durée d'un cycle complet pour la courbe sinusoïdale.
*   **Lunch effect** : Réduction de la volatilité à la mi-journée (effet "pause déjeuner").
*   **Bruit additionnel** : Ajout de fluctuations purement aléatoires pour corser l'apprentissage.
*   **Seed** : Graine aléatoire pour reproduire exactement la même courbe.

#### 2. Modèle (LSTM) & Backtest
Configurez et entraînez un réseau de neurones récurrents (LSTM) sur la courbe générée.
*   **Paramètres de données** :
    *   `look_back` : Nombre de minutes passées que le modèle observe pour faire une prédiction (mémoire).
    *   `stride` : Écart entre les points observés. Si stride=5, on prend 1 point toutes les 5 min.
    *   `nb_y` (Horizon) : Nombre de points futurs que le modèle doit prédire.
    *   `Premières minutes (obs)` : Période d'observation au début de chaque journée avant que le modèle ne commence à trader. **Doit être >= look_back * stride**.
    *   `Type de prédiction` :
        *   **Retours (Variations)** : Le modèle prédit le % de changement. Plus facile à normaliser mais la courbe reconstruite peut dériver.
        *   **Prix (Normalisé)** : Le modèle prédit le niveau de prix relatif. Mieux pour capturer la forme globale.
*   **Architecture IA** :
    *   `Unités LSTM` : Nombre de "neurones" par couche. Plus il y en a, plus le modèle est complexe.
    *   `Couches` : Nombre d'étages de neurones empilés.
    *   `Learning rate` : Vitesse d'apprentissage. Trop haut = instable, trop bas = lent.
    *   `Epochs` : Nombre de fois où le modèle voit l'ensemble des données pendant l'entraînement.

#### 3. Simulation Financière
Une fois le modèle entraîné, une simulation de trading est lancée.
*   `Capital initial` : Montant du portefeuille au départ.
*   `Montant par trade` : Somme investie à chaque prise de position.
*   `K trades/jour` : Limite du nombre de transactions par jour.

#### 4. Résultats
*   **Série synthétique** : Visualisation de la courbe générée.
*   **Segments** : Comparaison visuelle entre la courbe réelle et les prédictions du modèle (Train en bleu, Test en rouge).
*   **Équité** : Évolution de la valeur du portefeuille.
*   **Historique** : Courbe de Loss (erreur) et d'Accuracy pendant l'entraînement.
"""

    return html.Div([
        create_page_help("Aide Playground", help_text),
        html.H3('Playground', style={ 'color': '#FF8C00' }),

        dcc.Store(id='play_df_store', storage_type='session'),

        html.Div([
            html.Div([
                html.H4('Génération de courbe', style={ 'color': '#FF8C00', 'marginBottom': '8px' }),
                html.Div([
                    html.Label('Type de courbe', title=t_curve),
                    html.Div([
                        dcc.Dropdown(
                            id='play_curve_type',
                            options=[
                                { 'label': 'Random walk', 'value': 'random_walk' },
                                { 'label': 'Trend', 'value': 'trend' },
                                { 'label': 'Seasonal', 'value': 'seasonal' },
                                { 'label': 'Lunch effect', 'value': 'lunch_effect' },
                                { 'label': 'Sinusoidale', 'value': 'sinusoidale' },
                            ],
                            value='random_walk',
                            persistence=True, persistence_type='session',
                            style={ 'width': '100%', 'color': '#FF8C00' }
                        )
                    ], title=t_curve)
                ]),
                html.Div([
                    html.Label('Période', title=t_period),
                    html.Div([
                        dcc.DatePickerRange(
                            id='play_date_range',
                            start_date=start.date(),
                            end_date=end.date(),
                            display_format='YYYY-MM-DD'
                        )
                    ], title=t_period)
                ], style={'marginTop': '8px'}),
                
                html.Div([
                    html.Div([
                        html.Label('Heure ouverture', title=t_open),
                        html.Div(dcc.Input(id='play_open_time', value='09:30', type='text', style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_open),
                    ]),
                    html.Div([
                        html.Label('Heure fermeture', title=t_close),
                        html.Div(dcc.Input(id='play_close_time', value='16:00', type='text', style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_close),
                    ]),
                ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(140px, 1fr))', 'gap': '8px', 'marginTop': '8px' }),
                
                html.Div([
                    html.Div([
                        html.Label('Prix initial', title=t_price),
                        html.Div(dcc.Input(id='play_base_price', value=100.0, type='number', step=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_price),
                    ]),
                    html.Div([
                        html.Label('Volatilité', title=t_vol),
                        html.Div(dcc.Input(id='play_volatility', value=0.001, type='number', step=0.0001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_vol),
                    ]),
                    html.Div([
                        html.Label('Trend strength', title=t_trend),
                        html.Div(dcc.Input(id='play_trend_strength', value=0.0001, type='number', step=0.0001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_trend),
                    ]),
                    html.Div([
                        html.Label('Seasonality amplitude', title=t_seas),
                        html.Div(dcc.Input(id='play_seasonality_amp', value=0.01, type='number', step=0.001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_seas),
                    ]),
                    html.Div([
                        html.Label('Période sinusoïdale (min)', title=t_sine),
                        html.Div(dcc.Input(id='play_sine_period', value=360, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_sine),
                    ]),
                    html.Div([
                        html.Label('Lunch effect strength', title=t_lunch),
                        html.Div(dcc.Input(id='play_lunch_strength', value=0.005, type='number', step=0.001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_lunch),
                    ]),
                    html.Div([
                        html.Label('Bruit additionnel', title=t_noise),
                        html.Div(dcc.Input(id='play_extra_noise', value=0.0, type='number', step=0.001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_noise),
                    ]),
                    html.Div([
                        html.Label('Seed (optionnel)', title=t_seed),
                        html.Div(dcc.Input(id='play_seed', value=None, type='number', style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_seed),
                    ]),
                ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(160px, 1fr))', 'gap': '8px', 'marginTop': '8px' }),
                
                html.Button('Générer la courbe', id='play_generate', n_clicks=0, title='Générer une nouvelle série synthétique', style={ 'width': '100%', 'marginTop': '8px' }),
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),

            html.Div([
                html.H4('Modèle et backtest', style={ 'color': '#FF8C00', 'marginBottom': '8px' }),
                dcc.RadioItems(
                    id='play_model_mode',
                    options=[
                        { 'label': 'Nouveau modèle (LSTM)', 'value': 'new' },
                        { 'label': 'Modèle sauvegardé', 'value': 'saved' },
                    ],
                    value='new',
                    labelStyle={ 'display': 'inline-block', 'marginRight': '12px' },
                ),

                html.Div([
                    html.Div([
                        html.Label('look_back', title=t_lookback),
                        html.Div(dcc.Input(id='play_look_back', value=60, type='number', step=1, min=4, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_lookback),
                    ]),
                    html.Div([
                        html.Label('stride', title=t_stride),
                        html.Div(dcc.Input(id='play_stride', value=1, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_stride),
                    ]),
                    html.Div([
                        html.Label('nb_y (horizon)', title=t_nby),
                        # Slider wrapper
                        html.Div([
                            dcc.Slider(id='play_nb_y', min=1, max=60, step=1, value=5, marks={ 1: '1', 60: '60' }, persistence=True, persistence_type='session'),
                        ], title=t_nby),
                        html.Div(id='play_nb_y_value', style={ 'marginTop': '4px', 'color': '#FFFFFF', 'fontSize': '12px' }),
                    ]),
                    html.Div([
                        html.Label('Type de prédiction', title=t_predtype),
                        html.Div([
                            dcc.RadioItems(
                                id='play_prediction_type',
                                options=[
                                    { 'label': 'Retours (Variations)', 'value': 'return' },
                                    { 'label': 'Prix (Normalisé)', 'value': 'price' },
                                ],
                                value='price',
                                labelStyle={ 'display': 'inline-block', 'marginRight': '12px' },
                                persistence=True, persistence_type='session',
                            ),
                        ], title=t_predtype),
                    ]),
                    html.Div([
                        html.Label('Utiliser Directional Accuracy', title=t_da),
                        html.Div([
                            dcc.RadioItems(
                                id='play_use_directional_accuracy',
                                options=[
                                    { 'label': 'Oui', 'value': True },
                                    { 'label': 'Non', 'value': False },
                                ],
                                value=True,
                                labelStyle={ 'display': 'inline-block', 'marginRight': '12px' },
                                persistence=True, persistence_type='session',
                            ),
                        ], title=t_da),
                    ]),
                    html.Div([
                        html.Label('Premières minutes (obs)', title=t_first),
                        html.Div(dcc.Input(id='play_first_minutes', value=60, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_first),
                    ]),
                    html.Div([
                        html.Label('Unités LSTM', title=t_units),
                        html.Div(dcc.Input(id='play_units', value=64, type='number', step=1, min=4, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_units),
                    ]),
                    html.Div([
                        html.Label('Couches', title=t_layers),
                        html.Div(dcc.Input(id='play_layers', value=1, type='number', step=1, min=1, max=3, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_layers),
                    ]),
                    html.Div([
                        html.Label('Learning rate', title=t_lr),
                        html.Div(dcc.Input(id='play_lr', value=0.001, type='number', step=0.0001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_lr),
                    ]),
                    html.Div([
                        html.Label('Epochs', title=t_epochs),
                        html.Div(dcc.Input(id='play_epochs', value=5, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_epochs),
                    ]),
                ], id='panel_play_new', style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))', 'gap': '8px' }),

                html.Div([
                    html.Div([
                        html.Label('Symbole (pour lister les modèles)', title=t_symbol),
                        html.Div(dcc.Dropdown(id='play_symbol', options=_get_symbols_options(), placeholder='Choisir un symbole', style={ 'width': '100%', 'color': '#FF8C00' }, persistence=True, persistence_type='session'), title=t_symbol),
                    ]),
                    html.Div([
                        html.Label('Modèle sauvegardé', title=t_saved),
                        html.Div(dcc.Dropdown(id='play_saved_model', options=[], placeholder='Choisir un modèle', style={ 'width': '100%', 'color': '#FF8C00' }, persistence=True, persistence_type='session'), title=t_saved),
                    ]),
                ], id='panel_play_saved', style={ 'display': 'none', 'gridTemplateColumns': 'repeat(2, minmax(200px, 1fr))', 'gap': '8px' }),

                html.Hr(),
                html.Div([
                    html.Div([
                        html.Label('Capital initial (€)', title=t_cash),
                        html.Div(dcc.Input(id='play_initial_cash', value=10_000, type='number', step=100, min=0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_cash),
                    ]),
                    html.Div([
                        html.Label('Montant par trade (€)', title=t_trade_amt),
                        html.Div(dcc.Input(id='play_trade_amount', value=1_000, type='number', step=50, min=0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_trade_amt),
                    ]),
                    html.Div([
                        html.Label('K trades/jour', title=t_ktrades),
                        html.Div(dcc.Input(id='play_k_trades', value=2, type='number', step=1, min=1, max=5, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_ktrades),
                    ]),
                ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))', 'gap': '8px' }),

                html.Div([
                    html.Button('Entraîner et backtester', id='play_train_backtest', n_clicks=0, style={ 'width': '100%' }),
                ], id='panel_play_btn_new'),

                html.Div([
                    html.Button('Backtester modèle sauvegardé', id='play_backtest_saved', n_clicks=0, style={ 'width': '100%' }),
                ], id='panel_play_btn_saved', style={ 'display': 'none' }),
                html.Hr(),
                html.Div([
                    html.H4('Suivi entraînement', style={ 'color': '#FF8C00' }),
                    html.Div(id='play_training_progress', style={ 'marginBottom': '8px' }),
                    dcc.Graph(id='play_training_history', style={ 'height': '300px' }, config={ 'responsive': False }),
                ], style={ 'marginTop': '12px' }),
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),
        ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))', 'gap': '12px' }),

        html.Div([
            html.Div([
                html.H4('Série synthétique', style={ 'color': '#FF8C00' }),
                dcc.Graph(id='play_price_graph', style={ 'height': '420px' }, config={ 'responsive': False })
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),
            html.Div([
                html.H4('Segments entraînement / test', style={ 'color': '#FF8C00' }),
                dcc.Graph(id='play_segments_graph', style={ 'height': '320px' }, config={ 'responsive': False }),
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),
            html.Div([
                html.H4('Équité', style={ 'color': '#FF8C00' }),
                dcc.Graph(id='play_equity_graph', style={ 'height': '420px' }, config={ 'responsive': False }),
                html.Div(id='play_trades_table', style={ 'marginTop': '8px' }),
                html.Div(id='play_summary', style={ 'marginTop': '8px' }),
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),
        ], style={ 'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px', 'marginTop': '12px' }),

        create_navigation()
    ], style={ 'backgroundColor': 'black', 'padding': '20px', 'minHeight': '100vh' })


layout = layout_content()


@app.callback(
    [
        Output('panel_play_new', 'style'),
        Output('panel_play_saved', 'style'),
        Output('panel_play_btn_new', 'style'),
        Output('panel_play_btn_saved', 'style'),
    ],
    [Input('play_model_mode', 'value')]
)
def toggle_play_panels(mode):
    show = { 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))', 'gap': '8px' }
    hide = { 'display': 'none' }
    show_btn = { 'display': 'block' }
    if mode == 'saved':
        return hide, { 'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(200px, 1fr))', 'gap': '8px' }, hide, show_btn
    return show, hide, show_btn, hide


@app.callback(
    [
        Output('play_price_graph', 'figure'),
        Output('play_df_store', 'data'),
        Output('play_train_backtest', 'disabled'),
    ],
    [Input('play_generate', 'n_clicks')],
    [
        State('play_curve_type', 'value'),
        State('play_date_range', 'start_date'),
        State('play_date_range', 'end_date'),
        State('play_open_time', 'value'),
        State('play_close_time', 'value'),
        State('play_base_price', 'value'),
        State('play_volatility', 'value'),
        State('play_trend_strength', 'value'),
        State('play_seasonality_amp', 'value'),
        State('play_sine_period', 'value'),
        State('play_lunch_strength', 'value'),
        State('play_extra_noise', 'value'),
        State('play_seed', 'value'),
    ],
    prevent_initial_call=True,
)
def generate_curve(n_clicks, curve_type, start_date, end_date, open_time, close_time, base_price, vol, trend_s, seas_amp, sine_period, lunch_s, extra_noise, seed):
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000', font={ 'color': '#FFFFFF' }, title='Série synthétique — en attente de paramètres', height=400, uirevision='play_price')
    if not n_clicks:
        return fig, None, False
    try:
        df = generate_synthetic_timeseries(
            start_date, end_date,
            market_open=open_time or '09:30',
            market_close=close_time or '16:00',
            base_price=float(base_price or 100.0),
            data_type=str(curve_type or 'random_walk'),
            seed=int(seed) if seed is not None else None,
            volatility=float(vol or 0.001),
            trend_strength=float(trend_s or 0.0),
            seasonality_amplitude=float(seas_amp or 0.0),
            lunch_effect_strength=float(lunch_s or 0.0),
            extra_noise=float(extra_noise or 0.0),
            sine_period_minutes=int(sine_period or 360),
        )
        if df is None or df.empty:
            fig.update_layout(title='Aucune donnée générée (plage/horaires vides)')
            return fig, None, False
        price_fig = go.Figure(data=[go.Scatter(x=df.index, y=df['openPrice'], mode='lines', name='openPrice', line={ 'color': '#FF8C00' })])
        price_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Série synthétique — openPrice', height=400, uirevision='play_price')
        # Ne stocker que le prix; ignorer le volume
        store = df[['openPrice']].to_json(date_format='iso', orient='split')
        return price_fig, store, False
    except Exception as e:
        fig.update_layout(title=f'Erreur génération: {e}', height=400, uirevision='play_price')
        return fig, None, False


@app.callback(
    [
        Output('play_saved_model', 'options'),
        Output('play_saved_model', 'value')
    ],
    [Input('play_symbol', 'value')]
)
def populate_saved_models(symbol):
    if not symbol:
        return [], None
    try:
        rows = shM.list_models_for_symbol(symbol)
        options = []
        for mid, date_val, train_s, test_s in rows:
            label = f"{mid} — {str(date_val)[:19]} — train={train_s if train_s is not None else '-'} test={test_s if test_s is not None else '-'}"
            options.append({ 'label': label, 'value': mid })
        return options, (options[0]['value'] if options else None)
    except Exception:
        return [], None


def _build_lstm_model(look_back: int, num_features: int, nb_y: int, units: int, layers: int, lr: float, use_directional_accuracy: bool = True, prediction_type: str = 'return') -> tf.keras.Model:
    # Métrique de Directional Accuracy (DA)
    metrics_list = []
    if use_directional_accuracy:
        if prediction_type == 'price':
            def directional_accuracy_metric(y_true, y_pred):
                # DA sur Prix normalisés: compare si le prix va au-dessus/en-dessous du prix de référence (1.0)
                # y_true et y_pred sont des ratios par rapport à la dernière observation (ex: 1.01, 0.99)
                true_dir = tf.sign(y_true - 1.0)
                pred_dir = tf.sign(y_pred - 1.0)
                equal = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)
                return tf.reduce_mean(equal)
        else:
            def directional_accuracy_metric(y_true, y_pred):
                # DA sur retours: compare les signes des variations
                true_dir = tf.sign(y_true)
                pred_dir = tf.sign(y_pred)
                equal = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)
                return tf.reduce_mean(equal)
        
        # Nom stable pour les logs
        try:
            directional_accuracy_metric.__name__ = 'directional_accuracy'
        except Exception:
            pass
        metrics_list.append(directional_accuracy_metric)
    
    inputs = tf.keras.Input(shape=(int(look_back), int(num_features)))
    x = inputs
    for i in range(int(max(1, layers))):
        return_seq = (i != int(layers) - 1)
        x = tf.keras.layers.LSTM(int(units), return_sequences=return_seq, dropout=0.0)(x)
    outputs = tf.keras.layers.Dense(int(nb_y))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss='mse', metrics=metrics_list)
    return model


def _prepare_xy_from_store(store_json: str, look_back: int, stride: int, nb_y: int, first_minutes: int = None, prediction_type: str = 'return'):
    """
    Prépare les batches X et Y pour l'entraînement.
    prediction_type: 'return' (variation relative) ou 'price' (prix normalisé par rapport à la dernière observation)
    """
    if not store_json:
        return None, None, None, None, 0
    df = pd.read_json(StringIO(store_json), orient='split')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    nb_per_day = estimate_nb_quotes_per_day(df)
    if nb_per_day <= 0:
        return None, None, None, None, 0
    # Split train/test par jours (80/20)
    days = df.index.normalize().unique()
    if len(days) < 2:
        split = len(df)
        train_df = df.iloc[:split]
        test_df = df.iloc[0:0]
    else:
        split_idx = int(len(days) * 0.8)
        split_day = days[split_idx - 1]
        train_df = df.loc[df.index.normalize() <= split_day]
        test_df = df.loc[df.index.normalize() > split_day]

    obs_window = int(first_minutes) if first_minutes is not None and first_minutes > 0 else int(look_back * stride)

    def create_xy(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        if dataset is None or dataset.empty:
            return np.zeros((0, look_back, 1), dtype=float), np.zeros((0, nb_y), dtype=float)
        # Itérer par jour
        norm = dataset.index.normalize()
        days_u = norm.unique()
        for d in days_u:
            day_df = dataset.loc[norm == d, ['openPrice']]
            if day_df.shape[0] < obs_window + max(2, nb_y):
                continue
            # Sélectionner les obs_window premières minutes pour construire la fenêtre d'entrée
            if obs_window < look_back * stride:
                available_points = min(obs_window, day_df.shape[0])
                if available_points < look_back:
                    continue
                step = max(1, available_points // look_back)
                seq = day_df.iloc[0: available_points: step].iloc[:look_back].to_numpy(dtype=float)
            else:
                step = max(1, obs_window // look_back)
                seq = day_df.iloc[0: obs_window: step].iloc[:look_back].to_numpy(dtype=float)
            
            if seq.shape[0] != look_back:
                continue
            
            base_price = seq[-1, 0]
            if base_price == 0:
                continue
            # Normaliser input
            seq[:, 0] = seq[:, 0] / base_price
            if seq.shape[1] >= 2:
                seq[:, 1] = np.log1p(np.clip(seq[:, 1], a_min=0.0, a_max=None))
            
            remainder = day_df.shape[0] - obs_window
            if remainder <= 0:
                continue
            if remainder <= nb_y:
                continue
            stride_y = remainder // (nb_y + 1)
            if stride_y == 0:
                continue
            offsets = [(j + 1) * stride_y for j in range(nb_y)]
            
            y_vals = []
            prev_price = base_price
            prices_list = [base_price]  # Pour logging
            
            for i, off in enumerate(offsets):
                y_price = float(day_df.iloc[obs_window + off, 0])
                prices_list.append(y_price)
                
                if prediction_type == 'price':
                    # Mode Prix : ratio par rapport au dernier prix connu (base_price)
                    # target = P_future / P_base
                    # Si le prix monte de 5%, target = 1.05
                    val = y_price / base_price
                    y_vals.append(val)
                else:
                    # Mode Return (défaut) : variations relatives pas à pas
                    if i == 0:
                        variation = (y_price / prev_price) - 1.0
                        y_vals.append(variation)
                    else:
                        prev_off = offsets[i - 1]
                        prev_price_iter = float(day_df.iloc[obs_window + prev_off, 0])
                        variation = (y_price / prev_price_iter) - 1.0
                        y_vals.append(variation)
                    prev_price = y_price
            
            # Log détaillé seulement pour le premier
            if len(X) == 0:
                logging.info(f"[Prepare XY] Mode={prediction_type}. Exemple premier échantillon:")
                logging.info(f"  Prix: {[f'{p:.2f}' for p in prices_list]}")
                logging.info(f"  Valeurs cibles: {[f'{v:.4f}' for v in y_vals]}")
            
            X.append(seq)
            Y.append(y_vals)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        return X, Y

    trainX, trainY = create_xy(train_df)
    testX, testY = create_xy(test_df)
    return trainX, trainY, testX, testY, nb_per_day


def _build_segments_graph_from_store(store_json: str, look_back: int, stride: int, first_minutes: int, predictions=None, nb_y: int = None, predictions_train=None, prediction_type: str = 'return') -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Segments entraînement / test', height=320, uirevision='play_segments')
    if not store_json:
        return fig
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
    except Exception:
        return fig
    if df is None or df.empty:
        return fig
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(how='any')
    except Exception:
        pass
    if df.index.dtype == object:
        return fig
    idx = df.index
    norm = idx.normalize()
    days = norm.unique()
    if len(days) == 0:
        return fig
    try:
        fig.add_trace(go.Scatter(x=idx, y=df['openPrice'].values, mode='lines', name='Série', line={ 'color': '#888888', 'width': 1 }, opacity=0.35))
    except Exception:
        pass
    split_idx = int(len(days) * 0.8)
    split_day = days[split_idx - 1] if split_idx > 0 else days[0]
    n = len(df)
    masks = {
        'train_obs': np.zeros(n, dtype=bool),
        'train_rest': np.zeros(n, dtype=bool),
        'test_obs': np.zeros(n, dtype=bool),
        'test_rest': np.zeros(n, dtype=bool),
    }
    obs_len_steps = int(max(1, first_minutes or 60))
    for d in days:
        day_mask = (norm == d)
        pos = np.where(day_mask)[0]
        if pos.size == 0:
            continue
        obs_len = min(obs_len_steps, pos.size)
        obs_idx = pos[:obs_len]
        rest_idx = pos[obs_len:]
        if d <= split_day:
            masks['train_obs'][obs_idx] = True
            if rest_idx.size > 0:
                masks['train_rest'][rest_idx] = True
        else:
            masks['test_obs'][obs_idx] = True
            if rest_idx.size > 0:
                masks['test_rest'][rest_idx] = True
    def add_series(name, mask, color, width=2):
        y = np.where(mask, df['openPrice'].values, np.nan)
        fig.add_trace(go.Scatter(x=idx, y=y, mode='lines', name=name, line={ 'color': color, 'width': width }))
    add_series('Train (premières min)', masks['train_obs'], '#1f77b4', 2)
    add_series('Train (reste)', masks['train_rest'], '#2ca02c', 2)
    add_series('Test (premières min)', masks['test_obs'], '#9467bd', 2)
    add_series('Test (reste)', masks['test_rest'], '#d62728', 2)

    def reconstruct_predictions(predictions_data, day_list, color_name, color_hex):
        if predictions_data is not None and len(predictions_data) > 0:
            try:
                pred_idx_flat = []
                pred_values_flat = []
                preds_array = np.array(predictions_data) if isinstance(predictions_data, list) else predictions_data
                preds_flat = preds_array.flatten()
                
                pred_idx_in_flat = 0
                for day_idx, day in enumerate(day_list):
                    day_mask = (norm == day)
                    pos = np.where(day_mask)[0]
                    if pos.size == 0: continue
                    obs_len = min(obs_len_steps, pos.size)
                    if obs_len >= pos.size: continue
                    rest_pos = pos[obs_len:]
                    if len(rest_pos) == 0: continue
                    
                    remainder = len(rest_pos)
                    # Utiliser le nb_y passé en paramètre s'il est valide, sinon fallback
                    nb_y_used = nb_y if nb_y is not None and nb_y > 0 else min(5, remainder)
                    
                    remaining_preds = len(preds_flat) - pred_idx_in_flat
                    if remaining_preds < nb_y_used:
                        nb_y_used = remaining_preds
                    if nb_y_used <= 0: continue
                    
                    base_price = float(df.iloc[pos[obs_len - 1]]['openPrice'])
                    if base_price == 0: continue
                    
                    stride_y = remainder // (nb_y_used + 1) if nb_y_used > 0 else 1
                    offsets = [(j + 1) * stride_y for j in range(min(nb_y_used, remainder))]
                    
                    current_pred_price = base_price
                    for i in range(nb_y_used):
                        if pred_idx_in_flat >= len(preds_flat): break
                        if i < len(offsets) and offsets[i] < len(rest_pos):
                            off = offsets[i]
                            pred_val = float(preds_flat[pred_idx_in_flat])
                            
                            if prediction_type == 'price':
                                # Mode Prix : pred_val est un ratio par rapport à base_price
                                current_pred_price = base_price * pred_val
                            else:
                                # Mode Return : pred_val est une variation relative pas à pas
                                current_pred_price = current_pred_price * (1.0 + pred_val)
                                
                            pred_idx_flat.append(idx[rest_pos[off]])
                            pred_values_flat.append(current_pred_price)
                            pred_idx_in_flat += 1
                
                if pred_values_flat:
                    fig.add_trace(go.Scatter(x=pred_idx_flat, y=pred_values_flat, mode='lines+markers', name=color_name, line={ 'color': color_hex, 'width': 2 }, marker={ 'size': 4 }))
            except Exception:
                pass

    train_days = days[:split_idx]
    test_days = days[split_idx:]
    reconstruct_predictions(predictions_train, train_days, 'Prédiction (train)', '#17becf')
    reconstruct_predictions(predictions, test_days, 'Prédiction (test)', '#FF8C00')
    
    return fig


@app.callback(
    [
        Output('play_segments_graph', 'figure'),
    ],
    [
        Input('play_df_store', 'data'),
        Input('play_look_back', 'value'),
        Input('play_stride', 'value'),
        Input('play_first_minutes', 'value'),
        Input('play_nb_y', 'value'),  # Ajout pour corriger le nombre de points
        Input('play_prediction_type', 'value'), # Pour mettre à jour la visu si on change le type (optionnel, mais mieux)
    ],
    prevent_initial_call=True,
)
def update_segments_graph(store_json, look_back, stride, first_minutes, nb_y, prediction_type):
    try:
        look_back_val = int(look_back or DEFAULT_LOOK_BACK)
        stride_val = int(stride or DEFAULT_STRIDE)
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
        nb_y_val = int(nb_y or DEFAULT_NB_Y)
        pred_type = prediction_type or 'return'
        # On ne redessine pas les prédictions ici car on n'a pas accès aux arrays Y_pred, seulement aux inputs
        # Mais on veut que les segments soient corrects
        fig = _build_segments_graph_from_store(store_json, look_back_val, stride_val, first_minutes_val, None, nb_y_val, None, pred_type)
        return (fig,)
    except Exception:
        return (go.Figure(),)


@app.callback(
    [
        Output('play_nb_y', 'max'),
        Output('play_nb_y', 'value'),
        Output('play_nb_y', 'marks'),
        Output('play_nb_y_value', 'children'),
    ],
    [
        Input('play_first_minutes', 'value'),
        Input('play_open_time', 'value'),
        Input('play_close_time', 'value'),
    ],
    [
        State('play_nb_y', 'value'),
    ]
)
def update_nb_y_slider(first_minutes, open_time, close_time, current_nb_y):
    # Defaults
    try:
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
    except Exception:
        first_minutes_val = DEFAULT_FIRST_MINUTES
    # Parse times HH:MM
    def parse_minutes(hhmm, fallback):
        try:
            parts = str(hhmm or '').split(':')
            h = int(parts[0]); m = int(parts[1])
            return h * 60 + m
        except Exception:
            return fallback
    open_min = parse_minutes(open_time, 9 * 60 + 30)   # 09:30
    close_min = parse_minutes(close_time, 16 * 60)     # 16:00
    day_len = max(0, close_min - open_min)
    remainder = max(0, day_len - max(0, first_minutes_val))
    # Points répartis uniformément: il faut au moins nb_y+1 minutes pour répartir nb_y points
    # Donc max_nb_y = remainder - 1 (minimum), mais on veut au moins 1 point minimum
    max_nb_y = max(1, max(0, remainder - 1))
    try:
        cur_val = int(current_nb_y or 5)
    except Exception:
        cur_val = 5
    new_val = min(max_nb_y, max(1, cur_val))
    # Marks simples (début/fin) pour performance
    marks = { 1: '1', max_nb_y: str(max_nb_y) }
    value_display = html.Span(f"Valeur actuelle: {new_val}", style={ 'fontWeight': 'bold' })
    logging.info(f"[UI] Ajustement slider nb_y — day_len={day_len} remainder={remainder} max={max_nb_y} value={new_val}")
    return max_nb_y, new_val, marks, value_display

@app.callback(
    Output('play_nb_y_value', 'children'),
    Input('play_nb_y', 'value'),
)
def update_nb_y_display(nb_y_value):
    """Met à jour l'affichage de la valeur du slider nb_y quand l'utilisateur le bouge"""
    try:
        val = int(nb_y_value or 5)
    except Exception:
        val = 5
    return html.Span(f"Valeur actuelle: {val}", style={ 'fontWeight': 'bold' })

@app.callback(
    Output('play_first_minutes', 'value'),
    [
        Input('play_look_back', 'value'),
        Input('play_stride', 'value'),
    ],
    [State('play_first_minutes', 'value')],
    prevent_initial_call=True
)
def adjust_first_minutes(look_back, stride, current_first_minutes):
    """
    Ajuste automatiquement le paramètre 'play_first_minutes' pour respecter la contrainte :
    first_minutes >= look_back * stride
    """
    try:
        lb = int(look_back or DEFAULT_LOOK_BACK)
        st = int(stride or DEFAULT_STRIDE)
        fm = int(current_first_minutes or DEFAULT_FIRST_MINUTES)
        
        min_required = lb * st
        
        if fm < min_required:
            logging.info(f"[UI] Auto-adjust: first_minutes ({fm}) < look_back*stride ({min_required}). Updating to {min_required}.")
            return min_required
        
        return dash.no_update
    except Exception:
        return dash.no_update

@app.callback(
    [
        Output('play_equity_graph', 'figure'),
        Output('play_trades_table', 'children'),
        Output('play_summary', 'children'),
        Output('play_segments_graph', 'figure'),
    ],
    [Input('play_train_backtest', 'n_clicks')],
    [
        State('play_df_store', 'data'),
        State('play_look_back', 'value'),
        State('play_stride', 'value'),
        State('play_nb_y', 'value'),
        State('play_first_minutes', 'value'),
        State('play_use_directional_accuracy', 'value'),
        State('play_units', 'value'),
        State('play_layers', 'value'),
        State('play_lr', 'value'),
        State('play_epochs', 'value'),
        State('play_initial_cash', 'value'),
        State('play_trade_amount', 'value'),
        State('play_k_trades', 'value'),
        State('play_prediction_type', 'value'),
    ],
    background=True,
    progress=[
        Output('play_training_progress', 'children'),
        Output('play_training_history', 'figure'),
    ],
    running=[(Output('play_train_backtest', 'disabled'), True, False)],
)
def train_and_backtest(set_progress, n_clicks, store_json, look_back, stride, nb_y, first_minutes, use_directional_accuracy, units, layers, lr, epochs, initial_cash, per_trade, k_trades, prediction_type):
    fig = go.Figure(); fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Équité — en attente de données', height=400, uirevision='play_equity')
    history_fig = go.Figure(); history_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Historique entraînement (Accuracy/Loss)', height=300, uirevision='play_hist')
    empty_seg_fig = go.Figure()
    empty_seg_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Segments entraînement / test', height=320, uirevision='play_segments')
    if not n_clicks:
        return fig, html.Div(''), html.Div(''), empty_seg_fig
    try:
        # Forcer CPU si les drivers GPU posent problème
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                set_progress((html.Div(f"GPU détectés: {len(gpus)} — désactivation pour ce Playground"), history_fig))
                tf.config.set_visible_devices([], 'GPU')
            else:
                set_progress((html.Div("Aucun GPU détecté — utilisation CPU"), history_fig))
        except Exception:
            set_progress((html.Div("Impossible de configurer les devices GPU — fallback CPU"), history_fig))
        # Préparer données
        set_progress((html.Div('Préparation des données...'), history_fig))
        look_back_val = int(look_back or DEFAULT_LOOK_BACK)
        stride_val = int(stride or DEFAULT_STRIDE)
        nb_y_val = int(nb_y or DEFAULT_NB_Y)
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
        units_val = int(units or DEFAULT_UNITS)
        layers_val = int(layers or DEFAULT_LAYERS)
        lr_val = float(lr or DEFAULT_LEARNING_RATE)
        pred_type = prediction_type or 'return'
        
        logging.info(f"[Training] Paramètres batch: look_back={look_back_val}, stride={stride_val}, nb_y={nb_y_val}, first_minutes={first_minutes_val}, type={pred_type}")
        trainX, trainY, testX, testY, nb_per_day = _prepare_xy_from_store(store_json, look_back_val, stride_val, nb_y_val, first_minutes_val, pred_type)
        
        # Logs détaillés sur les cibles Y
        if trainY is not None and len(trainY) > 0:
            logging.info(f"[Training Y] Statistiques cibles train: shape={trainY.shape}")
            logging.info(f"[Training Y] Min={np.min(trainY):.6f}, Max={np.max(trainY):.6f}, Mean={np.mean(trainY):.6f}, Std={np.std(trainY):.6f}")
            if pred_type == 'price':
                ex1_str = ', '.join([f'{v:.4f}' for v in trainY[0]])
                logging.info(f"[Training Y] Exemple premier échantillon (Prix Normalisé): [{ex1_str}]")
            else:
                ex1_str = ', '.join([f'{v*100:.3f}%' for v in trainY[0]])
                logging.info(f"[Training Y] Exemple premier échantillon (variations en %): [{ex1_str}]")
            
            # Vérifier si les variations sont très petites
            abs_mean = np.mean(np.abs(trainY))
            logging.info(f"[Training Y] Moyenne des valeurs absolues: {abs_mean:.6f}")
        
        if trainX is None or trainX.shape[0] == 0:
            empty_seg_fig = go.Figure()
            empty_seg_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Segments entraînement / test', height=320, uirevision='play_segments')
            return fig, html.Div("Jeu d'entraînement insuffisant"), html.Div(''), empty_seg_fig
        num_features = trainX.shape[-1]
        # Construire modèle
        use_da = use_directional_accuracy if use_directional_accuracy is not None else True
        model = _build_lstm_model(look_back_val, int(num_features), nb_y_val, units_val, layers_val, lr_val, use_da, pred_type)
        # Callback de progression
        accs, vaccs, losses, vlosses = [], [], [], []

        def _make_hist_fig():
            # Figure avec deux axes Y: gauche = Loss, droite = DA (0..1)
            fig_h = go.Figure()
            # Traces Loss (axe gauche)
            if losses:
                fig_h.add_trace(go.Scatter(x=list(range(1, len(losses)+1)), y=losses, mode='lines+markers', name='Loss (train)', line={ 'color': '#2ca02c' }, yaxis='y'))
            if vlosses:
                fig_h.add_trace(go.Scatter(x=list(range(1, len(vlosses)+1)), y=vlosses, mode='lines+markers', name='Loss (val)', line={ 'color': '#d62728' }, yaxis='y'))
            # Traces DA (axe droit)
            if accs:
                fig_h.add_trace(go.Scatter(x=list(range(1, len(accs)+1)), y=accs, mode='lines+markers', name='DA (train)', line={ 'color': '#1f77b4' }, yaxis='y2'))
            if vaccs:
                fig_h.add_trace(go.Scatter(x=list(range(1, len(vaccs)+1)), y=vaccs, mode='lines+markers', name='DA (val)', line={ 'color': '#ff7f0e' }, yaxis='y2'))
            # Échelles dynamiques
            if (losses or vlosses):
                all_loss = []
                if losses: all_loss += list(losses)
                if vlosses: all_loss += list(vlosses)
                lmin = float(np.nanmin(all_loss)) if all_loss else 0.0
                lmax = float(np.nanmax(all_loss)) if all_loss else 1.0
                pad = (lmax - lmin) * 0.2 if (lmax > lmin) else (0.1 if lmax == 0 else abs(lmax) * 0.2)
                y_cfg = { 'rangemode': 'tozero', 'range': [max(0.0, lmin - pad), lmax + pad] }
            else:
                y_cfg = {}
            fig_h.update_layout(
                template='plotly_dark',
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font={ 'color': '#FFFFFF' },
                title='Historique entraînement (Accuracy/Loss)',
                height=300,
                uirevision='play_hist',
                yaxis=y_cfg,
                yaxis2={ 'overlaying': 'y', 'side': 'right', 'range': [0, 1] }
            )
            return fig_h

        num_epochs = int(epochs or DEFAULT_EPOCHS)
        class ProgCB(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs, use_da_metric):
                super().__init__()
                self.last_update_time = time.time()
                self.update_interval = TRAINING_GRAPH_UPDATE_INTERVAL_SECONDS
                self.total_epochs = total_epochs
                self.first_update_done = False
                self.use_da_metric = use_da_metric
                # Conserver la dernière figure générée pour l'utiliser entre deux rafraîchissements
                self.latest_fig = _make_hist_fig()
            
            def on_epoch_begin(self, epoch, logs=None):
                set_progress((html.Div(f"Epoch {epoch+1}/{self.total_epochs}"), self.latest_fig))
            
            def on_epoch_end(self, epoch, logs=None):
                a = None
                va = None
                if self.use_da_metric:
                    a = (logs or {}).get('directional_accuracy')
                    va = (logs or {}).get('val_directional_accuracy')
                    try:
                        a = float(a) if a is not None else None
                    except Exception:
                        a = None
                    try:
                        va = float(va) if va is not None else None
                    except Exception:
                        va = None
                
                l = (logs or {}).get('loss')
                vl = (logs or {}).get('val_loss')
                try:
                    l = float(l) if l is not None else None
                except Exception:
                    l = None
                try:
                    vl = float(vl) if vl is not None else None
                except Exception:
                    vl = None
                
                if a is not None: accs.append(a)
                if va is not None: vaccs.append(va)
                if l is not None: losses.append(l)
                if vl is not None: vlosses.append(vl)
                
                # Mettre à jour le graphe au premier epoch ou toutes les 5 secondes
                current_time = time.time()
                time_since_last_update = current_time - self.last_update_time
                should_update_graph = (
                    not self.first_update_done or  # Forcer la première mise à jour
                    (time_since_last_update >= self.update_interval)
                )
                
                da_msg = ''
                if self.use_da_metric and (a is not None or va is not None):
                    da_msg = f" - DA={'' if a is None else f'{a*100:.1f}%'} {'' if va is None else f'| val={va*100:.1f}%'}"
                elif l is not None or vl is not None:
                    da_msg = f" - Loss={'' if l is None else f'{l:.4f}'} {'' if vl is None else f'| val={vl:.4f}'}"
                
                if should_update_graph:
                    logging.info(f"[Training Graph] Mise à jour du graphe - Epoch {epoch+1}/{self.total_epochs} - Temps écoulé depuis dernière MAJ: {time_since_last_update:.2f}s")
                    fig_h = _make_hist_fig()
                    # Mémoriser et publier la nouvelle figure
                    self.latest_fig = fig_h
                    set_progress((html.Div(f"Epoch {epoch+1}/{self.total_epochs}{da_msg}"), self.latest_fig))
                    self.last_update_time = current_time
                    self.first_update_done = True
                else:
                    # Mettre à jour seulement le texte sans le graphe
                    logging.debug(f"[Training Graph] Mise à jour texte uniquement - Epoch {epoch+1}/{self.total_epochs} - Temps écoulé: {time_since_last_update:.2f}s")
                    set_progress((html.Div(f"Epoch {epoch+1}/{self.total_epochs}{da_msg}"), self.latest_fig))
            
            def on_train_end(self, logs=None):
                logging.info(f"[Training Graph] Entraînement terminé - {self.total_epochs} epochs complétés")
        # Entraîner
        set_progress((html.Div('Entraînement...'), history_fig))
        prog_cb = ProgCB(num_epochs, use_da)
        logging.info(f"[Training] Début de l'entraînement - {num_epochs} epochs - Directional Accuracy: {'activé' if use_da else 'désactivé'}")
        
        model.fit(trainX, trainY, epochs=num_epochs, validation_data=(testX, testY) if (testX is not None and testX.size) else None, verbose=0, callbacks=[prog_cb])
        # Forcer la mise à jour finale du graphe après l'entraînement
        final_fig = _make_hist_fig()
        
        # Logs détaillés sur les prédictions vs vraies valeurs
        if trainY is not None and len(trainY) > 0:
            y_pred_train = model.predict(trainX[:5], verbose=0)  # Prédire sur les 5 premiers échantillons
            logging.info(f"[Training Prediction] Analyse détaillée des 5 premiers échantillons:")
            for i in range(min(5, len(trainY))):
                y_true_sample = trainY[i]
                y_pred_sample = y_pred_train[i] if y_pred_train.ndim == 2 else y_pred_train
                # Reconstruire les prix pour voir si ça correspond visuellement
                base_price_sample = 100.0  # Prix fictif pour l'exemple
                true_prices = [base_price_sample]
                pred_prices = [base_price_sample]
                
                if pred_type == 'price':
                    # y_true est un ratio
                    for j in range(len(y_true_sample)):
                        true_prices.append(base_price_sample * y_true_sample[j])
                        pred_prices.append(base_price_sample * y_pred_sample[j])
                else:
                    # y_true est un return
                    for j in range(len(y_true_sample)):
                        true_prices.append(true_prices[-1] * (1 + y_true_sample[j]))
                        pred_prices.append(pred_prices[-1] * (1 + y_pred_sample[j]))
                        
                logging.info(f"[Training Prediction] Échantillon {i+1}:")
                logging.info(f"    Vrais prix (reconstruits): {[f'{p:.2f}' for p in true_prices]}")
                logging.info(f"    Prix prédits (reconstruits): {[f'{p:.2f}' for p in pred_prices]}")
        
        logging.info(f"[Training Graph] Mise à jour finale du graphe après {num_epochs} epochs")
        set_progress((html.Div(f'Entraînement terminé ({num_epochs} epochs)'), final_fig))
        # Backtest
        set_progress((html.Div('Backtest...'), final_fig))
        df = pd.read_json(StringIO(store_json), orient='split')
        initial_cash_val = float(initial_cash or DEFAULT_INITIAL_CASH)
        per_trade_val = float(per_trade or DEFAULT_TRADE_AMOUNT)
        k_trades_val = int(k_trades or DEFAULT_K_TRADES)
        bt = backtest_model_intraday(df[['openPrice', 'volume']] if 'volume' in df.columns else df[['openPrice']], model=model, initial_cash=initial_cash_val, per_trade_amount=per_trade_val, k_trades=k_trades_val)
        equity_curve_times = bt['equity_times']; equity_curve_values = bt['equity_values']; trades = bt['trades']; final_value = bt['final_value']
        # Figures finales
        eq_fig = build_equity_figure('model', 'SYNTH', None, None, None, None, None, equity_curve_times, equity_curve_values)
        eq_fig.update_layout(title='Équité', height=400, uirevision='play_equity')
        # Directional accuracy final (si test dispo) et récupération des prédictions
        y_pred = None
        predictions_flat = None
        predictions_train_flat = None
        try:
            # Prédictions sur test
            y_pred = model.predict(testX, verbose=0) if (testX is not None and testX.size) else None
            da = None
            if y_pred is not None and testY is not None and len(y_pred) == len(testY):
                baseline = 1.0 if pred_type == 'price' else 0.0
                true_dir = np.sign(testY - baseline)
                pred_dir = np.sign(y_pred - baseline)
                da = float((true_dir == pred_dir).mean())
                # Aplatir les prédictions pour l'affichage (y_pred est de shape (n_samples, nb_y))
                if y_pred.ndim == 2:
                    predictions_flat = y_pred.flatten().tolist()
                else:
                    predictions_flat = y_pred.tolist()
            
            # Prédictions sur train pour voir l'apprentissage
            y_pred_train = model.predict(trainX, verbose=0) if (trainX is not None and trainX.size) else None
            if y_pred_train is not None:
                if y_pred_train.ndim == 2:
                    predictions_train_flat = y_pred_train.flatten().tolist()
                else:
                    predictions_train_flat = y_pred_train.tolist()
        except Exception:
            da = None
        # Construire le graphe des segments avec les prédictions train et test
        seg_fig = _build_segments_graph_from_store(store_json, look_back_val, stride_val, first_minutes_val, predictions_flat, nb_y_val, predictions_train_flat, pred_type)
        _, _, _, avg_daily_ret = build_daily_outputs(equity_curve_times, equity_curve_values, trades)
        trades_table = build_trades_table(trades)
        summary_items = build_summary('model', None, None, None, None, df, initial_cash_val, float(final_value), trades, avg_daily_ret, perf_bt=bt.get('perf'))
        summary = html.Ul([html.Li(it) for it in summary_items], style={ 'color': '#FFFFFF' })
        return eq_fig, trades_table, summary, seg_fig
    except Exception as e:
        fig.update_layout(title=f'Erreur entraînement/backtest: {e}')
        empty_seg_fig = go.Figure()
        empty_seg_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Segments entraînement / test', height=320, uirevision='play_segments')
        return fig, html.Div(''), html.Div(''), empty_seg_fig
