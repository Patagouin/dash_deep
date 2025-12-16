# -*- coding: utf-8 -*-
"""
Panel de paramètres du modèle et d'entraînement.
"""

from dash import dcc, html

from web.apps.model_config import (
    get_model_type_options,
    get_fusion_mode_options,
    MODEL_TYPES,
    TOOLTIPS,
    DEFAULT_FUSION_MODE,
)


def create_model_params_panel(get_symbols_options_func=None):
    """
    Crée le panneau de paramètres du modèle.
    
    Args:
        get_symbols_options_func: Fonction retournant les options de symboles
    
    Returns:
        html.Div contenant le panneau
    """
    # Tooltips locaux
    t_lookback = "Taille de la fenêtre d'entrée (en points/minutes)"
    t_stride = "Pas d'échantillonnage pour la fenêtre d'entrée (ex: 5 = 1 point toutes les 5 min)"
    t_nby = "Nombre de points futurs à prédire (répartis uniformément sur le reste de la journée)"
    t_first = "Nombre de minutes d'observation en début de journée (Input du modèle)"
    t_predtype = "Type de cible à prédire : Variation (Return) ou Prix Normalisé (Price)"
    t_da = "Activer la métrique Directional Accuracy (pourcentage de bonnes directions)"
    t_loss_type = '''Type de fonction de perte (Loss) pour l'entraînement:
• MSE (défaut): Mean Squared Error
• Scaled MSE (×100): MSE multiplié par 100
• MAE: Mean Absolute Error'''
    t_units = "Nombre de neurones par couche LSTM"
    t_layers = "Nombre de couches LSTM empilées"
    t_lr = "Vitesse d'apprentissage (Learning Rate)"
    t_epochs = "Nombre d'itérations complètes sur le jeu d'entraînement"
    t_symbol = "Filtrer les modèles sauvegardés par symbole"
    t_saved = "Sélectionner un modèle déjà entraîné"
    t_embed_dim = TOOLTIPS['embed_dim']
    t_num_heads = TOOLTIPS['num_heads']
    t_trans_layers = TOOLTIPS['transformer_layers']
    t_ff_mult = TOOLTIPS['ff_multiplier']
    t_dropout = TOOLTIPS['dropout']
    
    # Options de symboles
    symbol_options = get_symbols_options_func() if get_symbols_options_func else []
    
    return html.Div([
        html.H4('Modèle et backtest', style={'color': '#FF8C00', 'marginBottom': '8px'}),
        
        # Mode nouveau/sauvegardé
        dcc.RadioItems(
            id='play_model_mode',
            options=[
                {'label': 'Nouveau modèle', 'value': 'new'},
                {'label': 'Modèle sauvegardé (BDD)', 'value': 'saved'},
            ],
            value='new',
            labelStyle={'display': 'inline-block', 'marginRight': '12px'},
        ),

        # Sélecteur de type de modèle
        html.Div([
            html.Label('Type de modèle IA', title=TOOLTIPS['model_type'], style={'fontWeight': 'bold', 'marginTop': '8px'}),
            dcc.Dropdown(
                id='play_model_type',
                options=get_model_type_options(include_gru=False, include_hybrid=True),
                value='lstm',
                persistence=True, persistence_type='session',
                style={'width': '100%', 'color': '#FF8C00'}
            ),
        ], id='panel_model_type_selector', style={'marginBottom': '12px'}),

        # Paramètres de données
        html.Div([
            html.Label('📊 Paramètres de données', style={'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '4px'}),
        ], style={'marginTop': '8px'}),

        html.Div([
            html.Div([
                html.Label('look_back (Window)', title=t_lookback),
                html.Div(dcc.Input(id='play_look_back', value='60', type='text', style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_lookback),
            ]),
            html.Div([
                html.Label('stride', title=t_stride),
                html.Div(dcc.Input(id='play_stride', value=1, type='number', step=1, min=1, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_stride),
            ]),
            html.Div([
                html.Label('nb_y (horizon)', title=t_nby),
                html.Div([
                    dcc.Slider(id='play_nb_y', min=1, max=60, step=1, value=5, marks={1: '1', 60: '60'}, persistence=True, persistence_type='session'),
                ], title=t_nby),
                html.Div(id='play_nb_y_value', style={'marginTop': '4px', 'color': '#FFFFFF', 'fontSize': '12px'}),
            ]),
            html.Div([
                html.Label('Premières minutes (obs)', title=t_first),
                html.Div(dcc.Input(id='play_first_minutes', value=60, type='number', step=1, min=1, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_first),
            ]),
            html.Div([
                html.Label('Type de prédiction', title=t_predtype),
                html.Div([
                    dcc.RadioItems(
                        id='play_prediction_type',
                        options=[
                            {'label': 'Variation (%)', 'value': 'return'},
                            {'label': 'Prix', 'value': 'price'},
                            {'label': 'Signal / Index', 'value': 'signal'},
                        ],
                        value='price',
                        labelStyle={'display': 'inline-block', 'marginRight': '8px'},
                        persistence=True, persistence_type='session',
                    ),
                ], title=t_predtype),
            ]),
            html.Div([
                html.Label('Directional Accuracy', title=t_da),
                html.Div([
                    dcc.RadioItems(
                        id='play_use_directional_accuracy',
                        options=[
                            {'label': 'Oui', 'value': True},
                            {'label': 'Non', 'value': False},
                        ],
                        value=True,
                        labelStyle={'display': 'inline-block', 'marginRight': '8px'},
                        persistence=True, persistence_type='session',
                    ),
                ], title=t_da),
            ]),
            html.Div([
                html.Label('Type de Loss', title=t_loss_type),
                dcc.Dropdown(
                    id='play_loss_type',
                    options=[
                        {'label': 'MSE (défaut)', 'value': 'mse'},
                        {'label': 'Scaled MSE (×100)', 'value': 'scaled_mse'},
                        {'label': 'MAE', 'value': 'mae'},
                    ],
                    value='mse',
                    persistence=True, persistence_type='session',
                    style={'width': '100%', 'color': '#FF8C00'}
                ),
            ]),
        ], id='panel_play_data_params', style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(130px, 1fr))', 'gap': '8px'}),

        # Paramètres LSTM
        html.Div([
            html.Label('🔄 Architecture LSTM', style={'fontWeight': 'bold', 'color': '#1f77b4', 'marginBottom': '4px', 'marginTop': '12px'}),
        ], id='label_lstm_params'),
        html.Div([
            html.Div([
                html.Label('Unités LSTM', title=t_units),
                html.Div(dcc.Input(id='play_units', value=64, type='number', step=1, min=4, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_units),
            ]),
            html.Div([
                html.Label('Couches LSTM', title=t_layers),
                html.Div(dcc.Input(id='play_layers', value=1, type='number', step=1, min=1, max=4, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_layers),
            ]),
        ], id='panel_lstm_params', style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px'}),

        # Paramètres Transformer
        html.Div([
            html.Label('🎯 Architecture Transformer', style={'fontWeight': 'bold', 'color': '#2ca02c', 'marginBottom': '4px', 'marginTop': '12px'}),
        ], id='label_transformer_params', style={'display': 'none'}),
        html.Div([
            html.Div([
                html.Label('Embed dim', title=t_embed_dim),
                html.Div(dcc.Input(id='play_embed_dim', value=64, type='number', step=8, min=16, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_embed_dim),
            ]),
            html.Div([
                html.Label('Num heads', title=t_num_heads),
                html.Div(dcc.Input(id='play_num_heads', value=4, type='number', step=1, min=1, max=16, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_num_heads),
            ]),
            html.Div([
                html.Label('Transformer layers', title=t_trans_layers),
                html.Div(dcc.Input(id='play_transformer_layers', value=2, type='number', step=1, min=1, max=6, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_trans_layers),
            ]),
            html.Div([
                html.Label('FF multiplier', title=t_ff_mult),
                html.Div(dcc.Input(id='play_ff_multiplier', value=4, type='number', step=1, min=1, max=8, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_ff_mult),
            ]),
            html.Div([
                html.Label('Dropout', title=t_dropout),
                html.Div(dcc.Input(id='play_dropout', value=0.1, type='number', step=0.05, min=0.0, max=0.5, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_dropout),
            ]),
        ], id='panel_transformer_params', style={'display': 'none', 'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))', 'gap': '8px'}),

        # Paramètres Hybride
        html.Div([
            html.Label('🔀 Architecture Hybride', style={'fontWeight': 'bold', 'color': '#9467bd', 'marginBottom': '4px', 'marginTop': '12px'}),
        ], id='label_hybrid_params', style={'display': 'none'}),
        html.Div([
            html.Div([
                html.Label('LSTM units', title=t_units),
                html.Div(dcc.Input(id='play_hybrid_lstm_units', value=64, type='number', step=8, min=8, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_units),
            ]),
            html.Div([
                html.Label('LSTM layers', title=t_layers),
                html.Div(dcc.Input(id='play_hybrid_lstm_layers', value=1, type='number', step=1, min=1, max=3, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_layers),
            ]),
            html.Div([
                html.Label('Embed dim', title=t_embed_dim),
                html.Div(dcc.Input(id='play_hybrid_embed_dim', value=64, type='number', step=8, min=16, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_embed_dim),
            ]),
            html.Div([
                html.Label('Trans. heads', title=t_num_heads),
                html.Div(dcc.Input(id='play_hybrid_num_heads', value=4, type='number', step=1, min=1, max=8, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_num_heads),
            ]),
            html.Div([
                html.Label('Trans. layers', title=t_trans_layers),
                html.Div(dcc.Input(id='play_hybrid_trans_layers', value=1, type='number', step=1, min=1, max=4, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_trans_layers),
            ]),
            html.Div([
                html.Label('Fusion mode', title=TOOLTIPS['fusion_mode']),
                dcc.Dropdown(
                    id='play_fusion_mode',
                    options=get_fusion_mode_options(),
                    value=DEFAULT_FUSION_MODE,
                    persistence=True, persistence_type='session',
                    style={'width': '100%', 'color': '#FF8C00'}
                ),
            ]),
            html.Div([
                html.Label('Dropout', title=t_dropout),
                html.Div(dcc.Input(id='play_hybrid_dropout', value=0.1, type='number', step=0.05, min=0.0, max=0.5, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_dropout),
            ]),
        ], id='panel_hybrid_params', style={'display': 'none', 'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))', 'gap': '8px'}),

        # Paramètres d'entraînement
        html.Div([
            html.Label('⚙️ Entraînement', style={'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '4px', 'marginTop': '12px'}),
        ]),
        html.Div([
            html.Div([
                html.Label('Learning rate', title=t_lr),
                html.Div(dcc.Input(id='play_lr', value=0.001, type='number', step=0.0001, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_lr),
            ]),
            html.Div([
                html.Label('Epochs', title=t_epochs),
                html.Div(dcc.Input(id='play_epochs', value=5, type='number', step=1, min=1, style={'width': '100%'}, persistence=True, persistence_type='session'), title=t_epochs),
            ]),
        ], id='panel_play_new', style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px'}),

        # Section Modèles sauvegardés
        html.Div([
            html.Label('📦 Charger depuis la BDD', style={'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '8px'}),
            html.Div([
                html.Div([
                    html.Label('Filtrer par type', title='Filtrer les modèles par architecture'),
                    dcc.Dropdown(
                        id='play_saved_model_type_filter',
                        options=[
                            {'label': 'Tous les types', 'value': 'all'},
                            {'label': f"{MODEL_TYPES['lstm']['icon']} {MODEL_TYPES['lstm']['short_label']}", 'value': 'lstm'},
                            {'label': f"{MODEL_TYPES['transformer']['icon']} {MODEL_TYPES['transformer']['short_label']}", 'value': 'transformer'},
                            {'label': f"{MODEL_TYPES['hybrid']['icon']} {MODEL_TYPES['hybrid']['short_label']}", 'value': 'hybrid'},
                        ],
                        value='all',
                        persistence=True, persistence_type='session',
                        style={'width': '100%', 'color': '#FF8C00'}
                    ),
                ]),
                html.Div([
                    html.Label('Symbole (optionnel)', title=t_symbol),
                    html.Div(dcc.Dropdown(id='play_symbol', options=symbol_options, placeholder='Tous les symboles', style={'width': '100%', 'color': '#FF8C00'}, persistence=True, persistence_type='session'), title=t_symbol),
                ]),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px'}),
            html.Div([
                html.Label('Modèle sauvegardé', title=t_saved),
                html.Div(dcc.Dropdown(id='play_saved_model', options=[], placeholder='Choisir un modèle', style={'width': '100%', 'color': '#FF8C00'}, persistence=True, persistence_type='session'), title=t_saved),
            ], style={'marginTop': '8px'}),
            html.Div(id='play_saved_model_info', style={'marginTop': '8px', 'color': '#888', 'fontSize': '12px'}),
        ], id='panel_play_saved', style={'display': 'none'}),

        # Bouton Entraîner
        html.Div([
            html.Button('🎯 Entraîner le modèle', id='play_train_backtest', n_clicks=0, style={'width': '100%', 'backgroundColor': '#4CAF50', 'padding': '12px', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], id='panel_play_btn_train', style={'marginTop': '12px', 'marginBottom': '8px'}),

        # Bouton Stop (utilisé par le callback background=True)
        html.Div([
            html.Button(
                '⛔ Arrêter l’entraînement',
                id='play_stop_training',
                n_clicks=0,
                style={
                    'display': 'none',
                    'width': '100%',
                    'backgroundColor': '#EF4444',
                    'color': 'white',
                    'padding': '12px',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'border': 'none',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                },
            ),
        ], style={'marginBottom': '8px'}),

        # Choix GPU/CPU (exploité par le worker d’entraînement)
        html.Div([
            html.Label('Accélération', title='Cocher pour demander le GPU si disponible (sinon fallback CPU).'),
            dcc.Checklist(
                id='play_use_gpu',
                options=[
                    {'label': ' Utiliser GPU (si dispo)', 'value': 'gpu'},
                ],
                value=['gpu'],
                persistence=True,
                persistence_type='session',
                style={'color': '#FFF'},
            ),
        ], style={'marginTop': '4px'}),
    ])

