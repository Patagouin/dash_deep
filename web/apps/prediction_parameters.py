from dash import dcc, html
from app import shM  # Import shM from app.py

def get_parameters_layout():
    return html.Div([
        dcc.Interval(id='config_bootstrap', interval=200, n_intervals=0, max_intervals=1),
        html.H4('Choix actions', style={
            'marginBottom': '0px',
            'padding': '4px 0',
            'color': '#FF8C00'
        }),

        # Container for all controls
        html.Div([
            # Ligne avec à gauche: secteur au-dessus puis actions en-dessous
            html.Div([
                # Colonne gauche
                html.Div([
                    # Sélection du Secteur (en premier)
                    html.Div([
                        html.Label('Sélection du Secteur', style={'paddingLeft': '0px'}),
                        dcc.Dropdown(
                            id='sector_dropdown',
                            options=[
                                {'label': sector, 'value': sector}
                                for sector in sorted(shM.dfShares['sector'].fillna('Non défini').unique())
                            ],
                            placeholder="Sélectionner un secteur",
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                    ], style={'marginBottom': '20px'}),

                    # Sélection des Actions (Graphe) supprimée: on utilise la même liste que l'entraînement

                    # Sélection dédiée à l'entraînement du modèle (indépendante)
                    html.Div([
                        html.Label('Choix actions', style={'paddingLeft': '0px'}),
                        dcc.Dropdown(
                            id='train_share_list',
                            options=[
                                *[
                                    {'label': '{}'.format(stock.symbol), 'value': stock.symbol}
                                    for stock in shM.dfShares.sort_values(by='symbol').itertuples()
                                ]
                            ],
                            multi=True,
                            placeholder="Choisir une ou plusieurs actions pour l'entraînement",
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                    ]),
                ], style={
                    'flex': '2',
                    'backgroundColor': '#2E2E2E',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'marginRight': '20px'
                })
            ], style={
                'display': 'grid',
                'gridTemplateColumns': '1fr',
                'gap': '10px',
                'alignItems': 'start',
                'marginBottom': '10px'
            }),

            html.H4('Paramètres du Modèle', style={'margin': '10px 0 0 0','color': '#FF8C00'}),
            html.Div([
            # Sliders for data (now on the same line)
            html.Details([
                html.Summary('Paramètres généraux - Données', style={'cursor': 'pointer', 'color': '#FF8C00'}),
                html.Div([
                # Slider for number of days
                html.Div([
                    html.Label('Nombre de jours de données', style={'color': '#FF8C00', 'paddingBottom': '10px', 'paddingLeft': '10px'}),
                    dcc.Slider(
                        id='training_days_slider',
                        min=7,
                        max=365,
                        value=30,
                        marks={
                            7: '7j',
                            30: '1m',
                            90: '3m',
                            180: '6m',
                            365: '1a'
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                        className='custom-slider',
                        persistence=True, persistence_type='session'
                    ),
                    html.Button('+', id='add_training_days', n_clicks=0, style={'marginTop': '6px'}),
                    dcc.Dropdown(
                        id='training_days_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Jours enregistrés',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 300px', 'minWidth': '260px', 'marginRight': '8px'}),

                # Slider for train-test ratio
                html.Div([
                    html.Label('Ratio Entraînement/Test', style={'color': '#FF8C00', 'paddingBottom': '10px', 'paddingLeft': '10px'}),
                    dcc.Slider(
                        id='train_test_ratio_slider',
                        min=50,
                        max=90,
                        step=5,
                        value=70,
                        marks={i: f'{i}%' for i in range(50, 91, 10)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className='custom-slider',
                        persistence=True, persistence_type='session'
                    ),
                    html.Button('+', id='add_train_ratio', n_clicks=0, style={'marginTop': '6px'}),
                    dcc.Dropdown(
                        id='train_test_ratio_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Ratios enregistrés',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 300px', 'minWidth': '260px'}),
                ], style={'display': 'grid','gridTemplateColumns': 'repeat(auto-fit, minmax(260px, 1fr))','gap': '8px','marginBottom': '0','backgroundColor': 'transparent','padding': '0','borderRadius': '0','alignItems': 'center'})
            ], open=False, style={'backgroundColor': 'transparent', 'margin': '0 0 8px 0'}),

            # Paramètres généraux - Hyperparamètres
            html.Details([
                html.Summary('Paramètres généraux - Hyperparamètres', style={'cursor': 'pointer', 'color': '#FF8C00'}),
                html.Div([
                # Look-back period
                html.Div([
                    html.Label('Look Back (X)', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Input(
                            id='look_back_x',
                            type='number',
                            value=30,
                            min=1,
                            step=1,
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_look_back', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                    dcc.Dropdown(
                        id='look_back_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Options disponibles (fichier)',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 200px', 'minWidth': '180px'}),

                # Stride
                html.Div([
                    html.Label('Stride (X)', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Input(
                            id='stride_x',
                            type='number',
                            value=1,
                            min=1,
                            step=1,
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_stride', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                    dcc.Dropdown(
                        id='stride_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Options disponibles (fichier)',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 200px', 'minWidth': '180px'}),

                # Number of Y outputs
                html.Div([
                    html.Label('Nombre de Y', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Input(
                            id='nb_y',
                            type='number',
                            value=1,
                            min=1,
                            step=1,
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_nb_y', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                    dcc.Dropdown(
                        id='nb_y_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Options disponibles (fichier)',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 180px', 'minWidth': '160px'}),
                # Epochs
                html.Div([
                    html.Label('Epochs', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Input(
                            id='epochs',
                            type='number',
                            value=15,
                            min=1,
                            step=1,
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_epochs', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                    dcc.Dropdown(
                        id='epochs_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Epochs enregistrés',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 200px', 'minWidth': '180px'}),

                # Number of units per layer (moved)
                html.Div([
                    html.Label('Nombre d\'Unités', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Dropdown(
                            id='nb_units',
                            options=[{'label': str(i), 'value': i} for i in [16, 32, 64, 128, 256]],
                            value=64,
                            placeholder="Sélectionner le nombre d'unités",
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_nb_units', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                    dcc.Dropdown(
                        id='nb_units_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Options disponibles (fichier)',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 220px', 'minWidth': '200px'}),

                # Number of layers (moved)
                html.Div([
                    html.Label('Nombre de Couches', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Dropdown(
                            id='layers',
                            options=[{'label': str(i), 'value': i} for i in range(1, 6)],
                            value=2,
                            placeholder="Sélectionner le nombre de couches",
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_layers', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                    dcc.Dropdown(
                        id='layers_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Options disponibles (fichier)',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 200px', 'minWidth': '180px'}),

                # Learning rate (moved)
                html.Div([
                    html.Label('Taux d\'Apprentissage', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Dropdown(
                            id='learning_rate',
                            options=[
                                {'label': f'{lr:.1e}', 'value': lr} for lr in [0.001, 0.0001, 0.00001]
                            ],
                            value=0.001,
                            placeholder="Sélectionner le taux d'apprentissage",
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_learning_rate', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                    dcc.Dropdown(
                        id='learning_rate_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Options disponibles (fichier)',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 220px', 'minWidth': '200px'}),

                # Dropout (toutes archis)
                html.Div([
                    html.Label('Dropout (toutes archis)', style={'paddingLeft': '10px'}),
                    html.Div([
                        dcc.Dropdown(
                            id='transformer_dropout',
                            options=[{'label': f"{v:.1f}", 'value': float(v)} for v in [0.0, 0.1, 0.2, 0.3]],
                            value=0.1,
                            placeholder='Taux de dropout',
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        ),
                        html.Button('+', id='add_transformer_dropout', n_clicks=0, style={'marginLeft': '10px'})
                    ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                    dcc.Dropdown(
                        id='transformer_dropout_saved',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='Options disponibles (fichier)',
                        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 220px', 'minWidth': '200px'}),

                # Méthode de Tuning
                html.Div([
                    html.Label('Méthode de Tuning', style={'paddingLeft': '10px'}),
                    dcc.Dropdown(
                        id='tuning_method',
                        options=[
                            {'label': 'Random Search', 'value': 'random'},
                            {'label': 'Hyperband', 'value': 'hyperband'},
                            {'label': 'Bayesian', 'value': 'bayesian'},
                            {'label': 'Grid Search', 'value': 'grid'}
                        ],
                        value='random',
                        placeholder='Choisir la méthode de tuning',
                        style={'width': '100%', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    )
                ], style={'flex': '0 1 220px', 'minWidth': '200px'}),
                ], style={'display': 'grid','gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))','gap': '8px','marginBottom': '0','backgroundColor': 'transparent','padding': '0','borderRadius': '0','alignItems': 'center'})
            ], open=False, style={'backgroundColor': 'transparent', 'margin': '0 0 8px 0'}),

            # Paramètres Transformer (juste en dessous des généraux)
            html.Details([
                html.Summary('Paramètres Transformer', style={'cursor': 'pointer', 'color': '#FF8C00'}),
                html.Div([
                    # Num Heads
                    html.Div([
                        html.Label('Têtes d\'Attention (Transformer)', style={'paddingLeft': '10px'}),
                        html.Div([
                            dcc.Dropdown(
                                id='transformer_num_heads',
                                options=[{'label': str(h), 'value': h} for h in [2, 4, 8]],
                                value=4,
                                placeholder='Nombre de têtes',
                                style={'width': '100%', 'color': '#FF8C00'},
                                persistence=True, persistence_type='session'
                            ),
                            html.Button('+', id='add_transformer_num_heads', n_clicks=0, style={'marginLeft': '10px'})
                        ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                        dcc.Dropdown(
                            id='transformer_num_heads_saved',
                            options=[],
                            value=[],
                            multi=True,
                            placeholder='Options disponibles (fichier)',
                            style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        )
                    ], style={'flex': '0 1 220px', 'minWidth': '200px'}),

                    # FF multiplier
                    html.Div([
                        html.Label('Facteur Feed-Forward (Transformer)', style={'paddingLeft': '10px'}),
                        html.Div([
                            dcc.Dropdown(
                                id='transformer_ff_multiplier',
                                options=[{'label': str(v), 'value': v} for v in [2, 4]],
                                value=4,
                                placeholder='Facteur de largeur FFN',
                                style={'width': '100%', 'color': '#FF8C00'},
                                persistence=True, persistence_type='session'
                            ),
                            html.Button('+', id='add_transformer_ff_multiplier', n_clicks=0, style={'marginLeft': '10px'})
                        ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                        dcc.Dropdown(
                            id='transformer_ff_multiplier_saved',
                            options=[],
                            value=[],
                            multi=True,
                            placeholder='Options disponibles (fichier)',
                            style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        )
                    ], style={'flex': '0 1 220px', 'minWidth': '200px'}),
                ], style={'display': 'grid','gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))','gap': '8px'})
            ], open=False, style={'backgroundColor': '#1E1E1E', 'padding': '10px', 'borderRadius': '8px', 'marginTop': '8px'}),

            # Stratégie de Trading (collapsable)
            html.Details([
                html.Summary('Stratégie de Trading (intraday)', style={'cursor': 'pointer', 'color': '#FF8C00'}),
                html.Div([
                    # Volume de vente (par trade)
                    html.Div([
                        html.Label('Volume de vente (par trade)', style={'paddingLeft': '10px'}),
                        html.Div([
                            dcc.Input(
                                id='trade_volume',
                                type='number',
                                value=100,
                                min=1,
                                step=1,
                                style={'width': '100%', 'color': '#FF8C00'},
                                persistence=True, persistence_type='session'
                            ),
                            html.Button('+', id='add_trade_volume', n_clicks=0, style={'marginLeft': '10px'})
                        ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                        dcc.Dropdown(
                            id='trade_volume_saved',
                            options=[],
                            value=[],
                            multi=True,
                            placeholder='Volumes enregistrés',
                            style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        )
                    ], style={'flex': '0 1 220px', 'minWidth': '200px'}),

                    # Nombre de K (trades maximum par jour)
                    html.Div([
                        html.Label('Nombre de K (trades max/jour)', style={'paddingLeft': '10px'}),
                        html.Div([
                            dcc.Input(
                                id='k_trades',
                                type='number',
                                value=2,
                                min=1,
                                step=1,
                                style={'width': '100%', 'color': '#FF8C00'},
                                persistence=True, persistence_type='session'
                            ),
                            html.Button('+', id='add_k_trades', n_clicks=0, style={'marginLeft': '10px'})
                        ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                        dcc.Dropdown(
                            id='k_trades_saved',
                            options=[],
                            value=[],
                            multi=True,
                            placeholder='Valeurs K enregistrées',
                            style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        )
                    ], style={'flex': '0 1 220px', 'minWidth': '200px'}),
                ], style={'display': 'grid','gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))','gap': '8px'})
            ], open=False, style={'backgroundColor': '#1E1E1E', 'padding': '10px', 'borderRadius': '8px', 'marginTop': '8px'}),

            # Type de Modèle
            html.Div([
                html.Label('Type de Modèle', style={'paddingLeft': '10px'}),
                html.Div([
                    dcc.Dropdown(
                        id='model_type',
                        options=[
                            {'label': 'LSTM', 'value': 'lstm'},
                            {'label': 'GRU', 'value': 'gru'},
                            {'label': 'Transformer', 'value': 'transformer'}
                        ],
                        value='lstm',
                        placeholder="Sélectionner l'architecture",
                        style={'width': '100%', 'color': '#FF8C00'},
                        persistence=True, persistence_type='session'
                    ),
                    html.Button('+', id='add_model_type', n_clicks=0, style={'marginLeft': '10px'})
                ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                dcc.Dropdown(
                    id='model_type_saved',
                    options=[],
                    value=[],
                    multi=True,
                    placeholder='Options disponibles (fichier)',
                    style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                    persistence=True, persistence_type='session'
                )
            ], style={'marginBottom': '0','backgroundColor': 'transparent','padding': '0','borderRadius': '0'}),

            

            # Fonctions de Perte (collapsable)
            html.Details([
                html.Summary('Fonctions de Perte', style={'cursor': 'pointer', 'color': '#FF8C00'}),
                html.Div([
                    html.Div([
                        html.Label('Catégorie de Perte', style={'paddingLeft': '10px'}),
                        dcc.Dropdown(
                            id='loss_category',
                            options=[{'label': 'Régression', 'value': 'regression'}],
                            value='regression',
                            placeholder='Choisir une catégorie',
                            style={'width': '100%', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        )
                    ], style={'flex': '0 1 220px', 'minWidth': '200px'}),
                    html.Div([
                        html.Label('Fonction de Perte', style={'paddingLeft': '10px'}),
                        html.Div([
                            dcc.Dropdown(
                                id='loss_function',
                                options=[
                                    {'label': 'Mean Squared Error (MSE)', 'value': 'mse'},
                                    {'label': 'Mean Absolute Error (MAE)', 'value': 'mae'},
                                    {'label': 'Huber Loss', 'value': 'huber_loss'}
                                ],
                                value='mse',
                                placeholder="Sélectionner la fonction de perte",
                                style={'width': '100%', 'color': '#FF8C00'},
                                persistence=True, persistence_type='session'
                            ),
                            html.Button('+', id='add_loss', n_clicks=0, style={'marginLeft': '10px'})
                        ], style={'display': 'grid','gridTemplateColumns': 'minmax(0, 1fr) auto','alignItems': 'center','gap': '10px'}),
                        dcc.Dropdown(
                            id='loss_saved',
                            options=[],
                            value=[],
                            multi=True,
                            placeholder='Options disponibles (fichier)',
                            style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
                            persistence=True, persistence_type='session'
                        )
                    ], style={'flex': '0 1 220px', 'minWidth': '200px'}),
                ], style={'display': 'grid','gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))','gap': '8px'})
            ], open=False, style={'backgroundColor': '#1E1E1E', 'padding': '10px', 'borderRadius': '8px', 'marginTop': '8px'}),

            # Sauvegarde / Chargement presets
            html.Div([
                dcc.Input(
                    id='preset_name_input',
                    type='text',
                    placeholder='Nom du preset',
                    style={'marginRight': '10px', 'width': '220px'},
                    persistence=True, persistence_type='session'
                ),
                html.Button('Sauvegarder le preset', id='save_training_preset', n_clicks=0, className='update-button', style={'marginRight': '10px'}),
                html.Button('Modifier le preset', id='edit_training_preset', n_clicks=0, className='update-button', style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='preset_dropdown',
                    options=[],
                    placeholder='Choisir un preset',
                    style={'width': '220px', 'marginRight': '10px', 'color': '#FF8C00'},
                    persistence=True, persistence_type='session'
                ),
                html.Button('Supprimer le preset', id='delete_training_preset', n_clicks=0, className='update-button', style={'marginRight': '10px', 'backgroundColor': '#b00020'}),
                html.Button('Charger le preset', id='load_training_preset', n_clicks=0, className='update-button'),
                html.Div(id='training_config_status', style={'marginLeft': '10px', 'color': '#4CAF50'})
            ], style={'display': 'grid','gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))','gap': '8px','marginBottom': '0','alignItems': 'center'}),
        ], style={'backgroundColor': '#2E2E2E','padding': '8px','borderRadius': '8px','marginBottom': '10px'}),

        # Training button
        html.Div([
            html.Div([
                html.Label('Nom du modèle (optionnel)'),
                dcc.Input(
                    id='model_save_name',
                    type='text',
                    placeholder='Ex: pred_AAPL_ui',
                    style={'width': '100%', 'color': '#FF8C00'},
                    persistence=True, persistence_type='session'
                )
            ], style={'marginBottom': '6px'}),
            html.Button(
                'Entraîner le Modèle',
                id='train_button',
                n_clicks=0,
                style={
                    'padding': '10px 20px',
                    'fontSize': '16px',
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer'
                }
            ),
            # Zone de progression en direct (texte + barre)
            html.Div(id='training_progress', style={'marginTop': '6px','color': '#4CAF50'}),
            html.Div(id='training-results', style={
                'marginTop': '6px',
                'color': '#4CAF50'
            }),
            # Terminal en direct (Socket.IO)
            html.Div([
                html.Div('Terminal (live)', style={'color': '#FF8C00', 'marginBottom': '6px', 'fontWeight': 'bold'}),
                dcc.Textarea(
                    id='terminal_output',
                    value='',
                    readOnly=True,
                    style={
                        'width': '100%',
                        'height': '160px',
                        'backgroundColor': '#000000',
                        'color': '#00FF7F',
                        'fontFamily': 'monospace',
                        'fontSize': '12px',
                        'padding': '8px',
                        'border': '1px solid #333',
                        'borderRadius': '4px',
                        'resize': 'vertical',
                        'whiteSpace': 'pre'
                    }
                ),
                html.Div([
                    html.Div(id='progress_bar', style={'width': '0%','height': '10px','backgroundColor': '#4CAF50','transition': 'width 0.2s','textAlign': 'right','color': 'white','fontSize': '10px'})
                ], style={'width': '100%','height': '10px','backgroundColor': '#555','borderRadius': '4px','overflow': 'hidden','marginTop': '8px'})
            ], style={'marginTop': '10px','backgroundColor': '#0D0D0D','padding': '10px','borderRadius': '8px'})
            ,
            # Modal pour paramètres manquants
            html.Div(id='missing_params_modal')
        ], style={
            'textAlign': 'center',
            'width': '100%'
        }),
        ], style={'width': '100%','padding': '10px'}),
    ], style={
        'width': '100%',
        'padding': '10px',
        'backgroundColor': '#1E1E1E',
        'borderRadius': '8px',
        'marginBottom': '10px'
    }) 