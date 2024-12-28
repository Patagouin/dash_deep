from dash import dcc, html
from app import shM  # Import shM from app.py

def get_parameters_layout():
    return html.Div([
        html.H4('Paramètres du Modèle', style={
            'marginBottom': '0px',
            'padding': '0px',
            'color': '#FF8C00'
        }),

        # Container for all controls
        html.Div([
            # Selection of actions and sector (on the same line)
            html.Div([
                # Selection of actions
                html.Div([
                    html.Label('Sélection des Actions', style={'paddingLeft': '0px'}),
                    dcc.Dropdown(
                        id='prediction_dropdown',
                        options=[
                            {'label': 'All', 'value': 'All'},  # Add "All" option
                            *[
                                {'label': '{}'.format(stock.symbol), 'value': stock.symbol} 
                                for stock in shM.dfShares.sort_values(by='symbol').itertuples()
                            ]
                        ],
                        multi=True,
                        placeholder="Sélectionner une action",
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1', 'marginRight': '20px'}),  # Flexbox for equal width

                # Selection of sector
                html.Div([
                    html.Label('Sélection du Secteur', style={'paddingLeft': '0px'}),
                    dcc.Dropdown(
                        id='sector_dropdown',
                        options=[
                            {'label': sector, 'value': sector} 
                            for sector in sorted(shM.dfShares['sector'].fillna('Non défini').unique())
                        ],
                        placeholder="Sélectionner un secteur",
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1'}),  # Flexbox for equal width
            ], style={
                'display': 'flex',  # Flexbox container
                'marginBottom': '20px',
                'backgroundColor': '#2E2E2E',
                'padding': '20px',
                'borderRadius': '8px'
            }),

            # Sliders for data (now on the same line)
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
                        className='custom-slider'
                    ),
                ], style={'flex': '1', 'marginRight': '20px'}),

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
                        className='custom-slider'
                    ),
                ], style={'flex': '1'}),
            ], style={
                'display': 'flex',
                'alignItems': 'center',  # Align items vertically
                'marginBottom': '20px',
                'backgroundColor': '#2E2E2E',
                'padding': '20px',
                'borderRadius': '8px'
            }),

            # Additional hyperparameter controls
            html.Div([
                # Look-back period
                html.Div([
                    html.Label('Look Back (X)', style={'paddingLeft': '10px'}),
                    dcc.Input(
                        id='look_back_x',
                        type='number',
                        value=30,
                        min=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1', 'marginRight': '20px'}),

                # Stride
                html.Div([
                    html.Label('Stride (X)', style={'paddingLeft': '10px'}),
                    dcc.Input(
                        id='stride_x',
                        type='number',
                        value=1,
                        min=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1', 'marginRight': '20px'}),

                # Number of Y outputs
                html.Div([
                    html.Label('Nombre de Y', style={'paddingLeft': '10px'}),
                    dcc.Input(
                        id='nb_y',
                        type='number',
                        value=1,
                        min=1,
                        step=1,
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1'}),
            ], style={
                'display': 'flex',
                'marginBottom': '20px',
                'backgroundColor': '#2E2E2E',
                'padding': '20px',
                'borderRadius': '8px'
            }),

            # Neural network architecture controls
            html.Div([
                # Number of units per layer
                html.Div([
                    html.Label('Nombre d\'Unités', style={'paddingLeft': '10px'}),
                    dcc.Dropdown(
                        id='nb_units',
                        options=[{'label': str(i), 'value': i} for i in [16, 32, 64, 128, 256]],
                        value=64,
                        placeholder="Sélectionner le nombre d'unités",
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1', 'marginRight': '20px'}),

                # Number of layers
                html.Div([
                    html.Label('Nombre de Couches', style={'paddingLeft': '10px'}),
                    dcc.Dropdown(
                        id='layers',
                        options=[{'label': str(i), 'value': i} for i in range(1, 6)],
                        value=2,
                        placeholder="Sélectionner le nombre de couches",
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1', 'marginRight': '20px'}),

                # Learning rate
                html.Div([
                    html.Label('Taux d\'Apprentissage', style={'paddingLeft': '10px'}),
                    dcc.Dropdown(
                        id='learning_rate',
                        options=[
                            {'label': f'{lr:.1e}', 'value': lr} for lr in [0.001, 0.0001, 0.00001]
                        ],
                        value=0.001,
                        placeholder="Sélectionner le taux d'apprentissage",
                        style={'width': '100%'}
                    ),
                ], style={'flex': '1'}),
            ], style={
                'display': 'flex',
                'marginBottom': '20px',
                'backgroundColor': '#2E2E2E',
                'padding': '20px',
                'borderRadius': '8px'
            }),

            # Loss function selection
            html.Div([
                html.Label('Fonction de Perte', style={'paddingLeft': '10px'}),
                dcc.Dropdown(
                    id='loss_function',
                    options=[
                        {'label': 'Mean Squared Error (MSE)', 'value': 'mse'},
                        {'label': 'Mean Absolute Error (MAE)', 'value': 'mae'},
                        {'label': 'Huber Loss', 'value': 'huber_loss'}
                    ],
                    value='mse',
                    placeholder="Sélectionner la fonction de perte",
                    style={'width': '100%'}
                ),
            ], style={
                'marginBottom': '20px',
                'backgroundColor': '#2E2E2E',
                'padding': '20px',
                'borderRadius': '8px'
            }),
        ], style={
            'width': '100%',
            'padding': '20px'
        }),

        # Training button
        html.Div([
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
            html.Div(id='training-results', style={
                'marginTop': '10px',
                'color': '#4CAF50'
            })
        ], style={
            'textAlign': 'center',
            'width': '100%'
        }),
    ], style={
        'width': '100%',
        'padding': '20px',
        'backgroundColor': '#1E1E1E',
        'borderRadius': '8px',
        'marginBottom': '20px'
    }) 