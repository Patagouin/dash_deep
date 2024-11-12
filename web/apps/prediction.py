#prediction.py
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.graph_objs as go
from web.components.navigation import create_navigation  # Importer le composant de navigation

import datetime

from app import app, shM

# Remplacer l'importation de fig par la création directe de la figure
fig = go.FigureWidget()

from Models.trading212 import buy_stock, sell_stock

layout = html.Div([
    # Conteneur fixe pour le bandeau supérieur
    html.Div([
        html.H3('Prediction'),
    ], style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'backgroundColor': 'black',
        'padding': '20px',
        'zIndex': 1000
    }),

    # Conteneur défilable pour le reste du contenu
    html.Div([
        # Section des paramètres du modèle
        html.Div([
            html.H4('Paramètres du Modèle', style={
                'marginBottom': '0px',
                'padding': '0px',
                'color': '#FF8C00'
            }),

            # Container pour tous les contrôles
            html.Div([
                # Sélection des actions
                html.Div([
                    html.Label('Sélection des Actions'),
                    dcc.Dropdown(
                        id='prediction_dropdown',
                        options=[
                            {'label': '{}'.format(stock.symbol), 'value': stock.symbol} 
                            for stock in sorted(shM.dfShares.itertuples(), key=lambda stock: stock.symbol)
                        ],
                        multi=True,
                        placeholder="Sélectionner une action",
                        style={'width': '100%'}
                    ),
                ], style={
                    'marginBottom': '0px',
                    'backgroundColor': '#2E2E2E',
                    'padding': '20px',
                    'borderRadius': '8px'
                }),

                # Section pour les sliders de données (maintenant sur la même ligne)
                html.Div([
                    # Slider pour le nombre de jours
                    html.Div([
                        html.Label('Nombre de jours de données'),
                        html.Div([
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
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style={'padding': '20px 0px'}),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Slider pour le ratio entrainement/test
                    html.Div([
                        html.Label('Ratio Entrainement/Test'),
                        html.Div([
                            dcc.Slider(
                                id='train_test_ratio_slider',
                                min=0.5,
                                max=0.9,
                                value=0.8,
                                marks={
                                    0.5: '50/50',
                                    0.6: '60/40',
                                    0.7: '70/30',
                                    0.8: '80/20',
                                    0.9: '90/10'
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style={'padding': '20px 0px'}),
                    ], style={'flex': '1'}),
                ], style={
                    'display': 'flex',
                    'marginBottom': '20px',
                    'backgroundColor': '#2E2E2E',
                    'padding': '20px',
                    'borderRadius': '8px'
                }),

                # Première ligne : paramètres de base
                html.Div([
                    # Colonne Look Back X
                    html.Div([
                        html.Label('Look Back X'),
                        dcc.Input(
                            id='look_back_x',
                            type='number',
                            value=60,
                            min=1,
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Stride X
                    html.Div([
                        html.Label('Stride X'),
                        dcc.Input(
                            id='stride_x',
                            type='number',
                            value=2,
                            min=1,
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Nombre de Prédictions
                    html.Div([
                        html.Label('Nombre de Prédictions'),
                        dcc.Input(
                            id='nb_y',
                            type='number',
                            value=1,
                            min=1,
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1'}),
                ], style={
                    'display': 'flex',
                    'marginBottom': '20px',
                    'gap': '10px'
                }),

                # Deuxième ligne : hyperparamètres
                html.Div([
                    # Colonne Nombre d'Unités
                    html.Div([
                        html.Label('Nombre d\'Unités (multiple)'),
                        dcc.Dropdown(
                            id='nb_units',
                            options=[{'label': str(units), 'value': units} for units in [32, 64, 128, 256, 512]],
                            value=[64, 128],
                            multi=True
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Nombre de Couches
                    html.Div([
                        html.Label('Nombre de Couches (multiple)'),
                        dcc.Dropdown(
                            id='layers',
                            options=[{'label': str(layers), 'value': layers} for layers in [1, 2, 3]],
                            value=[1, 2],
                            multi=True
                        ),
                    ], style={'flex': '1'}),
                ], style={
                    'display': 'flex',
                    'marginBottom': '20px',
                    'gap': '10px'
                }),

                # Troisième ligne : paramètres d'apprentissage
                html.Div([
                    # Colonne Taux d'Apprentissage
                    html.Div([
                        html.Label('Taux d\'Apprentissage (multiple)'),
                        dcc.Dropdown(
                            id='learning_rate',
                            options=[{'label': str(lr), 'value': lr} for lr in [1e-2, 5e-3, 1e-3]],
                            value=[1e-3, 5e-3],
                            multi=True
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Fonction de Perte
                    html.Div([
                        html.Label('Fonction de Perte'),
                        dcc.Dropdown(
                            id='loss_function',
                            options=[
                                {'label': 'Mean Squared Error', 'value': 'mean_squared_error'},
                                {'label': 'Mean Absolute Error', 'value': 'mean_absolute_error'},
                            ],
                            value='mean_squared_error'
                        ),
                    ], style={'flex': '1'}),
                ], style={
                    'display': 'flex',
                    'marginBottom': '20px',
                    'gap': '10px'
                }),
            ], style={
                'width': '100%',
                'padding': '20px'
            }),

            # Bouton d'entraînement
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
        }),

        # Section de visualisation
        html.Div([
            html.H4('Visualisation des Données', style={
                'marginBottom': '0px',  # Plus d'écart pour le deuxième sous-titre
                'padding': '20px',
                'color': '#FF8C00'
            }),
            
            # Contrôles de visualisation
            html.Div([
                # Date picker
                html.Div([
                    html.Label('Période'),
                    dcc.DatePickerRange(
                        id='date_picker_range',
                        display_format='DD/MM/YY',
                        start_date=datetime.datetime.now()-datetime.timedelta(days=7),
                        end_date=datetime.datetime.now()
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Radio items et Checkbox sur la même ligne
                html.Div([
                    dcc.RadioItems(
                        id='data_type_selector',
                        options=[
                            {'label': 'All Data', 'value': 'all'},
                            {'label': 'Main Hours', 'value': 'main'}
                        ],
                        value='all',
                        inline=True,
                        labelStyle={'marginRight': '20px'},
                        style={'color': '#4CAF50'}
                    ),
                    dcc.Checklist(
                        id='normalize_checkbox',
                        options=[{'label': 'Normalize', 'value': 'normalize'}],
                        value=[],
                        inline=True,
                        className='custom-checkbox',
                        labelStyle={
                            'color': '#4CAF50',
                            'display': 'flex',
                            'alignItems': 'center',
                            'cursor': 'pointer',
                            'padding': '5px 10px',
                            'borderRadius': '4px',
                            'transition': 'background-color 0.3s',
                            'backgroundColor': 'rgba(76, 175, 80, 0.1)',
                            'border': '1px solid #4CAF50'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'marginBottom': '20px',
                    'backgroundColor': '#2E2E2E',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'gap': '50px'  # Espace entre les éléments
                }),

                # Graph container
                html.Div([
                    dcc.Graph(
                        id='stock_graph',
                        figure=fig,
                        config={'scrollZoom': True},
                        style={
                            'backgroundColor': 'black',
                            'height': '50vh'
                        }
                    )
                ], style={
                    'width': '100%',
                    'backgroundColor': 'black',
                    'padding': '20px 0'
                }),
            ], style={
                'padding': '20px'
            }),
        ], style={
            'width': '100%',
            'backgroundColor': '#1E1E1E',
            'borderRadius': '8px',
            'marginBottom': '20px'
        }),

        create_navigation()
    ], style={
        'marginTop': '80px',  # Réduit car le bandeau est plus petit
        'padding': '20px',
        'backgroundColor': 'black',
        'minHeight': 'calc(100vh - 80px)',
        'overflowY': 'auto'
    })
], style={
    'backgroundColor': 'black',
    'minHeight': '100vh'
})

@app.callback(
    Output('training_days_slider', 'max'),
    Output('training_days_slider', 'marks'),
    Input('prediction_dropdown', 'value')
)
def update_days_slider(selected_symbols):
    if not selected_symbols:
        return 365, {
            7: '7j',
            30: '1m',
            90: '3m',
            180: '6m',
            365: '1a'
        }
    
    # Calculer le nombre maximum de jours disponibles pour les actions sélectionnées
    max_days = 0
    for symbol in selected_symbols:
        dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
        if not dfShares.empty:
            dfData = shM.getListDfDataFromDfShares(dfShares)[0]
            if not dfData.empty:
                days_available = (dfData.index.max() - dfData.index.min()).days
                max_days = max(max_days, days_available)
    
    # Arrondir au multiple de 30 supérieur
    max_days = min(((max_days + 29) // 30) * 30, 365)
    
    # Créer les marks dynamiquement
    marks = {
        7: '7j',
        30: '1m',
    }
    
    if max_days >= 90:
        marks[90] = '3m'
    if max_days >= 180:
        marks[180] = '6m'
    if max_days >= 365:
        marks[365] = '1a'
    
    return max_days, marks

# Callback pour le graphique (sans les sliders)
@app.callback(
    Output('stock_graph', 'figure'),
    [Input('prediction_dropdown', 'value'),
     Input('date_picker_range', 'start_date'),
     Input('date_picker_range', 'end_date'),
     Input('data_type_selector', 'value'),
     Input('normalize_checkbox', 'value'),
     Input('stock_graph', 'relayoutData')]
)
def display_stock_graph(values, start_date, end_date, data_type, normalize_value, relayoutData):
    if values is None or len(values) == 0:
        print("No values selected.")
        fig = go.FigureWidget()
        fig.layout.template = 'plotly_dark'
        return fig
    
    # Conversion des dates de début et de fin
    if end_date is not None:
        dateLast = datetime.datetime.fromisoformat(end_date)
    else:
        dateLast = datetime.datetime.now()

    if start_date is not None:
        dateBegin = datetime.datetime.fromisoformat(start_date)
    else:
        dateBegin = dateLast - datetime.timedelta(days=7)

    # Création d'une figure vide avec un thème sombre
    fig = go.FigureWidget()
    fig.layout.template = 'plotly_dark'
    fig.layout.hovermode = "x unified"
    fig.update_layout(
        hoverlabel=dict(
            bgcolor='#FFFFFF',
            font=dict(color='#000000'),
            bordercolor='#FFFFFF'
        ),
        legend=dict(
            font=dict(color='#FFFFFF'),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#FFFFFF'
        )
    )
    fig.layout.title = 'Shares quots'
    
    # Récupération des données des actions sélectionnées
    dfShares = shM.getRowsDfByKeysValues(['symbol']*len(values), values, op='|')

    if dfShares.empty:
        print("No data found for the selected symbols.")
        return fig

    # Créer un ensemble pour suivre les symboles déjà ajoutés
    added_symbols = set()

    # Récupération des données pour chaque action dans la plage de dates sélectionnée
    listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)

    # Ajout des données au graphique
    for i, dfData in enumerate(listDfData):
        if not dfData.empty and values[i] not in added_symbols:
            added_symbols.add(values[i])

            # Get the open and close market times for the current stock
            open_time = dfShares.iloc[i].openMarketTime
            close_time = dfShares.iloc[i].closeMarketTime

            # Filter data based on the selected data type
            if data_type == 'main':
                dfData = dfData.between_time(open_time.strftime('%H:%M'), close_time.strftime('%H:%M'))

            # Normalisation si nécessaire
            toNormalize = 'normalize' in normalize_value
            if toNormalize and dfData['openPrice'][0] > 0:
                dfData['openPrice'] /= dfData['openPrice'][0]

            # Convertir les données en format compatible avec JSON
            x_values = dfData.index.astype(str).tolist()
            y_values = dfData['openPrice'].astype(float).tolist()

            fig.add_scatter(
                name=values[i],
                x=x_values,
                y=y_values,
                hovertemplate='%{y:.2f}<extra></extra>'
            )

    # Apply the previous zoom level if available
    if relayoutData and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        fig.update_layout(
            xaxis_range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
        )

        # Filter data based on the zoomed x-axis range
        x_min = relayoutData['xaxis.range[0]']
        x_max = relayoutData['xaxis.range[1]']

        # Find the min and max y-values within the zoomed x-axis range
        y_min, y_max = None, None
        for dfData in listDfData:
            zoomed_data = dfData[(dfData.index >= x_min) & (dfData.index <= x_max)]
            if not zoomed_data.empty:
                current_y_min = zoomed_data['openPrice'].min()
                current_y_max = zoomed_data['openPrice'].max()
                if y_min is None or current_y_min < y_min:
                    y_min = current_y_min
                if y_max is None or current_y_max > y_max:
                    y_max = current_y_max

        # Update the y-axis range based on the zoomed data
        if y_min is not None and y_max is not None:
            fig.update_layout(
                yaxis_range=[y_min, y_max]
            )

    # Dans la fonction display_stock_graph, après la création des traces

    # Calculons le nombre de jours entre les dates
    nb_days = (dateLast - dateBegin).days
    
    # Calculons les intervalles
    interval_labels_ms = 86400000 * max(1, nb_days // 20)    # Pour environ 20 labels

    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            gridwidth=1,
            griddash='dot',               # Style pointillé pour la grille
            dtick=interval_labels_ms,      # Intervalle pour les labels
            tickmode='linear',
            tick0=dateBegin,
            tickformat='%d/%m',
            tickfont=dict(
                color='#FFFFFF',
                size=10
            ),
            tickangle=45,
            # Ajout de lignes verticales supplémentaires via shapes
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    )

    # Ajout des lignes verticales supplémentaires via shapes
    interval_grid_ms = 86400000 * max(1, nb_days // 100)     # Pour environ 100 lignes verticales
    shapes = []
    current_date = dateBegin
    while current_date <= dateLast:
        shapes.append(dict(
            type='line',
            xref='x',
            yref='paper',
            x0=current_date,
            x1=current_date,
            y0=0,
            y1=1,
            line=dict(
                color='rgba(128, 128, 128, 0.2)',
                width=1,
                dash='dot'
            )
        ))
        current_date += datetime.timedelta(milliseconds=interval_grid_ms)

    fig.update_layout(shapes=shapes)

    return fig

# Nouveau callback pour l'entraînement du modèle
@app.callback(
    Output('training-results', 'children'),  # Il faudra ajouter ce div dans le layout
    [Input('train_button', 'n_clicks')],
    [State('prediction_dropdown', 'value'),
     State('training_days_slider', 'value'),
     State('train_test_ratio_slider', 'value'),
     State('look_back_x', 'value'),
     State('stride_x', 'value'),
     State('nb_y', 'value'),
     State('nb_units', 'value'),
     State('layers', 'value'),
     State('learning_rate', 'value'),
     State('loss_function', 'value')]
)
def train_model(n_clicks, selected_symbols, training_days, train_test_ratio, 
                look_back_x, stride_x, nb_y, nb_units, layers, learning_rate, loss_function):
    if n_clicks is None or n_clicks == 0:
        return ""
    
    # Logique d'entraînement du modèle ici
    # ...
    
    return "Modèle entraîné avec succès!"

print("App object in prediction.py:", id(app))
